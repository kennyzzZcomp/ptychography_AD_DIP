# -*- coding: utf-8 -*-
"""在【论文的仿真数据】上跑 addip 那条线的两种算法：纯 AD 与 DIP(U-Net)。

与 functions/paperrepro/solvers.py 的 run()（论文原版 ProPtyNet）共用：
    同一份物体/探针真值、同一组扫描位置、同一个 Fresnel 前向、同一份噪声实现、
    同一个探针初值、同一个评价函数与同一块评价 ROI、同一个迭代数。
差别【只有】三处，这三处正是要比的东西：
    物体参数化   ad = 自由复数像素 | net = U-Net(softplus 振幅 + cos/sin 相位)
                 paper = U-Net(LeakyReLU 振幅 + π·tanh 相位)
    数据项       ad/net = 振幅域 ‖|U|-√I‖²      paper = 强度域 Eq.(4)(5) 双掩膜
    探针约束     ad/net = 无                     paper = Loss2 软支撑惩罚

【已知的不对称，报数据时必须说明】paper 那一路有 Loss2 给探针加软支撑，ad/net 没有。
这不是 bug，是算法自带的差别；但如果要单独考察"物体先验"的作用，应当把 Loss2
也加到 ad/net 上再跑一组。
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.addip.losses import cabs
from functions.addip.model import ProPtyUNet as AddipUNet, make_field
from functions.paperrepro.crosstalk import pinhole_mask
from functions.paperrepro.evaluate import evaluate, probe_relerr
from functions.paperrepro.optics import forward_field
from functions.paperrepro.report import _report_device, _save
from functions.paperrepro.scene import build_scene

PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ProPtyNet_paper import Cfg


def _log(tag, it, loss, real, m, rp):
    print(f"  it {it:5d} | loss {loss:.4e} | real {real:.4e} | "
          f"amp SSIM {m['ssim_amp']:.4f} PSNR {m['psnr_amp']:5.2f} | "
          f"phs SSIM {m['ssim_phs']:.4f} | relerr {m['relerr']:.4f} | "
          f"probe err {rp:.4f}", flush=True)


def _fwd(cfg, O, P, post, Q):
    return forward_field(O, P, post, Q, cfg.N, chunk=cfg.fwd_chunk)


def _probe_setup(cfg, sc, probe, device, tag):
    """三种探针参数化的唯一出口。返回 (mode, Pfix, Pr, Pi, Msup)。

    support 档的做法是 P = complex(Pr,Pi) · M。掩膜外的参数梯度恒为 0，
    永远停在初值 —— 与"只参数化掩膜内"数学等价，但不用改前向和存盘格式。
    """
    mode = cfg.probe_mode
    if mode not in ("pixel", "truth", "support"):
        raise ValueError(f"probe_mode={mode!r}，只能是 pixel | truth | support")
    n = cfg.N
    if mode == "truth":
        print(f"[{tag}] 探针 = 【真值且冻结】非盲上界对照，不参与优化")
        return mode, torch.from_numpy(probe.astype(np.complex64)).to(device), None, None, None

    Pr = nn.Parameter(torch.from_numpy(sc.P0.real.copy()).to(device))
    Pi = nn.Parameter(torch.from_numpy(sc.P0.imag.copy()).to(device))
    if mode == "pixel":
        print(f"[{tag}] 探针 = 自由复数像素 {n}²（{2*n*n} 个未知量），"
              f"从第 0 步起与物体联合更新")
        return mode, None, Pr, Pi, None

    m = pinhole_mask(cfg, cfg.probe_support_margin)
    Msup = torch.from_numpy(m.astype(np.float32)).to(device)
    k = int(m.sum())
    print(f"[{tag}] 探针 = 硬支撑掩膜：只在半径 {cfg.probe_support_margin:g}×R "
          f"= {cfg.probe_support_margin*cfg.probe_diam_px/2:.1f} px 的针孔内参数化，外面恒 0")
    print(f"[{tag}]        有效未知量 {2*k} 个（自由像素是 {2*n*n} 个，降到 "
          f"{100*k/(n*n):.2f}%）")
    return mode, None, Pr, Pi, Msup


def _probe_of(mode, Pfix, Pr, Pi, Msup):
    if mode == "truth":
        return Pfix
    P = torch.complex(Pr, Pi)
    return P if Msup is None else P * Msup


# ============================================================================ #
# 纯 AD ptychography
# ============================================================================ #

def run_ad(cfg: Cfg):
    device = cfg.dev()
    torch.manual_seed(cfg.seed)
    _report_device(cfg, device)
    sc = build_scene(cfg, device)
    obj, probe, pos = sc.obj, sc.probe, sc.pos
    (rs, cs), Q, post = sc.roi, sc.Q, sc.post
    sqrtIm, Icl = sc.sqrtIm, sc.Iclt

    M, n = cfg.obj_size, cfg.N
    Or = nn.Parameter(torch.ones((M, M), device=device))
    Oi = nn.Parameter(torch.zeros((M, M), device=device))
    print(f"[ad] 物体 = 自由复数像素 {M}²（{2*M*M} 个未知量），初值 O ≡ 1·exp(i0)")
    mode, Pfix, Pr, Pi, Msup = _probe_setup(cfg, sc, probe, device, "ad")
    groups = [{"params": [Or, Oi], "lr": cfg.lr_obj}]
    if mode != "truth":
        groups.append({"params": [Pr, Pi], "lr": cfg.lr_prb})
    opt = torch.optim.Adam(groups)
    print(f"[ad] 全批量 {cfg.n_pat} 位置/步 × {cfg.iters} 步  "
          f"lr_obj={cfg.lr_obj:g} lr_prb={cfg.lr_prb:g}（不衰减）")

    hist, t0, rec, pc = [], time.time(), None, None
    for it in range(cfg.iters):
        O = torch.complex(Or, Oi)
        P = _probe_of(mode, Pfix, Pr, Pi, Msup)
        Ua = cabs(_fwd(cfg, O, P, post, Q))
        # 全局幅度标度 O->aO, P->P/a 是规范自由度，与 run_net 用同一把尺。
        # 【不加这一行 AD 基本跑不动】数据经过 I/I.max() 全局归一，而初值 O≡1、P≡P0
        # 的预测幅度差着 1~2 个数量级（paper preset 实测 79×）。不消掉这个自由度，
        # 第 0 步 loss 的 99.9% 是纯尺度误差，Adam 的前几百步全在缩幅度而不是重建结构。
        Ua = Ua * ((Ua * sqrtIm).sum() / (Ua * Ua).sum().clamp_min(1e-20))
        loss = F.mse_loss(Ua, sqrtIm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                real = torch.linalg.vector_norm(Ua ** 2 - Icl).item()
                rec = torch.complex(Or, Oi).detach().cpu().numpy()
                pc = P.detach().cpu().numpy()
            m = evaluate(rec[rs, cs], obj[rs, cs])
            rp = probe_relerr(pc, probe)
            hist.append({"it": it + 1, "loss": loss.item(), "real": real,
                         "relerr_p": rp, **m})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                _log("ad", it + 1, loss.item(), real, m, rp)

    print(f"[ad] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")
    _save(cfg, rec, pc, obj, probe, hist, (rs, cs), pos, tag="ad")
    return hist


# ============================================================================ #
# DIP：未训练 U-Net 做物体先验，探针仍是自由像素
# ============================================================================ #

def run_net(cfg: Cfg):
    device = cfg.dev()
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    _report_device(cfg, device)
    sc = build_scene(cfg, device)
    obj, probe, pos = sc.obj, sc.probe, sc.pos
    (rs, cs), Q, post = sc.roi, sc.Q, sc.post
    sqrtIm, Icl, Im = sc.sqrtIm, sc.Iclt, sc.Imt

    # 网络输入：零填充到 net_size 的实测衍射图堆栈，全程固定（与 paper run 同一写法）
    M, n, NS = cfg.obj_size, cfg.N, cfg.net_size
    pad_o = (M - n) // 2
    pad_n = (NS - M) // 2
    x = F.pad(Im[None], (pad_o,) * 4)
    x = F.pad(x, (pad_n, NS - M - pad_n, pad_n, NS - M - pad_n))
    x = x / x.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)

    net = AddipUNet(cfg.n_pat, cfg.base_ch, n_fields=1, ph_ch=2).to(device)
    print(f"[net] U-Net(addip 参数化) {sum(p.numel() for p in net.parameters())/1e6:.2f} M "
          f"参数 | 输入 {tuple(x.shape)}")

    # 相位中性初始化：alpha=0 时第 0 步 O ≡ 1·exp(i0)，与 ad 的初值逐位相同
    _a = float(cfg.obj_init_alpha)
    with torch.no_grad():
        net.head_amp.weight[0].mul_(_a)
        net.head_phs.weight[0:2].mul_(_a)
        net.head_amp.bias[0] = math.log(math.e - 1.0)   # softplus(b) = 1
        net.head_phs.bias[0:2] = 0.0
        net.head_phs.bias[0] = 1.0                      # (c, sn) = (1, 0) -> phi = 0
    print(f"[net] 物体输出头 = 相位中性初始化 alpha={_a:g}"
          + ("（第 0 步 O ≡ 1·exp(i0)，与 ad 初值相同）" if _a == 0 else ""))

    mode, Pfix, Pr, Pi, Msup = _probe_setup(cfg, sc, probe, device, "net")

    c_ns = slice(pad_n, pad_n + M)

    def decode():
        a_raw, p_raw = net(x)
        O = make_field(a_raw[0], p_raw[0:2])[c_ns, c_ns]
        return O, _probe_of(mode, Pfix, Pr, Pi, Msup)

    opt_net = torch.optim.Adam(net.parameters(), lr=cfg.lr_net,
                               weight_decay=cfg.weight_decay)
    opt_prb = None if mode == "truth" else torch.optim.Adam([Pr, Pi], lr=cfg.lr_probe)

    hist, t0, rec, pc = [], time.time(), None, None
    for it in range(cfg.iters):
        if cfg.lr_cosine:
            f = 0.5 * (1 + math.cos(PI * it / max(cfg.iters - 1, 1)))
            for g in opt_net.param_groups:
                g["lr"] = cfg.lr_net * f
            if opt_prb is not None:
                for g in opt_prb.param_groups:
                    g["lr"] = cfg.lr_probe * f
        O, P = decode()
        Ua = cabs(_fwd(cfg, O, P, post, Q))
        # 全局幅度标度 O->aO, P->P/a 是规范自由度。每步解析求最优标量（VarPro），
        # 不 detach —— 该方向梯度恒为 0，自由度被消掉。
        Ua = Ua * ((Ua * sqrtIm).sum() / (Ua * Ua).sum().clamp_min(1e-20))
        loss = F.mse_loss(Ua, sqrtIm)

        opt_net.zero_grad(set_to_none=True)
        if opt_prb is not None:
            opt_prb.zero_grad(set_to_none=True)
        loss.backward()
        opt_net.step()
        if opt_prb is not None:
            opt_prb.step()

        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                real = torch.linalg.vector_norm(Ua ** 2 - Icl).item()
                rec = O.detach().cpu().numpy()
                pc = P.detach().cpu().numpy()
            m = evaluate(rec[rs, cs], obj[rs, cs])
            rp = probe_relerr(pc, probe)
            hist.append({"it": it + 1, "loss": loss.item(), "real": real,
                         "relerr_p": rp, **m})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                _log("net", it + 1, loss.item(), real, m, rp)

    print(f"[net] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")
    _save(cfg, rec, pc, obj, probe, hist, (rs, cs), pos, tag="net")
    return hist
