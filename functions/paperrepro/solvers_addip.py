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
import json
import time
from pathlib import Path

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
from functions.paperrepro.tgv import ObjectAmplitudeTGV, ObjectPhaseTGV, phase_from_head
from functions.paperrepro.timing import StageTimer
from functions.paperrepro.sampling import MeasurementSchedule
from functions.paperrepro.input_channels import install_selected_input

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
    if cfg.tgv_phase > 0:
        raise ValueError("AD currently supports amplitude TGV only; use --tgv-phase 0")
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

    tgv = ObjectAmplitudeTGV(cfg, sc.pos, device) if cfg.tgv_amp > 0 else None
    data_energy = sqrtIm.square().mean().detach() if tgv is not None else None
    if tgv is not None:
        tr, tc = tgv.roi
        print(f"[ad] object amplitude TGV2: lambda={cfg.tgv_amp:g}, "
              f"alpha0={cfg.tgv_alpha0:g}, alpha1={cfg.tgv_alpha1:g}, "
              f"inner_steps={cfg.tgv_inner_steps}, eps={cfg.tgv_eps:g}")
        print(f"[ad] TGV domain: nominal illuminated union in "
              f"[{tr.start}:{tr.stop}, {tc.start}:{tc.stop}], "
              f"pixels={int(tgv.mask.sum())}; mean(I)={data_energy.item():.4e}")

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
        data_loss = F.mse_loss(Ua, sqrtIm)
        loss = data_loss
        if tgv is not None:
            # Exact complex magnitude, with finite PyTorch subgradient at O=0.
            # Same normalized amplitude prior/domain/scale as net; no probe term.
            reg, reg_first, reg_second = tgv(O.abs())
            weighted_reg = cfg.tgv_amp * data_energy * reg
            loss = data_loss + weighted_reg
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
            tgv_stats = {}
            if tgv is not None:
                tgv_stats = {
                    "data_loss": data_loss.item(),
                    "relative_data_loss": (data_loss / data_energy.clamp_min(1e-20)).item(),
                    "tgv_amp": reg.item(), "tgv_first": reg_first.item(),
                    "tgv_second": reg_second.item(), "tgv_weighted": weighted_reg.item(),
                }
            hist.append({"it": it + 1, "loss": loss.item(), "real": real,
                         "relerr_p": rp, **m, **tgv_stats})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                _log("ad", it + 1, loss.item(), real, m, rp)
                if tgv is not None:
                    print(f"           data {tgv_stats['data_loss']:.4e} | "
                          f"TGV {tgv_stats['tgv_amp']:.4e} | "
                          f"weighted TGV {tgv_stats['tgv_weighted']:.4e}", flush=True)

    print(f"[ad] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")
    _save(cfg, rec, pc, obj, probe, hist, (rs, cs), pos, tag="ad")
    if tgv is not None:
        tgv.save(Path(cfg.outdir) / "tgv_aux.npz")
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
    input_layer = install_selected_input(net) if cfg.input_policy == "follow_measurements" else None
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

    tgv = ObjectAmplitudeTGV(cfg, sc.pos, device) if cfg.tgv_amp > 0 else None
    tgv_phase = ObjectPhaseTGV(cfg, sc.pos, device) if cfg.tgv_phase > 0 else None
    # Fixed measured-data scale: equivalent to relative amplitude MSE + lambda R.
    # Retain the original data-MSE scale and the exact original path when off.
    data_energy = sqrtIm.square().mean().detach() if tgv is not None or tgv_phase is not None else None
    if tgv is not None:
        tr, tc = tgv.roi
        print(f"[net] object amplitude TGV2: lambda={cfg.tgv_amp:g}, "
              f"alpha0={cfg.tgv_alpha0:g}, alpha1={cfg.tgv_alpha1:g}, "
              f"inner_steps={cfg.tgv_inner_steps}, eps={cfg.tgv_eps:g}")
        print(f"[net] TGV domain: nominal illuminated union in "
              f"[{tr.start}:{tr.stop}, {tc.start}:{tc.stop}], "
              f"pixels={int(tgv.mask.sum())}; mean(I)={data_energy.item():.4e}")
    if tgv_phase is not None:
        tr, tc = tgv_phase.roi
        print(f"[net] object phase wrapped-gradient TGV2: lambda={cfg.tgv_phase:g}, "
              f"alpha0={cfg.tgv_alpha0:g}, alpha1={cfg.tgv_alpha1:g}, "
              f"inner_steps={cfg.tgv_inner_steps}, eps={cfg.tgv_eps:g}; radians")
        print(f"[net] phase TGV domain: [{tr.start}:{tr.stop}, {tc.start}:{tc.stop}], "
              f"pixels={int(tgv_phase.mask.sum())}; mean(I)={data_energy.item():.4e}")

    c_ns = slice(pad_n, pad_n + M)

    def decode():
        a_raw, p_raw = net(x)
        O = make_field(a_raw[0], p_raw[0:2])[c_ns, c_ns]
        # The explicit softplus amplitude bypasses the phase head entirely.
        amp = F.softplus(a_raw[0])[c_ns, c_ns] if tgv is not None else None
        phase = phase_from_head(p_raw[0:2])[c_ns, c_ns] if tgv_phase is not None else None
        return O, _probe_of(mode, Pfix, Pr, Pi, Msup), amp, phase

    opt_net = torch.optim.Adam(net.parameters(), lr=cfg.lr_net,
                               weight_decay=cfg.weight_decay)
    opt_prb = None if mode == "truth" else torch.optim.Adam([Pr, Pi], lr=cfg.lr_probe)

    timer = StageTimer(device, cfg.timing_warmup)
    curriculum = MeasurementSchedule(cfg.measurement_schedule, cfg.grid, cfg.iters,
                                     cfg.measurement_policy, cfg.measurement_seed)
    use_curriculum = bool(cfg.measurement_schedule)
    selection_log, cumulative_patterns, evaluation_patterns = [], 0, 0
    if use_curriculum:
        print(f"[net] measurement curriculum {cfg.measurement_schedule}, "
              f"policy={cfg.measurement_policy}; U-Net input policy={cfg.input_policy}")
        print("[net] Early loss uses subset-wise VarPro (changed early objective); final stage is full.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    hist, t0, rec, pc = [], time.time(), None, None
    for it in range(cfg.iters):
        timer.start(it)
        if use_curriculum:
            indices, stride = curriculum.select(it)
            sel = torch.as_tensor(indices, device=device)
            active_post, target = post[sel], sqrtIm[sel]
            if input_layer is not None:
                input_layer.select(None if len(indices) == cfg.n_pat else sel)
            cumulative_patterns += len(indices)
            selection_log.append({"iteration": it + 1, "stride": stride,
                                  "indices": indices.tolist()})
        else:
            active_post, target = post, sqrtIm
        if cfg.lr_cosine:
            f = 0.5 * (1 + math.cos(PI * it / max(cfg.iters - 1, 1)))
            for g in opt_net.param_groups:
                g["lr"] = cfg.lr_net * f
            if opt_prb is not None:
                for g in opt_prb.param_groups:
                    g["lr"] = cfg.lr_probe * f
        timer.mark("scheduler")
        O, P, amp, phase = decode()
        timer.mark("network_decode")
        Ua = cabs(_fwd(cfg, O, P, active_post, Q))
        # 全局幅度标度 O->aO, P->P/a 是规范自由度。每步解析求最优标量（VarPro），
        # 不 detach —— 该方向梯度恒为 0，自由度被消掉。
        Ua = Ua * ((Ua * target).sum() / (Ua * Ua).sum().clamp_min(1e-20))
        data_loss = F.mse_loss(Ua, target)
        timer.mark("physics_and_data_loss")
        loss = data_loss
        tgv_stats = {}
        if tgv is not None:
            reg, reg_first, reg_second = tgv(amp)
            weighted_reg = cfg.tgv_amp * data_energy * reg
            loss = data_loss + weighted_reg
            timer.mark("amplitude_tgv")
        if tgv_phase is not None:
            phase_reg, phase_first, phase_second = tgv_phase(phase)
            weighted_phase_reg = cfg.tgv_phase * data_energy * phase_reg
            loss = loss + weighted_phase_reg
            timer.mark("phase_tgv")

        opt_net.zero_grad(set_to_none=True)
        if opt_prb is not None:
            opt_prb.zero_grad(set_to_none=True)
        timer.mark("zero_grad")
        loss.backward()
        timer.mark("outer_backward")
        opt_net.step()
        if opt_prb is not None:
            opt_prb.step()
        timer.mark("optimizer")

        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                Ua_eval = Ua
                if use_curriculum and len(indices) < cfg.n_pat:
                    Ua_eval = cabs(_fwd(cfg, O, P, post, Q))
                    Ua_eval = Ua_eval * ((Ua_eval * sqrtIm).sum() /
                                        Ua_eval.square().sum().clamp_min(1e-20))
                    evaluation_patterns += cfg.n_pat
                real = torch.linalg.vector_norm(Ua_eval ** 2 - Icl).item()
                rec = O.detach().cpu().numpy()
                pc = P.detach().cpu().numpy()
                if tgv is not None:
                    tgv_stats = {
                        "data_loss": data_loss.item(),
                        "relative_data_loss": (data_loss / data_energy.clamp_min(1e-20)).item(),
                        "tgv_amp": reg.item(),
                        "tgv_first": reg_first.item(),
                        "tgv_second": reg_second.item(),
                        "tgv_weighted": weighted_reg.item(),
                    }
                if tgv_phase is not None:
                    tgv_stats.update({
                        "data_loss": data_loss.item(),
                        "relative_data_loss": (data_loss / data_energy.clamp_min(1e-20)).item(),
                        "tgv_phase": phase_reg.item(),
                        "tgv_phase_first": phase_first.item(),
                        "tgv_phase_second": phase_second.item(),
                        "tgv_phase_weighted": weighted_phase_reg.item(),
                    })
            m = evaluate(rec[rs, cs], obj[rs, cs])
            rp = probe_relerr(pc, probe)
            curriculum_stats = {}
            if use_curriculum:
                curriculum_stats = {
                    "active_patterns": len(indices), "measurement_stride": stride,
                    "active_input_channels": len(indices) if input_layer is not None else cfg.n_pat,
                    "training_patterns_cumulative": cumulative_patterns,
                    "evaluation_patterns_cumulative": evaluation_patterns,
                    "full_data_loss": F.mse_loss(Ua_eval, sqrtIm).item(),
                    "elapsed_s": time.time() - t0,
                }
            hist.append({"it": it + 1, "loss": loss.item(), "real": real,
                         "relerr_p": rp, **m, **tgv_stats, **curriculum_stats})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                _log("net", it + 1, loss.item(), real, m, rp)
                if tgv is not None:
                    print(f"           data {tgv_stats['data_loss']:.4e} | "
                          f"TGV {tgv_stats['tgv_amp']:.4e} | "
                          f"weighted TGV {tgv_stats['tgv_weighted']:.4e}", flush=True)
                if tgv_phase is not None:
                    print(f"           data {tgv_stats['data_loss']:.4e} | "
                          f"phase TGV {tgv_stats['tgv_phase']:.4e} | "
                          f"weighted phase TGV {tgv_stats['tgv_phase_weighted']:.4e}", flush=True)
            timer.mark("evaluation_and_logging")
        timer.finish()

    print(f"[net] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")
    _save(cfg, rec, pc, obj, probe, hist, (rs, cs), pos, tag="net")
    if tgv is not None:
        tgv.save(Path(cfg.outdir) / "tgv_aux.npz")
    if tgv_phase is not None:
        tgv_phase.save(Path(cfg.outdir) / "tgv_phase_aux.npz")
    timer.save(Path(cfg.outdir) / "net_timing.json")
    if use_curriculum:
        (Path(cfg.outdir) / "measurement_schedule.json").write_text(json.dumps({
            "schedule": cfg.measurement_schedule, "policy": cfg.measurement_policy,
            "seed": cfg.measurement_seed, "input": cfg.input_policy,
            "early_objective": "subset-wise VarPro; not unbiased full-VarPro gradient",
            "steps": selection_log,
        }), encoding="utf-8")
    return hist
