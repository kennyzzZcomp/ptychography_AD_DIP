# -*- coding: utf-8 -*-
"""两种运行模式：自检 / 重建。"""

from __future__ import annotations

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.paperrepro.evaluate import (evaluate, evaluation_roi, illum_roi,
                                            probe_relerr, seam_diag)
from functions.paperrepro.losses import paper_loss
from functions.paperrepro.model import ProPtyUNet
from functions.paperrepro.optics import forward_ptycho, make_quad_phase
from functions.paperrepro.report import _report_device, _save
from functions.paperrepro.sample import make_positions, make_truth, simulate
from functions.paperrepro.scene import build_scene

PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_paper import Cfg

def run_check(cfg: Cfg):
    ok = lambda c: "OK " if c else "!! "
    print("=" * 78)
    print(f"preset = {cfg.preset}   （论文第 3 节参数：512 探测器 / 15.04µm / 16.5cm / 10×10）")
    print("=" * 78)
    print(f"  λ={cfg.wlength*1e9:.1f}nm  z={cfg.z*100:.2f}cm  Δx2={cfg.det_pixel*1e6:.2f}µm  M={cfg.N}")
    print(f"  -> Δx1 = λz/(M·Δx2) = {cfg.dx1*1e6:.3f} µm            [Eq.(2) 下的采样关系]")
    print(f"  {ok(cfg.dx1 <= cfg.chirp_limit)}chirp 采样: Δx1 ≤ √(λz/M) = {cfg.chirp_limit*1e6:.3f} µm"
          f"   (用了 {100*cfg.dx1/cfg.chirp_limit:.0f}%)")
    print(f"  针孔 {cfg.probe_diam_um:.0f} µm = {cfg.probe_diam_px:.1f} px"
          f"  (占窗口 {100*cfg.probe_diam_px/cfg.N:.1f}%)")
    print(f"  扫描 {cfg.grid}×{cfg.grid}={cfg.n_pat} 点  步长 {cfg.step_px} px = "
          f"{cfg.step_px*cfg.dx1*1e6:.1f} µm")
    print(f"  线性重叠 = 1 - step/D = {100*(1-cfg.step_px/cfg.probe_diam_px):.1f} %")
    print(f"  扫描跨度 {cfg.scan_span}  物体画布 {cfg.obj_size}  "
          f"{ok(cfg.scan_offset >= 0)}余量 {cfg.scan_offset} px")
    print(f"  {ok(cfg.obj_size % 8 == 0)}画布能被 8 整除（Fig.1b 三次池化要求）"
          f"  -> 网络内部再 pad 到 {cfg.net_size}")
    if cfg.preset == "paper":
        print("-" * 78)
        print("论文自身的三处不自洽（不是本代码的 bug，报数据时要说明取了哪一种）:")
        print(f"  1) 正文说重叠 0.7，但 step={cfg.step_px}px / D={cfg.probe_diam_px:.0f}px 给出 "
              f"{100*(1-cfg.step_px/cfg.probe_diam_px):.0f}%")
        print(f"  2) 物体写 612，而 {cfg.N}+({cfg.grid}-1)×{cfg.step_px} = "
              f"{cfg.N+(cfg.grid-1)*cfg.step_px}")
        print("  3) 正文说 zero-pad 到 612 是「网络结构要求」，但 612/8 = 76.5 不是整数，"
              "612 恰恰不满足三次池化")
    print("-" * 78)
    _o, _p, _s1, _rr = make_truth(cfg)
    for name, fld, span in (("物体", _o, cfg.phase_span_obj),
                            ("探针", _p, cfg.phase_span_prb)):
        m = np.abs(fld) > 1e-9
        ph = np.angle(fld)[m]
        rngspan = ph.max() - ph.min()
        need = max(abs(ph.max()), abs(ph.min())) / span
        head = (2 * span - rngspan) / 2
        print(f"  {name}相位: GT 跨度 {rngspan:.3f} rad = {rngspan/PI:.2f}π"
              f" | 网络窗口 ±{span/PI:.2f}π | 需要 |tanh| ≤ {need:.2f}"
              f" | 全局相位余量 ±{head:.2f} rad ({ok(head > 0.5*PI)}宽松)")
    print("-" * 78)

    device = cfg.dev()
    cfg.quad_sign = getattr(cfg, "quad_sign", -1.0)
    obj, probe, S1, rr = make_truth(cfg)
    pos = make_positions(cfg)
    Q = make_quad_phase(cfg, device)
    Ot = torch.from_numpy(obj).to(device); Pt = torch.from_numpy(probe).to(device)
    post = torch.from_numpy(pos).to(device)

    with torch.no_grad():
        I = forward_ptycho(Ot, Pt, post, Q, cfg.N, chunk=8)
    Im = (I / I.max()).clamp(0, 1)
    irs, ics = illum_roi(cfg, probe, pos)
    rs, cs = evaluation_roi(cfg, probe, pos)
    print(f"[F] 评价 ROI = 行 {rs.start}..{rs.stop-1} 列 {cs.start}..{cs.stop-1}"
          f"  ({rs.stop-rs.start}×{cs.stop-cs.start})，画布 {cfg.obj_size}"
          f"  | 自适应照明框 {irs.stop-irs.start}×{ics.stop-ics.start}")
    print(f"[A] 衍射图 {tuple(I.shape)}  动态范围 [{Im.min():.2e}, {Im.max():.2e}]")

    with torch.no_grad():
        I2 = forward_ptycho(Ot, Pt, post, Q, cfg.N, chunk=8)
        rel = ((I2 / I2.max()).clamp(0, 1) - Im).norm() / Im.norm()
    print(f"[B] 前向可重复性                 {rel.item():.3e}   (应 0)")

    with torch.no_grad():
        I3 = forward_ptycho(Ot, Pt, post + 1, Q, cfg.N, chunk=8)
        rel1 = ((I3 / I3.max()).clamp(0, 1) - Im).norm() / Im.norm()
    print(f"[C] 位置整体偏 1 px 的残差       {rel1.item():.3e}   (应远大于 B)")

    # 梯度校验（float64，在偏离真值的点上做有限差分）
    Qd = make_quad_phase(cfg, device, torch.complex128)
    Od, Pd = Ot.to(torch.complex128), Pt.to(torch.complex128)
    Imd = Im.to(torch.float64); sc = I.max().to(torch.float64)
    post_g = post[:min(8, len(post))]   # complex128 在 GPU 上极慢且吃显存，只取 8 个位置
    Imd_g = Imd[:len(post_g)]
    torch.manual_seed(0)
    base = (Od.abs() * (1 + 0.15 * torch.randn(Od.shape, device=device,
                                               dtype=torch.float64)))
    ang = torch.angle(Od)

    def L(a):
        return ((forward_ptycho((a * torch.exp(1j * ang)), Pd, post_g, Qd, cfg.N, chunk=8)
                 / sc - Imd_g) ** 2).sum()

    p_ = nn.Parameter(base.clone()); L(p_).backward()
    # 取样点必须落在【被这 8 个位置照明的】区域内，否则梯度本来就该是 0（假通过）
    i = int(pos[0][0]) + cfg.N // 2
    j = int(pos[0][1]) + cfg.N // 2
    with torch.no_grad():
        e = 1e-6; b = p_.detach().clone()

        def f(v):
            a = b.clone(); a[i, j] = v
            return L(a).item()

        num = (f(b[i, j] + e) - f(b[i, j] - e)) / (2 * e)
    ana = p_.grad[i, j].item()
    r = abs(num - ana) / max(abs(num), abs(ana), 1e-30)
    good = ana != 0.0 and r < 1e-4
    print(f"[D] 梯度 @({i},{j}) 数值 {num:+.4e} vs 解析 {ana:+.4e}  相对差 {r:.2e}  "
          + ("OK " if good else ("!! 解析梯度为 0，取样点没被照到，换点" if ana == 0.0 else "!! ")))

    _report_device(cfg, device)
    Imn, Icl = simulate(cfg, obj, probe, pos, Q, device)
    S2 = (Imn < 1.0 - 1e-6).mean()
    print(f"[E] 噪声 {cfg.noise}@{cfg.snr_db}dB  ->  过曝(S2=0)像素占比 "
          f"{100*(1-S2):.4f} %"
          + ("   <- noise=none 时几乎为 0，Eq.5 的 γ 无事可做" if cfg.noise == "none" else ""))
    print("=" * 78)

def run(cfg: Cfg):
    device = cfg.dev()
    torch.manual_seed(cfg.seed)
    cfg.quad_sign = getattr(cfg, "quad_sign", -1.0)
    if device.type == "cuda":
        # 网络输入尺寸全程固定，benchmark 能稳定选到最快 conv 算法
        torch.backends.cudnn.benchmark = True
    _report_device(cfg, device)

    # 【三种算法共用】数据只能从 build_scene 来，见 functions/paperrepro/scene.py
    sc = build_scene(cfg, device)
    obj, probe, pos = sc.obj, sc.probe, sc.pos
    (rs, cs), Q = sc.roi, sc.Q
    post, Im, Icl, S1 = sc.post, sc.Imt, sc.Iclt, sc.S1t
    S2 = (Im < 1.0 - 1e-6).float()                       # Eq.(6)

    # ---- 网络输入: 零填充后的实测衍射图堆栈，全程固定 (Fig.1c) ----
    M, n, NS = cfg.obj_size, cfg.N, cfg.net_size
    pad_o = (M - n) // 2                                  # 512 -> 612
    pad_n = (NS - M) // 2                                 # 612 -> 616 (被 8 整除)
    x = F.pad(Im[None], (pad_o,) * 4)
    x = F.pad(x, (pad_n, NS - M - pad_n, pad_n, NS - M - pad_n))
    x = x / x.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)

    net = ProPtyUNet(cfg.n_pat, cfg.base_ch).to(device)
    npar = sum(p.numel() for p in net.parameters())
    print(f"[net] 参数 {npar/1e6:.2f} M (论文 2.5 M) | 输入 {tuple(x.shape)} | "
          f"过曝像素 {100*float(1-S2.mean()):.4f}% | 设备 {device}")

    def decode():
        a_s, p_s, a_p, p_p = net(x)
        c = slice(pad_n, pad_n + M)
        a_s, p_s, a_p, p_p = a_s[c, c], p_s[c, c], a_p[c, c], p_p[c, c]
        O = (a_s * torch.exp(1j * cfg.phase_span_obj * p_s)).to(torch.complex64)
        # Fig.1(c): 612 的探针裁到中心 512 再进前向；不加任何硬 support，只靠 Loss2
        d = slice(pad_o, pad_o + n)
        P = (a_p[d, d] * torch.exp(1j * cfg.phase_span_prb * p_p[d, d])).to(torch.complex64)
        return O, P, a_p[d, d], a_s


    # ---- 中性初始化：直接把四个输出头设成常数 ---------------------------- #
    # 目标：O ≡ 1·exp(i0)，P ≡ 1·exp(i0)。
    # 不做预拟合 —— 解析地把输出头定死即可，第 0 步的输出【精确】等于目标：
    #   amp_s / amp_p : Conv2d -> leaky_relu(·,0.2)   weight=0, bias=1 -> 输出恒 1
    #   phs_s / phs_p : Conv2d -> tanh                weight=0, bias=0 -> 输出恒 0
    # 卷积 weight 全 0 时输出与输入无关、恒等于 bias（padding=1 的边界也一样），
    # 所以 |O|=|P|=1、arg(O)=arg(P)=0 是精确的，没有残差也没有额外迭代开销。
    #
    # 【weight=0 会不会断梯度】不会，只慢一步：∂out/∂y = weight = 0，所以 U-Net
    # 主干在第 0 步确实拿不到梯度；但头自身的 ∂out/∂weight = y 随像素变化、非零，
    # 一步 Adam 之后 weight≠0，主干立刻恢复回传。bias 的梯度自始至终非零。
    with torch.no_grad():
        for _h in (net.amp_s, net.amp_p):
            _h.weight.zero_(); _h.bias.fill_(1.0)
        for _h in (net.phs_s, net.phs_p):
            _h.weight.zero_(); _h.bias.zero_()


    with torch.no_grad():
        O, P, _, _ = decode()
        _oa, _op = O.abs(), torch.angle(O)
        _pa, _pp = P.abs(), torch.angle(P)[S1 > 0]

        print(f"[init] 物体初始振幅范围   [{_oa.min().item():.4f}, {_oa.max().item():.4f}]")
        print(f"[init] 物体初始相位 RMS   {_op.pow(2).mean().sqrt().item():.4f} rad")
        print(f"[init] probe 初始振幅范围 [{_pa.min().item():.4f}, {_pa.max().item():.4f}]")
        print(f"[init] probe 有效区内初始相位 RMS {_pp.pow(2).mean().sqrt().item():.4f} rad")
    # --------------------------------------------------------------------- #

    scale = 1.0
    if cfg.scale_cal:
        # 论文没写网络输出的绝对幅度怎么锚定。这里在第一次前向后算一个常数并冻结，
        # 纯数值辅助，不改物理；--no-scale-cal 可关掉看差别。
        with torch.no_grad():
            O, P, _, _ = decode()
            I0 = forward_ptycho(O, P, post, Q, n, chunk=8)
            scale = (Im.mean() / I0.mean().clamp_min(1e-20)).item()
        print(f"[net] 冻结幅度标定 = {scale:.4g}")

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.iters, eta_min=cfg.lr * cfg.lr_final_frac)
    rng = np.random.default_rng(cfg.seed)
    hist, t0 = [], time.time()

    for it in range(cfg.iters):
        gamma = cfg.gamma0 * (cfg.gamma_end / cfg.gamma0) ** (it / max(cfg.iters - 1, 1))
        if cfg.pos_batch > 0:
            sel = torch.from_numpy(rng.choice(cfg.n_pat, cfg.pos_batch, replace=False)).to(device)
        else:
            sel = torch.arange(cfg.n_pat, device=device)

        O, P, amp_p, amp_s = decode()
        Ic = forward_ptycho(O, P, post[sel], Q, n, chunk=0) * scale
        loss, l1, l2 = paper_loss(Ic, Im[sel], S2[sel], gamma, amp_p, S1, cfg.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                real = torch.linalg.vector_norm(Ic - Icl[sel]).item()   # 论文判据(2)
                rec = O.cpu().numpy()
                a_s_roi = amp_s[rs, cs].cpu().numpy()
            m = evaluate(rec[rs, cs], obj[rs, cs])
            wrap, negamp = seam_diag(rec[rs, cs], obj[rs, cs], a_s_roi)
            m["wrap_frac"], m["signmix"] = wrap, negamp
            m["relerr_p"] = probe_relerr(P.detach().cpu().numpy(), probe)
            hist.append({"it": it + 1, "loss": loss.item(), "real": real, **m})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                print(f"  it {it+1:5d} | loss {loss.item():.4e} (L1 {l1:.3e} L2 {l2:.3e}) | "
                      f"real {real:.4e} | γ {gamma:.3f} | amp SSIM {m['ssim_amp']:.4f} "
                      f"PSNR {m['psnr_amp']:5.2f} | phs SSIM {m['ssim_phs']:.4f} | "
                      f"relerr {m['relerr']:.4f} | wrap {100*wrap:.1f}% "
                      f"signmix {100*negamp:.1f}%", flush=True)

    print(f"[net] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")
    if len(hist) > 8:
        tail = np.array([h["real"] for h in hist[-len(hist)//4:]])
        k = np.polyfit(np.arange(len(tail)), tail, 1)[0]
        print(f"[net] 尾段 real error 斜率 {k:+.3e} "
              f"({'仍在下降' if k < 0 else '已回升 → 开始拟合噪声'})"
              + ("   （noise=none 时这一项没有意义）" if cfg.noise == "none" else ""))
    _save(cfg, rec, P.detach().cpu().numpy(), obj, probe, hist, (rs, cs), pos,
          tag="paper")
