# -*- coding: utf-8 -*-
"""三种运行模式：自检 / 纯 AD / DIP。"""

from __future__ import annotations

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.addip.adjoint import build_adjoint_features, save_adjoint_features
from functions.addip.evaluate import evaluate_object_roi, evaluate_probe
from functions.addip.losses import cabs, data_loss_direct, held_out_mse, masked_mse, safe_angle, tgv_loss
from functions.addip.model import ProPtyUNet, make_field
from functions.addip.optics import crop_patch_np, forward_np, forward_torch, make_H, propagate_np
from functions.addip.report import _banner, _save
from functions.addip.sample import check_scan_fits, make_probe_init, make_scan_positions, make_truth, probe_init_err, simulate

PI = math.pi

from typing import TYPE_CHECKING
if TYPE_CHECKING:                      # 仅类型标注用，运行时不导入，无循环依赖
    from ProPtyNet_torch import Cfg

def run_check(cfg: Cfg):
    dev = cfg.dev()
    obj, probe, rr = make_truth(cfg)
    positions = make_scan_positions(cfg)
    H_true = make_H(cfg, cfg.dz_true)

    print("=" * 74)
    print(f"角谱采样上限 dz_max = {cfg.dz_max*1e3:.2f} mm   当前 dz = {cfg.dz_true*1e3:.2f} mm"
          f"   -> {'OK' if abs(cfg.dz_true) <= cfg.dz_max else '超限！会混叠'}")
    a = cfg.probe_dia * cfg.dx / 2
    print(f"菲涅耳数 a²/(λ·dz) = {a**2/(cfg.wlength*cfg.dz_true):.1f}  (深近场 -> 角谱是正确选择)")
    print("-" * 74)
    check_scan_fits(cfg, positions)
    print(f"  评估区 {cfg.eval_size}²(crop {cfg.EVAL_CROP})   "
          f"正则区 {cfg.reg_size}²(crop {cfg.REG_CROP})")
    _banner(cfg, probe_init_err(cfg, rr, probe), "check")
    print("-" * 74)

    # A) 角谱往返
    rngt = (np.random.default_rng(0).random((cfg.N, cfg.N))
            + 1j * np.random.default_rng(1).random((cfg.N, cfg.N)))
    back = np.fft.ifft2(np.fft.fft2(propagate_np(rngt, H_true)) * np.conj(H_true))
    print(f"[A] 角谱往返相对误差            {np.linalg.norm(back-rngt)/np.linalg.norm(rngt):.3e}  (应 ~1e-8)")

    # B) torch 前向 vs numpy 前向
    Ot = torch.from_numpy(obj).to(dev)
    Pt = torch.from_numpy(probe).to(dev)
    Ht = torch.from_numpy(H_true).to(dev)
    corners = torch.from_numpy(cfg.PATCH_C0 - positions).to(dev)
    with torch.no_grad():
        Ut = forward_torch(cfg, Ot, Pt, corners, Ht).cpu().numpy()
    Un = np.stack([forward_np(cfg, obj, probe, p, H_true) for p in positions])
    print(f"[B] torch 前向 vs numpy 前向     {np.linalg.norm(Ut-Un)/np.linalg.norm(Un):.3e}  (应 <1e-6)")

    # C) 整数位置下 crop_patch == 直接切片
    ci = np.array([cfg.PATCH_C0, cfg.PATCH_C0])
    w1 = crop_patch_np(obj, ci, cfg.N)
    i0 = int(cfg.PATCH_C0)
    w2 = obj[i0:i0 + cfg.N, i0:i0 + cfg.N]
    print(f"[C] 整数位置 crop == 直接切片     {np.abs(w1-w2).max():.3e}  (应 0)")

    # D) 梯度回流窗口
    Op = torch.nn.Parameter(torch.ones((cfg.N_OBJ, cfg.N_OBJ), device=dev))
    Oc = (Op * torch.exp(torch.zeros_like(Op) * 1j)).to(torch.complex64)
    U = forward_torch(cfg, Oc, Pt, corners, Ht)
    U.abs().pow(2).sum().backward()
    g = (Op.grad.abs() > 0).cpu().numpy()
    ys, xs = np.where(g)
    print(f"[D] 物体梯度回流窗口             行 {ys.min()}..{ys.max()}  列 {xs.min()}..{xs.max()}"
          f"  ({ys.max()-ys.min()+1}×{xs.max()-xs.min()+1})，画布 {cfg.N_OBJ}")

    # E) 真值代入的数据残差 —— 无探针支撑，所以应当是 0（除噪声外）
    I, _ = simulate(cfg, obj, probe, positions, H_true)
    y0 = torch.from_numpy(np.sqrt(np.maximum(I, 0))).to(dev)
    with torch.no_grad():
        Um = forward_torch(cfg, Ot, Pt, corners, Ht)
        r = ((Um.abs() - y0).norm() / y0.norm()).item()
    print(f"[E] 真值代入的数据残差           {r:.3e}  "
          + ("  <- 无探针支撑，数据/模型零失配" if not cfg.poisson else "  (泊松噪声下非 0 属正常)"))
    print("=" * 74)

def run_ad(cfg: Cfg):
    """物体 = 自由复数像素。联合更新 + 全程单个 Adam + 全批量 + 无 lr 调度。

    与 run_net 共用：同一份数据、同一个探针初值、探针从第 0 步就更新。
    唯一的差别是物体的参数化。
    """
    dev = cfg.dev()
    obj, probe, rr = make_truth(cfg)
    positions = make_scan_positions(cfg)
    check_scan_fits(cfg, positions)
    H_true = make_H(cfg, cfg.dz_true)
    I, _ = simulate(cfg, obj, probe, positions, H_true)

    Ht = torch.from_numpy(H_true).to(dev)
    corners = torch.from_numpy(cfg.PATCH_C0 - positions).to(dev)
    y0 = torch.from_numpy(np.sqrt(np.maximum(I, 0))).to(dev)
    ones = np.ones_like(rr, dtype=np.float32)

    # 初值：物体 O ≡ 1（与 net 的 obj_init_alpha=0 相同）；探针 = make_probe_init
    Or = nn.Parameter(torch.ones((cfg.N_OBJ, cfg.N_OBJ), device=dev))
    Oi = nn.Parameter(torch.zeros((cfg.N_OBJ, cfg.N_OBJ), device=dev))
    P0 = make_probe_init(cfg, rr)
    _banner(cfg, probe_init_err(cfg, rr, probe), "ad")
    Pr = nn.Parameter(torch.from_numpy(P0.real.astype(np.float32)).to(dev))
    Pi = nn.Parameter(torch.from_numpy(P0.imag.astype(np.float32)).to(dev))

    c = cfg.REG_CROP
    opt = torch.optim.Adam([{"params": [Or, Oi], "lr": cfg.lr_obj},
                            {"params": [Pr, Pi], "lr": cfg.lr_prb}])
    print(f"[ad] 标准 AD: 全批量 {len(positions)} 位置/步 × {cfg.ad_iters} 步  "
          f"lr_obj={cfg.lr_obj:g} lr_prb={cfg.lr_prb:g}（不衰减）  "
          f"正则 tv1={cfg.tv1:g} tv2={cfg.tv2:g}")

    hist, t0, rec, pc = [], time.time(), None, None
    for it in range(cfg.ad_iters):
        O = torch.complex(Or, Oi)
        P = torch.complex(Pr, Pi)
        U = forward_torch(cfg, O, P, corners, Ht)
        loss = data_loss_direct(U, y0)
        if cfg.tv1 > 0 or cfg.tv2 > 0:
            Oreg = O[c:-c, c:-c] if c > 0 else O
            if cfg.tv1 > 0:
                loss = loss + cfg.tv1 * tgv_loss(Oreg.abs(), beta=1.0) * 1e-2
            if cfg.tv2 > 0:
                loss = loss + cfg.tv2 * tgv_loss(safe_angle(Oreg), beta=1.0) * 1e-2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % cfg.eval_every == 0 or it == cfg.ad_iters - 1:
            rec = torch.complex(Or, Oi).detach().cpu().numpy()
            pc = torch.complex(Pr, Pi).detach().cpu().numpy()
            mo = evaluate_object_roi(cfg, rec, obj)
            mp = evaluate_probe(pc, probe, ones, r_far=cfg.probe_dia)
            hist.append({"it": it + 1, "loss": loss.item(), **mo, **mp})
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.ad_iters - 1:
                print(f"  it {it+1:5d} | loss {loss.item():.4e} | "
                      f"amp PSNR {mo['psnr_o_amp']:6.2f} dB | SSIM {mo['ssim_o_amp']:.4f} | "
                      f"phase RMSE {mo['rmse_o_phi_rad']:.4f} rad | complex err "
                      f"{mo['relerr_o_complex']:.4f} | probe err {mp['relerr_p_complex']:.4f}",
                      flush=True)
    print(f"[ad] 用时 {time.time()-t0:.1f}s / {cfg.ad_iters} it")
    _save(cfg, "ad", rec, pc, obj, probe, ones, hist)
    return hist

def run_net(cfg: Cfg):
    dev = cfg.dev()
    torch.manual_seed(cfg.seed)
    obj, probe, rr = make_truth(cfg)
    positions = make_scan_positions(cfg)
    check_scan_fits(cfg, positions)
    H_true = make_H(cfg, cfg.dz_true)
    I, I_clean = simulate(cfg, obj, probe, positions, H_true)

    Ht = torch.from_numpy(H_true).to(dev)
    corners = torch.from_numpy(cfg.PATCH_C0 - positions).to(dev)
    Im = torch.from_numpy(I).to(dev)
    Iclean = torch.from_numpy(I_clean).to(dev)
    sqrtIm = torch.sqrt(Im.clamp_min(0))
    pad = (cfg.N_OBJ - cfg.N) // 2
    ones = np.ones_like(rr, dtype=np.float32)

    # 留出探测器像素：无 GT 的早停判据。不破坏扫描几何（重叠率不变）。
    if cfg.holdout_frac > 0:
        g = torch.Generator().manual_seed(cfg.holdout_seed)
        Mtr = (torch.rand(Im.shape, generator=g) >= cfg.holdout_frac).float().to(dev)
        print(f"[net] 留出探测器像素 {100*(1-Mtr.mean().item()):.2f}% 作验证集（不进 loss）")
    else:
        Mtr = None

    # 探针
    prb_params, Ptruth = [], None
    if cfg.probe_mode == "truth":
        Ptruth = torch.from_numpy(probe.astype(np.complex64)).to(dev)
        print("[net] 探针 = 真值且冻结（上界对照：只考察物体表示本身的能力）")
    else:
        P0 = make_probe_init(cfg, rr)
        _banner(cfg, probe_init_err(cfg, rr, probe), "net")
        Pr = nn.Parameter(torch.from_numpy(P0.real.astype(np.float32)).to(dev))
        Pi = nn.Parameter(torch.from_numpy(P0.imag.astype(np.float32)).to(dev))
        prb_params = [Pr, Pi]
        print(f"[net] 探针 = 自由复数像素（{2*rr.size} 个未知量），从第 0 步起与物体联合更新")

    # ---- 网络输入 ----------------------------------------------------- #
    # 【探针块必须在这之前】伴随融合要用 P_ref = P0；它只读 P0，不改后续参与优化的 Pr/Pi。
    idx_full = torch.arange(len(positions), device=dev)
    idx_sparse, feat_sparse, feat_full = idx_full, None, None
    if cfg.input_mode == "raw":
        # 历史行为，逐位不变：零填充的实测衍射图堆栈，通道数 = 扫描点数
        x_in = F.pad(Im[None], (pad, pad, pad, pad))
        x_in = x_in / x_in.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        in_ch = len(positions)
    else:
        # 物理伴随实空间融合：固定 4 通道，与扫描点数无关
        # P_ref 只能是 P0；probe_mode='truth' 的上界对照才允许传冻结的真值探针
        P_ref = Ptruth if Ptruth is not None else torch.complex(Pr.detach(), Pi.detach())
        O_ref = torch.ones((cfg.N_OBJ, cfg.N_OBJ), dtype=torch.complex64, device=dev)
        idx_sparse = torch.arange(0, len(positions), cfg.curriculum_stride, device=dev)
        feat_sparse = build_adjoint_features(cfg, sqrtIm, corners, Ht, P_ref,
                                             idx_sparse, O_ref)
        feat_full = build_adjoint_features(cfg, sqrtIm, corners, Ht, P_ref,
                                           idx_full, O_ref)
        x_in, in_ch = feat_sparse, 4
        _s1 = int(cfg.iters * cfg.curriculum_stage1_frac)
        _rp = max(1, int(cfg.iters * cfg.curriculum_ramp_frac))
        print(f"[net-input] mode={cfg.input_mode}")
        print(f"[net-input] sparse positions={len(idx_sparse)}/{len(positions)} "
              f"stride={cfg.curriculum_stride}")
        print(f"[net-input] features_sparse={tuple(feat_sparse.shape)}")
        print(f"[net-input] features_full={tuple(feat_full.shape)}")
        print(f"[curriculum] stage1={_s1} iters, ramp={_rp} iters, "
              f"full={max(cfg.iters - _s1 - _rp, 0)} iters")
        print(f"[curriculum] probe frozen during sparse stage: "
              f"{bool(cfg.curriculum_freeze_probe_stage1)}")
        os.makedirs(cfg.outdir, exist_ok=True)
        save_adjoint_features(cfg, feat_sparse, 1e-3,
                              os.path.join(cfg.outdir, "adjoint_features_sparse.png"))
        save_adjoint_features(cfg, feat_full, 1e-3,
                              os.path.join(cfg.outdir, "adjoint_features_full.png"))

    net = ProPtyUNet(in_ch, cfg.base_ch, n_fields=1, ph_ch=2).to(dev)
    print(f"[net] U-Net {sum(p.numel() for p in net.parameters())/1e6:.2f} M 参数  "
          f"输入 {tuple(x_in.shape)}")

    # 物体输出头的相位中性初始化。alpha=0 时第 0 步 O ≡ 1·exp(i0)，与 AD 初值完全相同。
    # alpha=0 不会死梯度：cossin 在 (c,sn)=(1,0) 处 d(phi)/d(sn)=1，softplus 在
    # b=ln(e-1) 处导数 0.632，且权重梯度随像素变化，第一步就破对称。
    _a = float(cfg.obj_init_alpha)
    with torch.no_grad():
        net.head_amp.weight[0].mul_(_a)
        net.head_phs.weight[0:2].mul_(_a)
        net.head_amp.bias[0] = math.log(math.e - 1.0)   # softplus(b) = 1
        net.head_phs.bias[0:2] = 0.0
        net.head_phs.bias[0] = 1.0                      # (c, sn) = (1, 0) -> phi = 0
    print(f"[net] 物体输出头 = 相位中性初始化 alpha={_a:g}"
          + ("（第 0 步 O ≡ 1·exp(i0)，与 AD 初值相同）" if _a == 0 else ""))

    def decode():
        a_raw, p_raw = net(x_in)
        O = make_field(a_raw[0], p_raw[0:2])
        Pc = Ptruth if Ptruth is not None else torch.complex(Pr, Pi)
        return O, Pc

    opt_net = torch.optim.Adam(net.parameters(), lr=cfg.lr_net,
                               weight_decay=cfg.weight_decay)
    opt_prb = torch.optim.Adam(prb_params, lr=cfg.lr_probe) if prb_params else None

    hist, t0 = [], time.time()
    best_val = {"val": float("inf"), "it": -1, "ssim": float("nan")}
    best_gt = {"ssim": -1.0, "it": -1}
    rec = pc = None

    # 两阶段课程的分界（raw 模式下这三个量不参与任何计算）
    stage1_end = int(cfg.iters * cfg.curriculum_stage1_frac)
    ramp_len = max(1, int(cfg.iters * cfg.curriculum_ramp_frac))
    alpha, scan_weight, probe_lr_now = 1.0, None, cfg.lr_probe

    for it in range(cfg.iters):
        cos_f = (0.5 * (1 + math.cos(PI * it / max(cfg.iters - 1, 1)))
                 if cfg.lr_cosine else 1.0)
        if cfg.input_mode != "raw":
            if it < stage1_end:
                alpha = 0.0
            elif it < stage1_end + ramp_len:
                alpha = (it - stage1_end) / ramp_len
            else:
                alpha = 1.0
            # 同语义特征之间的线性过渡（不是拼成 8 通道）
            x_in = (1.0 - alpha) * feat_sparse + alpha * feat_full
            scan_weight = torch.full((len(positions),), alpha, device=dev)
            scan_weight[idx_sparse] = 1.0
            probe_lr_now = (cfg.lr_probe * alpha * cos_f
                            if cfg.curriculum_freeze_probe_stage1
                            else cfg.lr_probe * cos_f)
        else:
            probe_lr_now = cfg.lr_probe * cos_f
        if cfg.lr_cosine or cfg.input_mode != "raw":
            for g in opt_net.param_groups:
                g["lr"] = cfg.lr_net * cos_f
            if opt_prb is not None:
                for g in opt_prb.param_groups:
                    g["lr"] = probe_lr_now
        O, Pc = decode()
        Ua = cabs(forward_torch(cfg, O, Pc, corners, Ht))
        # 全局幅度标度：物体与探针之间有 O->aO, P->P/a 的规范自由度。每步解析地求最优
        # 标量（VarPro），不 detach —— 这样标度方向上的梯度恒为 0，那个自由度被消掉。
        Ua = Ua * ((Ua * sqrtIm).sum() / (Ua * Ua).sum().clamp_min(1e-20))
        if scan_weight is None:
            loss = masked_mse(Ua, sqrtIm, Mtr)          # raw：逐位不变
        else:
            # 按扫描位置加权：未激活的位置既不进分子也不进分母
            err = (Ua - sqrtIm).square()
            if Mtr is not None:
                err = err * Mtr
                den = (scan_weight[:, None, None] * Mtr).sum().clamp_min(1e-12)
            else:
                den = (scan_weight.sum() * cfg.N * cfg.N).clamp_min(1e-12)
            loss = (err * scan_weight[:, None, None]).sum() / den

        opt_net.zero_grad(set_to_none=True)
        if opt_prb is not None:
            opt_prb.zero_grad(set_to_none=True)
        loss.backward()
        opt_net.step()
        if opt_prb is not None:
            opt_prb.step()

        if (it + 1) % cfg.eval_every == 0 or it == cfg.iters - 1:
            with torch.no_grad():
                real = torch.linalg.vector_norm(Ua ** 2 - Iclean).item()
                val = held_out_mse(Ua, sqrtIm, Mtr) if Mtr is not None else float("nan")
                rec = O.detach().cpu().numpy(); pc = Pc.detach().cpu().numpy()
            mo = evaluate_object_roi(cfg, rec, obj)
            mp = evaluate_probe(pc, probe, ones, r_far=cfg.probe_dia)
            hist.append({"it": it + 1, "loss": loss.item(), "val": val, "real": real,
                         **mo, **mp})
            if mo["ssim_o_amp"] > best_gt["ssim"]:
                best_gt = {"ssim": mo["ssim_o_amp"], "it": it + 1}
            if Mtr is not None and val < best_val["val"]:
                best_val = {"val": val, "it": it + 1, "ssim": mo["ssim_o_amp"]}
            if (it + 1) % (cfg.eval_every * 4) == 0 or it == cfg.iters - 1:
                v = f" | val {val:.4e}" if Mtr is not None else ""
                cur = ("" if cfg.input_mode == "raw" else
                       f" | alpha={alpha:.3f} active_weight="
                       f"{scan_weight.sum().item():.1f} probe_lr={probe_lr_now:.2e}")
                print(f"  it {it+1:5d} | loss {loss.item():.4e}{v} | real {real:.4e} | "
                      f"amp PSNR {mo['psnr_o_amp']:6.2f} dB | SSIM {mo['ssim_o_amp']:.4f} | "
                      f"phase RMSE {mo['rmse_o_phi_rad']:.4f} rad | complex err "
                      f"{mo['relerr_o_complex']:.4f} | probe err "
                      f"{mp['relerr_p_complex']:.4f}{cur}", flush=True)

    print(f"[net] 用时 {time.time()-t0:.1f}s / {cfg.iters} it")
    if len(hist) > 8:
        tail = [h["real"] for h in hist[-len(hist)//4:]]
        k = np.polyfit(np.arange(len(tail)), np.array(tail), 1)[0]
        print(f"[net] 尾段 real error 斜率 {k:+.3e}  "
              f"({'仍在下降' if k < 0 else '已回升 -> 开始拟合噪声'})")
    if Mtr is not None:
        print(f"[net] 留出验证最优步 = {best_val['it']} (val {best_val['val']:.4e}, "
              f"该步 SSIM {best_val['ssim']:.4f})")
        print(f"[net] 真值 SSIM 最优步 = {best_gt['it']} (SSIM {best_gt['ssim']:.4f})"
              f"   <- 两者越接近，留出验证越可以替代 GT 做早停")

    _save(cfg, "net", rec, pc, obj, probe, ones, hist)
    return hist
