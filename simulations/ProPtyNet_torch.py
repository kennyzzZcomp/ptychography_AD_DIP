#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ProPtyNet (PyTorch) —— 未训练网络先验 vs 纯 AD 的 ptychography 对照。

固定下来的约定（不再是开关，改了就是换了一个实验）：
  * 扫描        规则栅格 raster，位置严格确定性
  * 探针支撑    无。模型能表示完整探针，数据/模型零失配
  * 探针初值    平滑圆盘 + 零相位、不传播（err_P0 ≈ 0.27）。探针从第 0 步就参与优化
  * 物体初值    相位中性：第 0 步 O ≡ 1·exp(i0)，与 AD 完全相同
  * 参数化      振幅 softplus、相位 cos/sin 单位圆、标度每步闭式最小二乘(VarPro)
  * 数据项      振幅域 ‖|U|-√I‖²
  * AD          标准 AD ptychography：联合更新 + 单个 Adam + 全批量 + 无 lr 调度

两个方法在【同一份数据、同一个初值、同一套协议】下跑，唯一变量是物体的参数化：
  ad  -> 物体 = 自由复数像素
  net -> 物体 = U-Net 的输出（DIP 先验）
"""

from __future__ import annotations

import sys
import math
import argparse
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 让 functions/ 可被 import

from functions.addip.solvers import run_check, run_ad, run_net

PI = math.pi


# ============================================================================ #
# 配置 —— 本文件【只放配置与 CLI】，所有实现在 functions/addip/ 与 functions/common/
# ============================================================================ #

@dataclass
class Cfg:
    # ---- 光学 ----
    wlength: float = 632.8e-9
    N: int = 128                 # 探测器 / patch 边长
    N_OBJ: int = 224             # 物体画布
    dx: float = 10e-6            # 样品面 = 探测器面像素（角谱保持像素尺寸）
    dz_true: float = 8e-3        # 样品-探测器
    probe_dia: int = 48          # R_AP = probe_dia/2 - 6 = 18 px
    z_probe: float = 3e-3        # 光阑 -> 样品，合成真值探针
    probe_aberr: float = 0.0     # >0 时给真值光阑加随机波前（真值探针不再理想）
    aberr_seed: int = 7

    # ---- 探针初值 P0：平滑圆盘 + 零相位、不传播 ----
    # 只编码"光阑大概多大"，不给波前、也不给传播产生的 Fresnel 环纹
    # （真值探针被传播摊开到约 33px = 光阑的 1.8 倍）。
    # sigma 单位 = R_AP 的倍数，物理含义 = "知道光阑大小、不知道边缘锐度"。
    # ⚠ 不要按 err_P0 最小去调它 —— 那等于偷用真值信息。定死 0.15。
    probe_init_sigma: float = 0.15

    # ---- 扫描 ----
    scan_npos: int = 25              # 必须是完全平方数
    scan_step: float = 20.0          # 重叠轴有 26.7 / 13.3 这些小数档，必须是 float

    # ---- 区域 ----
    eval_size: int = 96          # -> EVAL_CROP = (224-96)/2 = 64
    reg_size: int = 128          # -> REG_CROP  = (224-128)/2 = 48
    phase_support: float = 0.05  # 相位指标只在振幅 > 该比例×峰值 的像素上统计

    # ---- 噪声 ----
    poisson: bool = False
    peak_photons: float = 5000.0
    noise_global_norm: bool = True
    noise_seed: int = 42

    # ---- AD（标准 AD ptychography）----
    ad_iters: int = 2000         # 全批量梯度步数（与 net 的 iters 同刻度）
    lr_obj: float = 1e-2
    lr_prb: float = 1e-2
    tv1: float = 0.0             # 物体振幅 TGV，0 = 关
    tv2: float = 0.0             # 物体相位 TGV，0 = 关

    # ---- net（DIP）----
    iters: int = 2000
    lr_net: float = 1e-3
    lr_probe: float = 1e-2       # 自由像素探针，与 AD 的 lr_prb 同量级
    lr_cosine: bool = False      # 两个 lr 一起余弦退火到 0，治后期 loss 尖峰
    base_ch: int = 32            # -> 约 2.2 M 参数
    weight_decay: float = 0.0    # DIP 不该有权重衰减
    eval_every: int = 25
    # 物体输出头权重的缩放：0 = 严格中性（第 0 步 O ≡ 1，与 AD 初值相同），
    # 1 = 保持 PyTorch 默认随机。中间值 = 一条"初始相位粗糙度"的连续轴。
    obj_init_alpha: float = 0.0
    # 探针参数化：pixel = 自由复数像素；truth = 冻结在真值上（上界对照，不参与优化）
    probe_mode: str = "pixel"    # pixel | truth

    # ---- 无 GT 早停：留出探测器像素 ----
    holdout_frac: float = 0.0    # >0 时每张图随机留出这么多像素，永不进 loss
    holdout_seed: int = 1234

    # ---- 真值物体 ----
    obj_amp_min: float = 0.4     # 振幅 = [obj_amp_min, 1.0]
    obj_phase_span: float = 0.8  # 相位 = ±该值 (rad)。0 = 纯振幅物体
    obj_amp_img: str = "Siemens.jpg"
    obj_phase_img: str = "Peppers.jpg"

    # ---- 结果 PNG 的显示范围（只影响出图，不影响任何指标）----
    disp_amp_lo: float = 0.15
    disp_amp_hi: float = 1.15
    disp_phase_lim: float = 1.0
    disp_err_amp: float = 0.3
    disp_err_phase: float = 0.3

    # ---- 其它 ----
    seed: int = 0                # 只影响 net。AD 完全确定性，--seed 对它无效
    device: str = "auto"
    assets: str = ""
    outdir: str = "results_proptynet"

    def __post_init__(self):
        self.k0 = 2 * PI / self.wlength
        self.R_AP = self.probe_dia / 2 - 6
        self.EVAL_CROP = (self.N_OBJ - self.eval_size) // 2
        self.REG_CROP = (self.N_OBJ - self.reg_size) // 2
        self.PATCH_C0 = (self.N_OBJ - self.N) / 2.0
        self.SCAN_LIMIT = self.PATCH_C0
        self.dz_max = self.N * self.dx ** 2 / self.wlength

    def dev(self):
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================ #
# CLI —— 直接从 Cfg 的字段生成，不再手工维护一份平行列表
# ============================================================================ #

_CHOICES = {
    "probe_mode": ["pixel", "truth"],
    "device": None,
}


def build_parser():
    ap = argparse.ArgumentParser(
        description="ProPtyNet —— 未训练网络先验 vs 纯 AD 的 ptychography 对照")
    ap.add_argument("mode", choices=["check", "ad", "net"],
                    help="check=自检 | ad=纯 AD 基线 | net=DIP")
    for f in dataclass_fields(Cfg):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, dest=f.name, action="store_true", default=None)
            if f.default:            # 默认 True 的开关，补一个 --no-xxx 才关得掉
                ap.add_argument("--no-" + f.name.replace("_", "-"),
                                dest=f.name, action="store_false", default=None)
        elif f.name in _CHOICES and _CHOICES[f.name]:
            ap.add_argument(flag, dest=f.name, choices=_CHOICES[f.name], default=None)
        else:
            ap.add_argument(flag, dest=f.name, type=type(f.default), default=None)
    return ap


def main():
    a = build_parser().parse_args()
    kw = {k: v for k, v in vars(a).items() if k != "mode" and v is not None}
    cfg = Cfg(**kw)
    {"check": run_check, "ad": run_ad, "net": run_net}[a.mode](cfg)


if __name__ == "__main__":
    main()
