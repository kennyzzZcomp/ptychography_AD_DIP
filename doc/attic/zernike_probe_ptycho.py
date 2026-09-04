"""封存：ptychography 里的光瞳面 Zernike 探针参数化（已从主线代码删除）

删除理由与完整实验结论见 doc/README_zernike_probe.md。一句话：12 个系数确实能把
探针误差降到 1/2.6，但只有传播距离 z 正确时才赢；z 错 30% 时它比自由探针还差
（物体 SSIM 0.4635 vs 0.7009）。深近场下离焦吸收不了 z 误差，用它就必须先做自动对焦，
而自动对焦不是本课题的主线。真实探针还有散斑与高频粗糙度，低阶基底原理上拟合不了
（ABERR_ROUGH=0.3 时偏差下限 29.1%）。

本文件是【存档】，不能直接运行。要复原：把下面三段贴回 INNM_Ptycho.ipynb 的
cell 2 / cell 4 / 一个新 cell，并在 cell 6-7 里恢复 PROBE_MODEL 分支
（Zk/apz/Hp 常量、MyLayer_COEF 分支、get_zcoef、PROBE_LAYERS、PZ_NORM、
_init_weights 里的 Zc 初始化）。

innm_common.py 里的 zernike_basis / zernike_fit / zernike_pupil_np / MyLayer_COEF /
evaluate_pupil 都【没有删】—— INNM_FPM.ipynb 的光瞳像差恢复还在用它们。

存档时间 2026-09-02，对应前向模型：224 画布 + crop patch，direct 数据项。
"""

# ===================================================================
# [cell 2] 参数：探针参数化方式 + 真值像差
# ===================================================================
CELL2 = r"""
# --- 探针的参数化方式 ---
# 'free'    自由复数矩阵。名义上 2N^2，但 support 掩膜外恒为 0，
#           有效未知量 = 2 * support 内像素数 = 3922（现状）
# 'zernike' 【光瞳面】Zernike 相位 + 角谱传播到样品面，K = 12 个未知量
#
# 【为什么必须放在光瞳面而不是样品面】样品面的探针是菲涅耳环带，相位高频振荡，
# 低阶多项式拟合不了（实测 10 个模式残差 1.09 rad RMS，而动态范围才 6.15 rad）。
# 但像差的物理位置本来就在光瞳里，环带是传播【出来】的。写成
#       P(r) = Propagate_z[ A(rho) * exp(i * sum_k c_k Z_k(rho)) ]
# 之后 Zernike 只需描述那个平滑的像差，环带由传播免费生成。
PROBE_MODEL   = 'free'          # 'free' | 'zernike'
ZERN_NMAX     = 4               # 最高径向阶 n（n_max=4 -> 15 个模式）
ZERN_DROP     = (0, 1, 2)       # 剔除的 OSA 序号：活塞 + 两个倾斜（见下）
R_AP          = probe_dia/2 - 6 # 光瞳半径（像素），与真值孔径一致
z_probe_model = z_probe_init    # 模型使用的传播距离；见 Zernike 对照 cell 里的 z 扫描

# 【剔除活塞与倾斜】活塞与物体的全局相位简并（align_global_factor 正在消它），
# 倾斜与扫描位置的整体平移简并。留着不报错，但那 3 个方向在数据里不受任何约束。
#
# 【离焦与 z 不简并 —— 实测】原以为光瞳离焦能吸收传播距离误差，实测【不能】：
# z 从 3.0mm 改到 3.9mm 造成探针 24.98% 的变化，最优离焦系数只能吸收 2%。
# 原因是这里菲涅耳数很高（R^2/(lambda*z) = 17），是深近场，传播产生的是环带
# 而不是二次相位，离焦学不来。后果很实际：自由探针有 32768 个自由度可以硬吃掉
# z 的误差，Zernike 探针只有 K 个、吃不掉 —— 所以用 'zernike' 时 z 必须自己拟合，
# 见对照 cell 里的 scan_z_probe()。

# --- 真值探针的像差 ---
# 【为什么必须加】cell 4 原本的真值探针是"圆孔直接传播"，即光瞳面像差恰好为零。
# 光瞳面 Zernike 模型只要令全部系数 = 0 就是精确解 —— 这是 inverse crime，
# 结果一定漂亮但什么都没证明。要检验参数化的价值就得给真值加上真实像差。
PROBE_ABERR  = 0.0    # 落在模型张成空间【内】的像差强度（rad RMS）；测方差降低
ABERR_ROUGH  = 0.0    # 张成空间【外】的粗糙度（rad RMS）；测偏差-方差权衡
ABERR_SEED   = 7
# 两个都为 0 时，本 notebook 的行为与加这段之前【逐位相同】
# （回归实测：PROBE_ABERR=0，free，8 stage -> SSIM 0.6976，与改动前一致）。
#
# 【这个数远比直觉小 —— 实测灵敏度】菲涅耳数 R^2/(lambda*z)=17，是深近场，
# 光瞳上一点点相位会把样品面的环带结构整个重排：
#     PROBE_ABERR   样品面探针变化   衍射图变化
#       0.05           17.1%          15.3%
#       0.1            33.3%          29.4%     <- 推荐值
#       0.2            60.9%          51.4%
#       0.3            81.2%          63.2%
#       0.5           101.6%          76.8%     <- 已经是"换了个探针"，两种模型都崩
# 0.5 rad RMS 听起来只有 lambda/12，实测却让 free 和 zernike 双双掉到 SSIM<0.1。
# 想测"参数化有没有用"就用 0.1；想测"崩溃边界"再往上加。

"""

# ===================================================================
# [cell 4] 真值光瞳基底与像差的构造
# ===================================================================
CELL4 = r"""
# 光瞳面 Zernike 基底（模型和真值共用同一套，保证"张成空间内/外"的说法有意义）
Zb, pupil_mask, ZLAB = ic.zernike_basis(N, R_AP, n_max=ZERN_NMAX, drop=ZERN_DROP)
K_ZERN = Zb.shape[2]

_rab = np.random.default_rng(ABERR_SEED)
if PROBE_ABERR > 0:
    _c = _rab.normal(size=K_ZERN)
    coef_true = (PROBE_ABERR*_c/np.sqrt(np.mean(_c**2))).astype(np.float32)
else:
    coef_true = np.zeros(K_ZERN, np.float32)

_phi_ap = Zb @ coef_true
if ABERR_ROUGH > 0:
    # 张成空间【之外】的成分：先造平滑随机场，再把能被基底表示的部分投影掉，
    # 剩下的就是 Zernike 模型原理上表示不了的偏差。
    _r = cv2.GaussianBlur(_rab.normal(size=(N, N)).astype(np.float64), (0, 0), 1.5)
    _perp = _r - Zb @ ic.zernike_fit(_r, Zb, pupil_mask)
    _phi_ap = _phi_ap + _perp*(ABERR_ROUGH/np.sqrt(np.mean(_perp[pupil_mask]**2)))

pupil_true = (pupil_mask*np.exp(1j*_phi_ap)).astype(np.complex64)
probe = propagate_np(pupil_true, z_probe)
probe = (probe/np.abs(probe).max()).astype(np.complex64)
print(f'真值光瞳: 半径 {R_AP:.0f}px  Zernike {K_ZERN} 模式(n<={ZERN_NMAX}, 去掉活塞/倾斜)')
print(f'  张成空间内像差 {PROBE_ABERR:.3f} rad RMS   空间外粗糙度 {ABERR_ROUGH:.3f} rad RMS'
      + ('   <- 全 0，等价于原来的无像差圆孔' if PROBE_ABERR == 0 and ABERR_ROUGH == 0 else ''))

"""

# ===================================================================
# [独立 cell] 探针参数化对照实验（含 scan_z_probe 的 z 粗扫）
# ===================================================================
CELL_COMPARE = r"""
# ============ 探针参数化对照：自由复数矩阵 vs 光瞳面 Zernike ============
#
# 自由探针 3922 个有效未知量，Zernike 探针 12 个。低剂量/低重叠下探针通常比
# 物体先崩，所以把探针未知量压掉两个数量级，很可能比在物体上加 TGV 更管用。
#
# 【已实测的结论】PROBE_ABERR=0.1、ABERR_ROUGH=0、25 点扫描、8 stage：
#     配置                     SSIM     物体err   探针err   系数err
#     free   (z 错 30%)       0.7009    0.1637    0.0954      -
#     zernike(z 错 30%)       0.4635    0.2669    0.2250    0.0275
#     zernike(z 用真值)       0.8274    0.1245    0.0372    0.0111
# 参数化确实赢（SSIM +0.13，探针误差降到 1/2.6），但【只有 z 对了才赢】——
# z 错的时候它比自由探针还差。这就是下面必须先扫 z 的原因。
#
# 【三件事必须先做对，否则结论是假的】
#
# 1. 真值必须有像差。cell 2 里 PROBE_ABERR=0 时真值光瞳恰好无像差，
#    Zernike 模型令全部系数=0 就是精确解 —— inverse crime。本 cell 会强制检查。
#
# 2. z 必须自己拟合。实测光瞳离焦只能吸收 2% 的 z 误差（深近场，菲涅耳数 17，
#    传播产生的是环带不是二次相位）。自由探针有 3922 个自由度可以硬吃掉
#    z_probe_init 那 30% 的误差，Zernike 探针只有 12 个、吃不掉。
#
# 3. 要分"真值在张成空间内"和"部分在空间外"两种情形。只测前者是在测方差降低
#    （必然有利于参数化）；后者才是诚实的偏差-方差权衡。
#    实测 ABERR_ROUGH=0.3 时，即使代入真值系数，探针也仍有 29.1% 的残差 ——
#    那就是这个模型加再多数据也消不掉的偏差下限。
#
# 【lr_PRB】实测 3e-2 最好（8 stage）：3e-2 -> SSIM 0.8274，1e-1 -> 0.7402。
# 与自由探针用同一个值，不需要单独调。

ZC_STAGES   = 12
ZC_ZSCAN    = np.linspace(2.4e-3, 4.2e-3, 19)   # z 的粗扫范围（真值 3.0mm）
ZC_NOISE    = 0.0        # >0 则加泊松噪声，数值 = 峰值光子数（例如 1e3）
ZC_LOWOVER  = False      # True 则改用 4x4/step27（面积重叠 39%）低重叠扫描

assert PROBE_ABERR > 0 or ABERR_ROUGH > 0, (
    'cell 2 里 PROBE_ABERR 和 ABERR_ROUGH 都是 0 —— 真值光瞳无像差，\n'
    'Zernike 模型全零系数即精确解，这是 inverse crime。\n'
    '建议 PROBE_ABERR=0.5（空间内）或再加 ABERR_ROUGH=0.3（空间外），重跑 cell 2->5。')


def scan_z_probe(zs, verbose=True):
    """粗扫传播距离：对每个候选 z，用【真值物体】不可用，所以退而求其次 ——
    只比较各 z 下无像差探针的前向数据残差。这只是给 Zernike 拟合一个像样的起点，
    像差本身随后由系数吸收。"""
    y0 = np.sqrt(np.maximum(input2, 0))
    out = []
    for z in zs:
        P = propagate_np(pupil_mask.astype(np.complex64), z)
        P = P/np.abs(P).max()*support
        r = 0.0
        for i, pp in enumerate(positions):
            _, U = forward_np(np.ones((N, N), np.complex64), P, pp, H_true)
            r += float(np.sum((np.abs(U) - y0[i, :, :, 0])**2))
        out.append(r)
    out = np.array(out)
    zb_ = float(zs[int(np.argmin(out))])
    if verbose:
        print(f'  z 粗扫: 最优 {zb_*1e3:.2f} mm  (真值 {z_probe*1e3:.2f}, '
              f'初值 {z_probe_init*1e3:.2f})  残差比 {out.min()/out.max():.3f}')
    return zb_, out


# ---------------- 可选：噪声 / 低重叠 ----------------
_clean = input2.copy()
_saved = (positions.copy(), batchsize, input0.copy(), input1.copy())


def _regen_scan(n_pos, step):
    """重新生成扫描位置与衍射图。模型不烘焙位置/batch，所以只需换这三个输入。"""
    global positions, batchsize, input0, input1, input2
    positions = make_scan_positions(n_pos=n_pos, step=step)
    batchsize = len(positions)
    check_scan_fits(positions)
    input0 = np.ones((batchsize, N_OBJ, N_OBJ, 1), np.float32)
    input1 = positions.reshape(batchsize, 2, 1, 1).astype(np.float32)
    input2 = np.empty((batchsize, N, N, 1), np.float32)
    for i, pp in enumerate(positions):
        _, U = forward_np(obj, probe, pp, H_true)
        input2[i, :, :, 0] = np.abs(U)**2
    scan_report(positions)


zc = {}
try:
    if ZC_LOWOVER:
        print('低重叠模式: 4x4 网格 step 26.7')
        _regen_scan(16, (N - probe_dia)/3)
        _clean = input2.copy()
    if ZC_NOISE > 0:
        _s = ZC_NOISE/float(_clean.max())
        input2 = (np.random.default_rng(4242)
                  .poisson(np.maximum(_clean, 0)*_s).astype(np.float32)/_s)
        print(f'泊松噪声: 峰值 {ZC_NOISE:g} 光子  幅值相对误差 '
              f'{np.linalg.norm(np.sqrt(input2)-np.sqrt(_clean))/np.linalg.norm(np.sqrt(_clean)):.1%}')

    # -------- A: 自由复数探针 --------
    print(f'\n[A] free  (探针未知量 {2*int((support>0).sum())})')
    PROBE_MODEL = 'free'
    PROBE_LAYERS = ['Pr', 'Pi']
    mA = Ptycho(); roA, rpA = mA.train(ZC_STAGES, verbose=0)
    zc['free'] = (roA, rpA, get_object(mA.OBJ), get_probe(mA.OBJ))

    # -------- B: Zernike，用初值 z（不拟合，作为反面对照） --------
    print(f'\n[B] zernike, z 不拟合 ({z_probe_init*1e3:.2f} mm)  (探针未知量 {K_ZERN})')
    PROBE_MODEL = 'zernike'
    PROBE_LAYERS = ['Zc']
    z_probe_model = z_probe_init
    PZ_NORM = float(np.abs(propagate_np(pupil_mask.astype(np.complex64), z_probe_model)).max())
    mB = Ptycho(); roB, rpB = mB.train(ZC_STAGES, verbose=0)
    zc['zernike (z 未拟合)'] = (roB, rpB, get_object(mB.OBJ), get_probe(mB.OBJ))
    cB = get_zcoef(mB.OBJ)

    # -------- C: Zernike，先粗扫 z --------
    print('\n[C] zernike, z 粗扫后拟合')
    z_best, zcurve = scan_z_probe(ZC_ZSCAN)
    z_probe_model = z_best
    PZ_NORM = float(np.abs(propagate_np(pupil_mask.astype(np.complex64), z_probe_model)).max())
    mC = Ptycho(); roC, rpC = mC.train(ZC_STAGES, verbose=0)
    zc['zernike (z 已拟合)'] = (roC, rpC, get_object(mC.OBJ), get_probe(mC.OBJ))
    cC = get_zcoef(mC.OBJ)
finally:
    input2 = _clean
    positions, batchsize, input0, input1 = _saved
    PROBE_MODEL = 'free'
    PROBE_LAYERS = ['Pr', 'Pi']
    z_probe_model = z_probe_init
    PZ_NORM = float(np.abs(propagate_np(pupil_mask.astype(np.complex64), z_probe_model)).max())
    print('\n(已恢复 free / 无噪声 / 原扫描)')

# ---------------- 汇总 ----------------
print('\n' + '='*78)
print(f'{"配置":<22s}{"探针未知量":>11s}{"物体SSIM":>10s}{"物体复场err":>12s}{"探针err":>10s}')
print('-'*78)
_nfree = 2*int((support > 0).sum())
for t, (ro, rp, _o, _p) in zc.items():
    n = _nfree if t == 'free' else K_ZERN
    print(f'{t:<22s}{n:>11d}{ro["ssim_o_amp"][-1]:>10.4f}'
          f'{ro["relerr_o_complex"][-1]:>12.4f}{rp["relerr_p_complex"][-1]:>10.4f}')
print('='*78)

# ---------------- 系数恢复 ----------------
# 表示下限：把真值光瞳相位投影到基底上，投影残差就是这个模型的偏差底。
_phi_gt = np.angle(pupil_true)
_c_proj = ic.zernike_fit(_phi_gt, Zb, pupil_mask)
_res = _phi_gt - Zb @ _c_proj
print(f'\n真值光瞳相位在基底上的投影残差 {np.sqrt(np.mean(_res[pupil_mask]**2)):.4f} rad RMS')
print('  <- 这是 Zernike 模型的【偏差下限】，探针误差低于它对应的水平就是过拟合了')
print(f'\n{"模式":>8s}{"真值":>9s}{"投影":>9s}{"z未拟合":>10s}{"z已拟合":>10s}')
for i, (n, m) in enumerate(ZLAB):
    print(f'{f"Z({n},{m:+d})":>8s}{coef_true[i]:>9.3f}{_c_proj[i]:>9.3f}'
          f'{cB[i]:>10.3f}{cC[i]:>10.3f}')
print(f'{"RMS误差":>8s}{"":>9s}{"":>9s}'
      f'{np.sqrt(np.mean((cB-_c_proj)**2)):>10.3f}'
      f'{np.sqrt(np.mean((cC-_c_proj)**2)):>10.3f}')

# ---------------- 图 ----------------
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
axes[0, 0].plot(ZC_ZSCAN*1e3, zcurve/zcurve.max(), 'o-', ms=3)
axes[0, 0].axvline(z_probe*1e3, color='g', ls='--', label=f'真值 {z_probe*1e3:.1f}')
axes[0, 0].axvline(z_best*1e3, color='r', ls=':', label=f'扫描最优 {z_best*1e3:.2f}')
axes[0, 0].set_xlabel('z (mm)'); axes[0, 0].set_ylabel('归一化数据残差')
axes[0, 0].set_title('z 粗扫', fontsize=10); axes[0, 0].legend(fontsize=7); axes[0, 0].grid(alpha=.3)

for ax, key, ttl in [(axes[0, 1], 'ssim_o_amp', '物体幅值 SSIM'),
                     (axes[0, 2], 'relerr_o_complex', '物体复场误差')]:
    for t, (ro, rp, _o, _p) in zc.items():
        ax.plot(range(1, ZC_STAGES+1), ro[key], 'o-', ms=3, label=t)
    ax.set_xlabel('Stage'); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7)
for t, (ro, rp, _o, _p) in zc.items():
    axes[0, 3].semilogy(range(1, ZC_STAGES+1), rp['relerr_p_complex'], 'o-', ms=3, label=t)
axes[0, 3].set_xlabel('Stage'); axes[0, 3].set_title('探针复场误差', fontsize=10)
axes[0, 3].grid(alpha=.3); axes[0, 3].legend(fontsize=7)

_c0 = EVAL_CROP
_gt = np.abs(obj)[_c0:-_c0, _c0:-_c0]
for ax, t in zip(axes[1, :3], zc):
    r, _ = ic.align_global_factor(zc[t][2][_c0:-_c0, _c0:-_c0], obj[_c0:-_c0, _c0:-_c0])
    ax.imshow(np.abs(r), cmap='gray', vmin=_gt.min(), vmax=_gt.max())
    ax.set_title(f'{t}\nSSIM={zc[t][0]["ssim_o_amp"][-1]:.4f}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
axes[1, 3].imshow(_gt, cmap='gray'); axes[1, 3].set_title('Ground truth', fontsize=9)
axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])
fig.tight_layout(); plt.show()"""
