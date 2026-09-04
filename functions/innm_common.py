"""
functions/innm_common.py —— INNM_FPM.ipynb 与 INNM_Ptycho.ipynb 的公共模块

放在这里的都是与"前向模型是 FPM 还是 ptychography"无关的东西：
  - 数值安全工具 (safe_angle)
  - 自定义可训练层 (MyLayer_OBJ / MyLayer_PUP)
  - 评估指标 (ssim / psnr / evaluate_object / evaluate_pupil)
  - 可替换的正则项 (TV / 各向异性TV / TGV / Hessian / 峰度)
  - 结果绘图与多组实验对比

与 notebook 里的版本相比，评估函数改成【接收显式参数】而不是依赖全局变量，
这样两个 notebook 才能共用（它们的网格尺寸、照明函数、真值对象都不一样）。

用法：
    import innm_common as ic
    ic.silence_known_tf_warnings()
    loss = ic.make_tgv_loss(w1=1.0, w2=2.0)
"""

import logging
from math import factorial
import numpy as np
import tensorflow as tf
from scipy.ndimage import uniform_filter
from tensorflow.keras.layers import Layer

__all__ = [
    'silence_known_tf_warnings', 'safe_angle', 'fourier_shift_np', 'fourier_shift_tf',
    'crop_patch_np', 'crop_patch_tf',
    'MyLayer_OBJ', 'MyLayer_PUP', 'MyLayer_COEF',
    'ssim', 'psnr', 'align_global_factor', 'evaluate_object', 'evaluate_probe', 'evaluate_pupil',
    'kurtosis_np', 'zernike_fit', 'zernike_basis', 'zernike_pupil_np',
    'make_tv_loss', 'make_aniso_tv_loss', 'make_tgv_loss', 'make_hessian_loss',
    'kurtosis_loss', 'make_kurtosis_target_loss',
    'plot_records', 'print_records', 'compare_runs',
    'EVAL_CROP', 'PHASE_SUPPORT',
]

EVAL_CROP = 10      # 评估时四周裁掉的像素数（避开 FFT 循环卷积的边缘振铃）
PHASE_SUPPORT = 0.05  # 只在 |O_gt| > 5% * max 的区域评估相位


# =============================================================================
# 数值安全
# =============================================================================

class _DropKnownNoise(logging.Filter):
    """精确匹配两类已确认无害的刷屏警告，其它 TF 警告照常显示。

    1) "...discard the imaginary part..." —— 来自反向传播：
       tf.cast(实数 -> complex64) 的梯度规则就是「取梯度的实部」，
       TF 内部用 tf.cast(complex -> float) 实现，于是每个复数节点都 warn 一次。
       丢虚部在这里是数学上正确的做法。
    2) "The name tf.xxx is deprecated..." —— keras/src/ 内部代码自己在用 v1 老符号。
    """
    _NOISE = ("discard the imaginary part", "is deprecated. Please use tf.compat.v1")

    def filter(self, record):
        return not any(s in record.getMessage() for s in self._NOISE)


def silence_known_tf_warnings():
    tf.get_logger().addFilter(_DropKnownNoise())


def safe_angle(z, eps=1e-20):
    """tf.math.angle 在 z=0 处的梯度是 -inf，会让整个训练变 nan。

    泊松噪声会在测量图里制造大量精确零（实测 peak=5000 光子、全局归一化时
    约 15% 的像素为 0），这些零经 sqrt 继承到物体估计里，再被 angle 一碰就炸。

    这里保证 angle 永远不作用在 0 上：近零点先替换成 1+0j，算完再把输出压回 0。
    两个 tf.where 缺一不可 —— 只在输出端 mask 是不行的，因为 tf.where 的两个分支
    都会被求导，0 * inf 仍然产生 nan（tf.where 的经典陷阱）。
    """
    re = tf.math.real(z)
    im = tf.math.imag(z)
    mag2 = re * re + im * im
    ok = mag2 > eps
    one = tf.complex(tf.ones_like(mag2), tf.zeros_like(mag2))
    z_safe = tf.where(ok, z, one)
    return tf.where(ok, tf.math.angle(z_safe), tf.zeros_like(mag2))


# =============================================================================
# 亚像素平移（傅里叶相位斜坡）
#
# 平移定理：g(x) = f(x - r)  <=>  G(k) = F(k)·exp(-i2π·k·r/N)
# 位移量为整数时与 np.roll 逐位一致（实测相对误差 ~1e-15），但支持小数位移。
#
# 为什么需要它：ptychography 的扫描位置是连续的物理量，取整会引入最多 0.5 px 误差。
# 实测抖动光栅取整后平均位置误差 0.235 px，重建质量从 SSIM 0.909 掉到 0.731。
#
# 附带好处：相位斜坡靠广播生成，不需要 tf.roll 那种 tf.reshape(pos,[2])，
# 于是 batch_size=1 的限制自然解除。
# =============================================================================

def fourier_shift_np(f, rx, ry):
    """把二维复数场 f 平移 (rx, ry) 像素，支持小数。"""
    n0, n1 = f.shape[:2]
    fx = np.fft.fftfreq(n0).reshape(n0, 1)
    fy = np.fft.fftfreq(n1).reshape(1, n1)
    return np.fft.ifft2(np.fft.fft2(f) * np.exp(-2j * np.pi * (fx * rx + fy * ry)))


_SHIFT_GRID_CACHE = {}


def _shift_grids(n):
    """缓存的是 numpy 数组而【不是】tf.constant。

    tf.constant 会绑定到创建它的那张图上；本代码要建两个模型（OBJ / PRB），
    缓存的常量在第二个模型里就会报 "is out of scope and cannot be used here"。
    存 numpy 让 TF 每次在当前图里自行转换，开销可忽略（常量折叠）。
    """
    if n not in _SHIFT_GRID_CACHE:
        _SHIFT_GRID_CACHE[n] = (
            np.fft.fftfreq(n).reshape(1, n, 1, 1).astype(np.float32),
            np.fft.fftfreq(n).reshape(1, 1, n, 1).astype(np.float32),
        )
    return _SHIFT_GRID_CACHE[n]


def fourier_shift_tf(field_c, pos, n):
    """field_c: (batch, n, n, 1) complex64 ; pos: (batch, 2, 1, 1) float32（像素，可小数）"""
    fx, fy = _shift_grids(n)
    ph = -2 * np.pi * (fx * pos[:, 0:1, :, :] + fy * pos[:, 1:2, :, :])
    ramp = tf.complex(tf.cos(ph), tf.sin(ph))
    return tf.signal.ifft3d(tf.signal.fft3d(field_c) * ramp)


# =============================================================================
# 局部裁剪（crop patch）—— AD ptychography 的主流做法
#
# 物体画布比探测器大，每个扫描位置从画布上裁出一块 n×n 送进前向模型。
# 相比"整幅物体做傅里叶相位斜坡平移"的写法：
#   - 没有循环边界。相位斜坡隐含地把物体当成周期的，画布边缘会互相污染；
#     裁剪不会，物体可以是任意大的非周期场。
#   - FFT 只在 patch 上做。画布再大也不增加前向成本，而相位斜坡是 O(N_obj^2 log N_obj)。
#   - 与 tike / PtyRAD / ptychodus 等主流 AD 实现一致。
#
# 小数位置仍然用相位斜坡，但只作用在裁出来的 patch 上（tike/PtyRAD 的做法）。
# patch 边缘 1px 的循环误差落在探针支撑域之外，会被探针的零值吃掉。
#
# 【可微性】整数部分经 floor，梯度为 0；小数部分完全可微。所以扫描位置对
# 数据项仍然有梯度 —— position refinement 只要把 corner 变成可训练量即可。
# =============================================================================

def crop_patch_np(canvas, corner, n):
    """canvas: (N_obj, N_obj) complex ; corner: (2,) 浮点左上角。返回 (n, n)。"""
    c = np.asarray(corner, float)
    i = np.floor(c).astype(int)
    f = c - i
    assert i.min() >= 0 and i[0]+n <= canvas.shape[0] and i[1]+n <= canvas.shape[1], \
        f'patch 越界: corner={c}, n={n}, canvas={canvas.shape}'
    w = canvas[i[0]:i[0]+n, i[1]:i[1]+n]
    return w if not f.any() else fourier_shift_np(w, -f[0], -f[1])


def crop_patch_tf(canvas, corner, n):
    """canvas: (batch, N_obj, N_obj, 1) complex64 ; corner: (batch, 2, 1, 1) float32。

    整数部分用 tf.gather 取行再取列 —— 可微，反向是 scatter-add 回画布上对应的窗口，
    这正是 ePIE 那类算法手写的"把更新贴回物体"，只不过由 autodiff 自动完成。
    """
    ij = tf.floor(corner)
    fr = corner - ij
    ij = tf.cast(ij, tf.int32)[:, :, 0, 0]                 # (batch, 2)
    ar = tf.range(n, dtype=tf.int32)[None, :]              # (1, n)
    x = canvas[:, :, :, 0]                                 # (batch, N_obj, N_obj)
    x = tf.gather(x, ij[:, 0:1] + ar, axis=1, batch_dims=1)   # (batch, n, N_obj)
    x = tf.gather(x, ij[:, 1:2] + ar, axis=2, batch_dims=1)   # (batch, n, n)
    return fourier_shift_tf(x[:, :, :, None], -fr, n)


# =============================================================================
# 自定义层：把「待求的未知量」伪装成「层权重」
# =============================================================================

class MyLayer_OBJ(Layer):
    """输出 = kernel（广播到 batch），于是 kernel 本身就是待优化的未知量。
    Keras 的层必须有输入，这是原作者绕开该限制的写法。

    【只借 x 的 batch 维】原来写的是 x * kernel，要求 dummy 输入与 kernel 同尺寸；
    改成与 MyLayer_COEF 一致的写法之后，同一个 dummy 输入可以同时喂尺寸不同的层
    （ptycho 里物体画布是 N_OBJ、探针是 N）。加 0 与乘全 1 逐位等价。"""

    def __init__(self, output_dims, **kwargs):
        self.output_dims = output_dims
        super(MyLayer_OBJ, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=self.output_dims,
                                      initializer='ones', trainable=True)
        super(MyLayer_OBJ, self).build(input_shape)

    def call(self, x):
        return self.kernel + tf.zeros_like(x[:, :1, :1, :1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + tuple(self.output_dims)


class MyLayer_PUP(MyLayer_OBJ):
    """与 MyLayer_OBJ 完全相同，仅为保持命名区分（ptycho 的探针 / FPM 的光瞳）。"""
    pass


class MyLayer_COEF(Layer):
    """持有 K 个标量系数的可训练层，输出形状 (batch, 1, 1, K)。

    与 MyLayer_OBJ 是同一个技巧（输入只用来提供 batch 维），区别在于权重不是
    一整张图而是 K 个数 —— 用于把探针从 2N^2 个自由像素压缩成 K 个基底系数。

    权重形状写成 (1,1,1,K) 而不是 (K,)，是为了能直接和 (1,N,N,K) 的固定基底
    广播相乘，省掉 reshape。
    """

    def __init__(self, n_coef, init=None, **kwargs):
        self.n_coef = int(n_coef)
        self.init_val = (np.zeros(self.n_coef, np.float32) if init is None
                         else np.asarray(init, np.float32).reshape(self.n_coef))
        super(MyLayer_COEF, self).__init__(**kwargs)

    def build(self, input_shape):
        v = self.init_val.reshape(1, 1, 1, self.n_coef)
        self.kernel = self.add_weight(
            name='kernel', shape=(1, 1, 1, self.n_coef), dtype='float32',
            initializer=lambda shape, dtype=None: tf.constant(v, dtype='float32'),
            trainable=True)
        super(MyLayer_COEF, self).build(input_shape)

    def call(self, x):
        # 只借 x 的 batch 维；x 的数值不参与运算
        return self.kernel + tf.zeros_like(x[:, :1, :1, :1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, self.n_coef)

    def get_config(self):
        cfg = super(MyLayer_COEF, self).get_config()
        cfg.update({'n_coef': self.n_coef})
        return cfg


# =============================================================================
# 评估指标
# =============================================================================

def ssim(x, y, data_range=None, win_size=7):
    """结构相似性。与 skimage.metrics.structural_similarity 的默认参数
    (win_size=7, uniform 窗) 逐值一致，实现在此以避免引入 scikit-image 依赖。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if data_range is None:
        data_range = max(x.max(), y.max()) - min(x.min(), y.min())
    if data_range == 0:
        return 1.0
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    NP = win_size ** x.ndim
    cov_norm = NP / (NP - 1)
    f = lambda a: uniform_filter(a, size=win_size)
    ux, uy = f(x), f(y)
    vx = cov_norm * (f(x * x) - ux * ux)
    vy = cov_norm * (f(y * y) - uy * uy)
    vxy = cov_norm * (f(x * y) - ux * uy)
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux ** 2 + uy ** 2 + C1) * (vx + vy + C2))
    pad = (win_size - 1) // 2
    return float(S[pad:-pad, pad:-pad].mean())


def psnr(rec, gt, data_range=None):
    rec = np.asarray(rec, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if data_range is None:
        data_range = gt.max() - gt.min()
    mse = np.mean((rec - gt) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(data_range ** 2 / mse))


def _crop(a, n):
    return a if n <= 0 else a[n:a.shape[0] - n, n:a.shape[1] - n]


def align_global_factor(rec, gt):
    """消去全局复数因子（同时含幅值尺度与相位偏移）：
    min_c ||c*rec - gt||^2 的闭式解 c = <rec,gt> / <rec,rec>

    相位恢复问题天然存在这个自由度 —— 整个结果乘一个复常数不改变任何测量值，
    不先消掉它，PSNR/SSIM 这些指标算出来是假的。"""
    denom = np.vdot(rec, rec)
    if denom == 0:
        return rec, 0j
    c = np.vdot(rec, gt) / denom
    return rec * c, c


def kurtosis_np(x):
    """标准（非超额）峰度，KACP 式(5)。用于 KACP 风格的自动对焦搜索。"""
    x = np.asarray(x, dtype=np.float64)
    d = x - x.mean()
    m2 = np.mean(d ** 2)
    if m2 == 0:
        return float('nan')
    return float(np.mean(d ** 4) / (m2 ** 2))


def evaluate_object(rec, gt, crop=EVAL_CROP, phase_support=PHASE_SUPPORT):
    """物体重建质量。rec / gt 都是复数二维数组。

    键名沿用 INNM 原作者的命名（_o_ = object）。
    相位用幅值掩膜：|O|≈0 的像素相位是纯噪声，两边一起置零，
    否则完美重建的自检都拿不到满分。
    """
    rec, _ = align_global_factor(rec, gt)
    r = _crop(rec, crop)
    g = _crop(gt, crop)
    ra, ga = np.abs(r), np.abs(g)
    dr = ga.max() - ga.min()

    mask = ga > phase_support * ga.max()
    pr = np.where(mask, np.angle(r), 0.0)
    pg = np.where(mask, np.angle(g), 0.0)
    dphi = np.angle(np.exp(1j * (np.angle(r) - np.angle(g))))[mask]  # 缠绕到 [-pi, pi]
    pdr = max(pg.max() - pg.min(), 1e-12)
    return {
        'psnr_o_amp': psnr(ra, ga, dr),
        'ssim_o_amp': ssim(ra, ga, dr),
        'psnr_o_phi': psnr(pr, pg, pdr),
        'ssim_o_phi': ssim(pr, pg, pdr),
        'rmse_o_phi_rad': float(np.sqrt(np.mean(dphi ** 2))) if mask.any() else float('nan'),
        'relerr_o_complex': float(np.linalg.norm(r - g) / np.linalg.norm(g)),
    }


def evaluate_probe(rec_amp, rec_phi, gt_field, mask, coef=None, coef_gt=None):
    """照明函数的恢复质量 —— ptychography 里是【探针 P(r)】，FPM 里是【光瞳 P(k)】。

    rec_amp / rec_phi : 恢复出的振幅与相位（二维实数组）
    gt_field          : 真值复数探针/光瞳
    mask              : 支撑域布尔掩膜（ptycho 用探针孔径，FPM 用 CTF>0）
    coef / coef_gt    : 可选，Zernike 系数与真值系数，给出则额外算系数 RMSE

    相位只在支撑域内比较，并去掉 piston（常数项）—— 它和全局相位一样是自由度。
    返回键名统一用 _p_（probe/pupil）。
    """
    gt_amp = np.abs(gt_field)
    gt_phi = np.angle(gt_field) * mask
    p_rec = rec_phi[mask] - rec_phi[mask].mean()
    p_gt = gt_phi[mask] - gt_phi[mask].mean()
    d = np.angle(np.exp(1j * (p_rec - p_gt)))
    dr = max(gt_amp.max() - gt_amp.min(), 1e-12)
    pdr = max(gt_phi.max() - gt_phi.min(), 1e-12)
    # 复场相对误差：自由复数探针的首选指标（Zernike 参数化时没有系数可比）
    rec_c = rec_amp * np.exp(1j * rec_phi) * mask
    rec_al, _ = align_global_factor(rec_c, gt_field * mask)
    denom = np.linalg.norm(gt_field * mask)
    out = {
        'psnr_p_amp': psnr(rec_amp, gt_amp, dr),
        'ssim_p_amp': ssim(rec_amp, gt_amp, dr),
        'psnr_p_phi': psnr(rec_phi * mask, gt_phi, pdr),
        'ssim_p_phi': ssim(rec_phi * mask, gt_phi, pdr),
        'rms_p_phi_rad': float(np.sqrt(np.mean(d ** 2))),
        'relerr_p_complex': float(np.linalg.norm(rec_al - gt_field * mask) / denom) if denom else float('nan'),
    }
    if coef is not None and coef_gt is not None:
        out['rmse_p_zernike'] = float(np.sqrt(np.mean((np.asarray(coef) - np.asarray(coef_gt)) ** 2)))
    return out


# 兼容别名：FPM 侧习惯叫 pupil
evaluate_pupil = evaluate_probe


def _zernike_nm(n_max):
    """OSA/ANSI 顺序枚举 (n, m)：n 从 0 到 n_max，m 从 -n 步进 2 到 n。"""
    return [(n, m) for n in range(n_max + 1) for m in range(-n, n + 1, 2)]


def _zernike_radial(n, m, rho):
    m = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - m)//2 + 1):
        c = ((-1)**k * factorial(n - k) /
             (factorial(k)*factorial((n + m)//2 - k)*factorial((n - m)//2 - k)))
        R += c*rho**(n - 2*k)
    return R


def zernike_basis(n_grid, radius, n_max=4, drop=(0, 1, 2), center=None):
    """在 n_grid x n_grid 网格上生成 Zernike 基底（OSA/ANSI 顺序、Noll 归一化）。

    返回 (basis, mask, labels)：
      basis  (n_grid, n_grid, K) float32，圆盘外为 0
      mask   (n_grid, n_grid) bool，半径 radius（像素）的圆盘
      labels [(n, m), ...] 长度 K

    【Noll 归一化】每个模式在单位圆盘上的 RMS = 1。这不是美观问题：
    未归一化时不同模式的幅度能差 8 倍，而 Adam 对所有系数用同一个步长，
    等于偷偷给低阶模式加了 8 倍学习率。项目根目录那个 Zernike_Polyminals.mat
    就没归一化（Gram 对角线 0.13~1.0），且它的圆盘半径只有 10.6 px
    （给 FPM 的 NA 光瞳用的），与 ptycho 的探针孔径对不上，不能直接复用。

    【drop】要剔除的 OSA 序号，默认 (0,1,2) = 活塞 + 两个倾斜：
      活塞与物体的全局相位简并（align_global_factor 正在消它）
      倾斜与扫描位置的整体平移简并
    留着不会报错，但那几个方向在数据里完全不受约束，纯粹是噪声的自由度。

    离散化误差：radius=18 时圆盘只有约 1009 个像素，实测 Gram 对角 0.94~1.00、
    非对角最大 0.054。半径越小越不正交，radius < 10 就不要用高阶模式了。
    """
    c = (n_grid/2.0, n_grid/2.0) if center is None else center
    yy, xx = np.mgrid[0:n_grid, 0:n_grid].astype(np.float64)
    x = (xx - c[1])/radius
    y = (yy - c[0])/radius
    rho = np.hypot(x, y)
    th = np.arctan2(y, x)
    mask = rho <= 1.0

    nm = _zernike_nm(n_max)
    drop = set(drop)
    keep = [j for j in range(len(nm)) if j not in drop]
    out = np.zeros((n_grid, n_grid, len(keep)), np.float32)
    for i, j in enumerate(keep):
        n, m = nm[j]
        R = _zernike_radial(n, m, np.where(mask, rho, 0.0))
        Z = R*np.cos(m*th) if m > 0 else (R*np.sin(-m*th) if m < 0 else R)
        Z = Z*np.sqrt((n + 1.0)*(1.0 if m == 0 else 2.0))     # Noll 归一化
        out[:, :, i] = np.where(mask, Z, 0.0)
    return out, mask, [nm[j] for j in keep]


def zernike_pupil_np(basis, mask, coef):
    """由系数合成光瞳复场：mask * exp(i * sum_k coef_k * basis_k)。
    与模型内部的 TF 版本逐值一致，用于生成真值 / 取出重建结果。"""
    phi = basis @ np.asarray(coef, np.float32)
    return (mask*np.exp(1j*phi)).astype(np.complex64)


def zernike_fit(phase_map, zernike_poly, mask):
    """把任意相位图最小二乘投影到给定 Zernike 基底。

    真值离焦相位并不严格落在有限个 Zernike 模式张成的空间里，
    只有先投影才能和恢复出的系数做公平对比。
    投影残差就是该模型的表示下限 —— 恢复误差低于它说明过拟合了。
    """
    coef, *_ = np.linalg.lstsq(zernike_poly[mask], np.asarray(phase_map)[mask], rcond=None)
    return coef


# =============================================================================
# 可替换的正则项
#
# 全部做成"工厂函数"：调用后返回符合 Keras 签名 (y_true, y_pred) 的闭包，
# 超参数写在调用处，不用改函数体。
#
# 【重要】reduce 默认 'sum'，与 INNM 原版 my_loss 一致。改成 'mean' 数值会小
#   约 1.6e4 倍（实测 TGV: 3769 -> 0.2326），此时沿用原来的 tv1/tv2 等于把正则关掉。
#
# 【梯度安全】除各向同性 TV 需要 eps 之外，其余都用 tf.abs，
#   在 0 处梯度为 0（TF 定义 sign(0)=0），不会像 tf.sqrt 那样产生 inf。
# =============================================================================

def _grad_xy(u):
    """一阶前向差分。u 形状 (batch, H, W, 1)"""
    dx = u[:, 1:, :] - u[:, :-1, :]
    dy = u[:, :, 1:] - u[:, :, :-1]
    return dx, dy



def make_tv_loss(beta=1e-2, iso=True, eps=1e-8, reduce='sum'):
    """标准总变差。

    iso=True  -> 各向同性 sqrt(dx^2+dy^2)，与 INNM 原版 my_loss 完全等价
                 （实测数值逐位一致）
    iso=False -> 各向异性 |dx|+|dy|（曼哈顿几何，偏好横平竖直的边缘）

    注意：INNM 原版是【各向同性】的，所以换成 iso=False 是真实改动，不是等价重写。
    eps 必须保留：sqrt 在 0 处导数是 inf，初始化时相位几乎处处为 0，第一步就会 nan。
    """
    R = tf.reduce_sum if reduce == 'sum' else tf.reduce_mean

    def loss(y_true, y_pred):
        dx, dy = _grad_xy(y_pred)
        if iso:
            a = tf.square(dx[:, :, :-1])
            b = tf.square(dy[:, :-1, :])
            return beta * R(tf.sqrt(a + b + eps))
        return beta * (R(tf.abs(dx)) + R(tf.abs(dy)))
    return loss


def make_aniso_tv_loss(beta=1e-2, reduce='sum'):
    """各向异性 L1-TV：|dx| + |dy|。偏好水平/垂直边缘，
    适合半导体掩模、光栅这类横平竖直的结构。"""
    return make_tv_loss(beta=beta, iso=False, reduce=reduce)


def make_tgv_loss(w1=1.0, w2=2.0, beta=1e-2, reduce='sum'):
    """TGV（二阶总变差）的实用近似：一阶 TV + 二阶 TV 加权。

    治的是 TV 的"阶梯效应"：TV 偏好分片常数，会把本该平滑渐变的区域压成台阶。
    加入二阶项后偏好变成分片【线性】，渐变得以保留。
    w2/w1 越大越偏向平滑渐变；生物样品（相位连续变化）可试 w2=3~5。
    """
    R = tf.reduce_sum if reduce == 'sum' else tf.reduce_mean

    def loss(y_true, y_pred):
        dx, dy = _grad_xy(y_pred)
        tv1 = R(tf.abs(dx)) + R(tf.abs(dy))
        dxx = dx[:, 1:, :] - dx[:, :-1, :]
        dyy = dy[:, :, 1:] - dy[:, :, :-1]
        tv2 = R(tf.abs(dxx)) + R(tf.abs(dyy))
        return beta * (w1 * tv1 + w2 * tv2)
    return loss


def make_hessian_loss(beta=1e-2, reduce='sum'):
    """Hessian 正则：惩罚二阶导各分量（L1 形式，比 L2 更保边）。
    uxy 前的系数 2 来自 Hessian 对称、非对角元出现两次。
    与 TGV 很接近，建议当作 TGV 的对照组。"""
    R = tf.reduce_sum if reduce == 'sum' else tf.reduce_mean

    def loss(y_true, y_pred):
        dx, dy = _grad_xy(y_pred)
        uxx = dx[:, 1:, :] - dx[:, :-1, :]
        uyy = dy[:, :, 1:] - dy[:, :, :-1]
        uxy = dx[:, :, 1:] - dx[:, :, :-1]
        return beta * (R(tf.abs(uxx)) + R(tf.abs(uyy)) + 2 * R(tf.abs(uxy)))
    return loss


def kurtosis_loss(y_true, y_pred, beta=1e-2):
    """最大化强度分布峰度（KACP 式(5) 的可微版本）。

    【警告：这一项无下界】峰度上界是像素数 n（图像塌缩成单个尖峰时取到），
    所以能一路跌到 -beta*n，优化器会去薅这个免费的下降空间。
    实测 3 个 stage 后强度峰度从 6.3 涨到 13843（真值 2.287），
    重建 max/mean 冲到 24.8（真值 2.1）—— loss 在降但重建在烂。

    另注：只应用在幅值输出上。对相位取平方不是任何物理量。

    更根本的问题：峰度是【选择准则】而非【惩罚项】。TV 的最优解是分片常数，
    是合理的图像先验；峰度的最优解是 delta 函数，永远不是正确答案。
    KACP 原文只用它在若干个物理上都合法的候选 z 之间做 argmax，
    从不让它去形变图像 —— 建议照原文那样用。
    """
    I = tf.square(y_pred)
    I = I / (tf.reduce_mean(I) + 1e-12)   # 仅数值意义：防 m4 溢出（峰度本身尺度不变）
    d = I - tf.reduce_mean(I)
    m2 = tf.reduce_mean(tf.square(d))
    m4 = tf.reduce_mean(tf.square(tf.square(d)))
    return -beta * m4 / (tf.square(m2) + 1e-12)


def make_kurtosis_target_loss(k_target=2.3, beta=1e-2):
    """峰度的有下界版本：不最大化峰度，而是让峰度靠近目标值。
    下界为 0，优化器无法薅到负无穷。
    k_target 可用一次粗重建估计，或取同类样品先验。"""
    def loss(y_true, y_pred):
        I = tf.square(y_pred)
        I = I / (tf.reduce_mean(I) + 1e-12)
        d = I - tf.reduce_mean(I)
        m2 = tf.reduce_mean(tf.square(d))
        m4 = tf.reduce_mean(tf.square(tf.square(d)))
        K = m4 / (tf.square(m2) + 1e-12)
        return beta * tf.square(K - k_target)
    return loss


# =============================================================================
# 结果展示
# =============================================================================

_PANELS = {
    'zh': [
        ('obj',   'psnr_o_amp',       '物体幅值 PSNR (dB)'),
        ('obj',   'ssim_o_amp',       '物体幅值 SSIM'),
        ('obj',   'rmse_o_phi_rad',   '物体相位 RMSE (rad)'),
        ('obj',   'relerr_o_complex', '复场相对误差'),
        ('probe', 'rms_p_phi_rad',    '探针/光瞳相位 RMS (rad)'),
        ('probe', 'relerr_p_complex', '探针/光瞳复场相对误差'),
    ],
    'en': [
        ('obj',   'psnr_o_amp',       'Object Amplitude PSNR (dB)'),
        ('obj',   'ssim_o_amp',       'Object Amplitude SSIM'),
        ('obj',   'rmse_o_phi_rad',   'Object Phase RMSE (rad)'),
        ('obj',   'relerr_o_complex', 'Object Complex Relative Error'),
        ('probe', 'rms_p_phi_rad',    'Probe Phase RMS (rad)'),
        ('probe', 'relerr_p_complex', 'Probe Complex Relative Error'),
    ],
}


def plot_records(record_obj, record_probe, title='', plt=None, lang='zh'):
    """把每个 stage 记录的指标画成曲线。plt 传 matplotlib.pyplot。

    lang='en' 用英文标签 —— matplotlib 默认字体没有 CJK，中文会显示成豆腐块。
    """
    if plt is None:
        import matplotlib.pyplot as plt
    xlabel = 'stage' if lang == 'zh' else 'Stage'
    plt.figure(figsize=(13, 6))
    for i, (which, key, name) in enumerate(_PANELS[lang]):
        rec = record_obj if which == 'obj' else record_probe
        v = rec.get(key, [])
        plt.subplot(2, 3, i + 1)
        if len(v):
            plt.plot(range(1, len(v) + 1), v, 'o-')
        plt.title(name, fontsize=10)
        plt.xlabel(xlabel)
        plt.grid(alpha=.3)
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def print_records(record_obj, record_probe, lang='zh'):
    """打印最终一行汇总，方便直接抄进表格。"""
    if not record_obj.get('psnr_o_amp'):
        print('（还没有记录，先跑 train）' if lang == 'zh' else '(no records yet, run train first)')
        return
    hdr = (f"{'指标':<22s}{'最终值':>12s}" if lang == 'zh'
           else f"{'metric':<22s}{'final':>12s}")
    print(hdr)
    for rec, key in [(record_obj, 'psnr_o_amp'), (record_obj, 'ssim_o_amp'),
                     (record_obj, 'psnr_o_phi'), (record_obj, 'ssim_o_phi'),
                     (record_obj, 'rmse_o_phi_rad'), (record_obj, 'relerr_o_complex'),
                     (record_probe, 'rms_p_phi_rad'), (record_probe, 'relerr_p_complex')]:
        if rec.get(key):
            print(f'{key:<22s}{rec[key][-1]:12.4f}')


def compare_runs(records, labels=None, plt=None):
    """横向对比多个保存在内存中的 object evaluation record。"""
    if plt is None:
        import matplotlib.pyplot as plt
    labels = labels or [f'run {i + 1}' for i in range(len(records))]
    keys = ['psnr_o_amp', 'ssim_o_amp', 'rmse_o_phi_rad', 'relerr_o_complex']
    names = ['幅值PSNR(dB)', '幅值SSIM', '相位RMSE(rad)', '复场相对误差']
    rows = []
    for d in records:
        rows.append([float(np.asarray(d[k]).ravel()[-1])
                     if k in d and np.asarray(d[k]).size else float('nan') for k in keys])
    if not rows:
        return
    print(f"{'配置':<16s}" + ''.join(f'{n:>16s}' for n in names))
    for lab, r in zip(labels, rows):
        print(f'{lab:<16s}' + ''.join(f'{v:16.4f}' for v in r))

    plt.figure(figsize=(13, 3.2))
    for i, (k, n) in enumerate(zip(keys, names)):
        plt.subplot(1, 4, i + 1)
        for d, lab in zip(records, labels):
            v = np.asarray(d[k]).ravel()
            if v.size:
                plt.plot(range(1, len(v) + 1), v, 'o-', label=lab)
        plt.title(n, fontsize=10)
        plt.xlabel('stage')
        plt.grid(alpha=.3)
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
