"""
峰度正则的两类失效诊断
=====================

用法（在 simulations/INNM_Ptycho.ipynb 里，跑完训练之后新开一个 cell）：

    from functions import diag_kurtosis as dk
    import importlib; importlib.reload(dk)
    dk.full_report(pty_gk, gradient_kurtosis_loss,
                   input0, input1, input2, GK_BETA, REG_CROP, EVAL_CROP, obj)

回答两个问题：
  1. 梯度更新有没有爆炸
  2. K 是不是因为方差趋零而无界增长

结论怎么读见每个函数的 docstring 末尾。
"""

import numpy as np
import tensorflow as tf


# =============================================================================
# 问题 1：梯度爆炸 / 谁在主导更新方向
# =============================================================================

def grad_report(holder, reg_loss, input0, input1, input2, n_sample=8):
    """把数据项和正则项的梯度【分开】算，比较范数。

    为什么不能只看 loss 数值：Adam 按二阶矩归一化，单步位移约等于 lr，
    与梯度大小基本无关。所以决定"往哪走"的是两项梯度的【相对大小】，
    而不是两项 loss 的相对大小。两者可以差很多个数量级。

    怎么读：
      ratio = ||grad_reg|| / ||grad_data||
        << 1   正则只是微调，数据项主导        -> 健康
        ~ 1    两者拉锯                        -> 临界
        >> 1   正则主导，数据项说了不算        -> 已经跑偏
      nan/inf 计数 > 0 -> 真的爆炸了（Adam 下通常表现为 nan 而不是巨大步长）
    """
    model = holder.OBJ
    y0_all = np.sqrt(np.maximum(input2, 0)).astype(np.float32)
    rows = []
    idx = np.linspace(0, len(input1) - 1, min(n_sample, len(input1))).astype(int)
    for i in idx:
        X = [input0[i:i+1], input1[i:i+1], input2[i:i+1]]
        y0 = y0_all[i:i+1]
        with tf.GradientTape(persistent=True) as tape:
            o0, o1, o2 = model(X, training=False)
            L_data = tf.reduce_mean(tf.square(o0 - y0))
            L_reg = reg_loss(None, o1)
        w = model.trainable_weights
        g_data = tape.gradient(L_data, w)
        g_reg = tape.gradient(L_reg, w)
        del tape

        def stat(gs):
            gs = [g for g in gs if g is not None]
            if not gs:
                return 0.0, 0.0, 0, 0
            flat = tf.concat([tf.reshape(g, [-1]) for g in gs], 0)
            nan = int(tf.reduce_sum(tf.cast(tf.math.is_nan(flat), tf.int32)))
            inf = int(tf.reduce_sum(tf.cast(tf.math.is_inf(flat), tf.int32)))
            return (float(tf.linalg.global_norm(gs)),
                    float(tf.reduce_max(tf.abs(flat))), nan, inf)

        nd, md, nan_d, inf_d = stat(g_data)
        nr, mr, nan_r, inf_r = stat(g_reg)
        rows.append((i, float(L_data), float(L_reg), nd, nr,
                     nr/(nd+1e-30), md, mr, nan_d+nan_r, inf_d+inf_r))

    print(f'{"pos":>5s}{"L_data":>11s}{"L_reg":>11s}{"|g_data|":>11s}'
          f'{"|g_reg|":>11s}{"ratio":>10s}{"max|g_d|":>11s}{"max|g_r|":>11s}'
          f'{"nan":>5s}{"inf":>5s}')
    for r in rows:
        print(f'{r[0]:5d}{r[1]:11.3e}{r[2]:11.3e}{r[3]:11.3e}'
              f'{r[4]:11.3e}{r[5]:10.3e}{r[6]:11.3e}{r[7]:11.3e}'
              f'{r[8]:5d}{r[9]:5d}')
    ratios = [r[5] for r in rows]
    print(f'\n梯度范数比 reg/data:  中位数 {np.median(ratios):.3e}  '
          f'最大 {np.max(ratios):.3e}')
    bad = sum(r[8] + r[9] for r in rows)
    print(f'nan/inf 总计: {bad}   -> {"有爆炸" if bad else "无爆炸"}')
    return rows


def adam_state_report(holder):
    """Adam 内部状态。

    Adam 特有的失效方式：某一步出现巨大梯度，会污染二阶矩 v，
    此后 step = lr * m/(sqrt(v)+eps) 被压得极小，优化器实质"冻结"。
    表现是指标突然走平 —— 容易被误认为"收敛了"。

    怎么读：v 的最大值比中位数大很多个数量级 -> 曾经吃过一次巨大梯度。
    """
    for tag, model in [('OBJ', holder.OBJ), ('PRB', holder.PRB)]:
        opt = getattr(model, 'optimizer', None)
        if opt is None:
            continue
        vs = [w for w in opt.variables() if 'v' in w.name or 'velocity' in w.name]
        if not vs:
            print(f'{tag}: 取不到 Adam 二阶矩（可能是新版 optimizer 命名不同）')
            continue
        flat = np.concatenate([w.numpy().ravel() for w in vs])
        flat = flat[flat > 0]
        if flat.size == 0:
            continue
        print(f'{tag} Adam v:  中位数 {np.median(flat):.3e}  '
              f'最大 {flat.max():.3e}  比值 {flat.max()/np.median(flat):.3e}')


# =============================================================================
# 问题 2：K 是不是被"方差趋零"顶上去的
# =============================================================================

def kurtosis_anatomy(u, eps=1e-20, topk=10):
    """把 K = m4/m2^2 拆开看，判断它大是【真重尾】还是【数值假象】。

    先澄清一点：K 是【尺度不变】的 —— g -> c*g 不改变 K。
    所以"方差整体变小"本身【不会】把 K 顶上去。
    K 变大只有一个来源：分布形状变重尾（大多数样本接近 0 + 少数离群）。
    这时 m2 确实很小，但那是【结果】不是【原因】。

    真正的数值风险只有一个：m2^2 掉到 eps 附近时，分母被 eps 接管，K 被
    人为压小（不是放大）。所以要看 m2^2 / eps 这个比值。

    怎么读：
      m2^2 / eps >> 1        -> eps 不参与，K 是真实值
      top1 占 m4 的比例 > 50% -> K 完全由【一个像素】决定，退化
      有效样本数 n_eff        -> 参与贡献 m4 的等效样本数，越小越退化
    """
    u = np.asarray(u, np.float64)
    g = np.concatenate([np.diff(u, axis=0).ravel(), np.diff(u, axis=1).ravel()])
    n = g.size
    d = g - g.mean()
    m2 = float(np.mean(d**2))
    m4 = float(np.mean(d**4))
    K = m4/(m2**2) if m2 > 0 else 0.0

    c = d**4
    order = np.argsort(c)[::-1]
    tot = c.sum()
    top1 = c[order[0]]/tot if tot > 0 else 0.0
    topK = c[order[:topk]].sum()/tot if tot > 0 else 0.0
    n_eff = (tot**2)/np.sum(c**2) if np.sum(c**2) > 0 else 0.0

    print(f'  n (梯度样本数)      = {n}')
    print(f'  m2                  = {m2:.6e}')
    print(f'  m4                  = {m4:.6e}')
    print(f'  K = m4/m2^2         = {K:.4f}      (理论上界 n = {n})')
    print(f'  K / n               = {K/n:.4f}      (越接近 1 越退化)')
    print(f'  m2^2 / eps          = {m2**2/eps:.3e}  '
          f'-> {"eps 不参与，K 可信" if m2**2/eps > 1e3 else "eps 已介入，K 被低估"}')
    print(f'  最大单样本占 m4      = {100*top1:.2f}%')
    print(f'  最大 {topk} 个占 m4    = {100*topK:.2f}%')
    print(f'  有效样本数 n_eff     = {n_eff:.1f}  (总数 {n})')
    return dict(n=n, m2=m2, m4=m4, K=K, top1=top1, topk=topK, n_eff=n_eff)


# =============================================================================
# 汇总
# =============================================================================

def full_report(holder, reg_loss, input0, input1, input2,
                beta, reg_crop, eval_crop, obj_gt):
    from importlib import import_module
    ic = import_module('functions.innm_common')

    r = np.asarray(holder.OBJ.get_layer('Or').get_weights())[0, :, :, 0]
    i = np.asarray(holder.OBJ.get_layer('Oi').get_weights())[0, :, :, 0]
    O = r + 1j*i
    O_reg = O if reg_crop <= 0 else O[reg_crop:-reg_crop, reg_crop:-reg_crop]

    print('=' * 62)
    print('【1】权重本身有没有 nan/inf')
    print('=' * 62)
    for nm in ['Or', 'Oi', 'Pr', 'Pi']:
        w = np.asarray(holder.OBJ.get_layer(nm).get_weights())
        print(f'  {nm}: nan {int(np.isnan(w).sum())}  inf {int(np.isinf(w).sum())}  '
              f'max|w| {np.abs(w).max():.3e}')

    print()
    print('=' * 62)
    print('【2】梯度爆炸 / 谁主导更新方向')
    print('=' * 62)
    grad_report(holder, reg_loss, input0, input1, input2)
    print()
    adam_state_report(holder)

    print()
    print('=' * 62)
    print('【3】K 的解剖 —— 重建结果')
    print('=' * 62)
    a = kurtosis_anatomy(np.abs(O_reg))

    print()
    print('【3b】K 的解剖 —— 真值（对照）')
    b = kurtosis_anatomy(np.abs(obj_gt))

    print()
    print('=' * 62)
    print('【4】结论速查')
    print('=' * 62)
    print(f'  K 重建 {a["K"]:.1f}  vs  真值 {b["K"]:.1f}   '
          f'-> 偏离 {a["K"]/max(b["K"],1e-9):.0f} 倍')
    print(f'  重建的 m4 有 {100*a["top1"]:.1f}% 来自单个样本，'
          f'真值只有 {100*b["top1"]:.1f}%')
    print(f'  有效样本数 {a["n_eff"]:.1f} vs 真值 {b["n_eff"]:.1f}')
    print(f'  beta*K = {beta*a["K"]:.3e}   （拿它和 data MSE 比）')
    print()
    print('  判据：')
    print('    - 权重无 nan/inf 且梯度范数有限   -> 不是数值爆炸')
    print('    - m2^2/eps 很大                   -> 不是 eps 造成的假象')
    print('    - top1 占比高、n_eff 很小          -> K 大是真的重尾，优化器在造尖峰')
    print('    这三条同时成立 = 无界正则被正常地"薅"了，属于目标函数缺陷，不是 bug')
