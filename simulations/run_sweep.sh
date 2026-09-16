#!/usr/bin/env bash
# ProPtyNet 剂量轴 / 重叠轴扫描
#   bash run_sweep.sh smoke     单点验证（先跑这个）
#   bash run_sweep.sh dose      剂量轴
#   bash run_sweep.sh overlap   重叠轴
#   bash run_sweep.sh all       两条轴都跑
# AD 调度: AD_OPT=plain 用标准 AD ptychography（默认 alternating = INNM/Keras 移植）
#   例: AD_OPT=plain AD_ITERS=2000 SUP=0.995 SEEDS=1 PY=python3 bash run_sweep.sh overlap
#   plain 的学习率用 AD_LR 覆盖(默认 1e-2); alternating 不受影响, 仍用 Cfg 默认 3e-2
# 换样品: IMG_AMP=USAF.jpg ROOT=runs_usaf ... bash run_sweep.sh overlap
#   两种调度一次跑完: AD_OPT="alternating plain" ... bash run_sweep.sh overlap
#   每个配置的执行顺序: net -> ad(alternating) -> adplain，三者共用同一份仿真数据
# 扫描图案: JITTER=0.1 SCAN_PATTERN=fermat ... bash run_sweep.sh overlap
#   JITTER 默认 0 = 纯周期光栅，会触发 raster grid pathology（物体↔探针的周期性简并）。
#   非零 jitter 或 fermat 都会改变实测重叠率 -> 与 JITTER=0 的历史结果不可比，换 ROOT。
#
# 【随机种子：三条独立的轴，互不锁定】
#   SEEDS       -> --seed        网络初值 (torch.manual_seed)。只影响 net。
#   SCAN_SEEDS  -> --scan-seed   扫描位置抖动。JITTER>0 时才有效；影响 net 和 AD。
#   NOISE_SEEDS -> --noise-seed  泊松噪声实现。只在 dose 轴；影响 net 和 AD。
#   overlap 轴跑 SCAN_SEEDS × SEEDS 的全交叉；dose 轴跑 NOISE_SEEDS × SEEDS，
#   扫描实现固定在 SCAN_SEEDS 的第一个值（否则四重循环组合爆炸）。
#   默认 SCAN_SEEDS="0" / NOISE_SEEDS="0" -> 只扫网络初值这一条轴。
#
#   为什么必须解耦: 早先的版本让一个索引同时驱动 --seed 和 --scan-seed。JITTER=0 时
#   无害（--scan-seed 空转），但 JITTER>0 之后两个随机源被锁死，任何一次失败都分不清
#   是网络初值还是扫描实现造成的。要做归因就用 SEEDS=1 SCAN_SEEDS=0 这类不对角的组合。
#
#   AD 没有任何随机初始化（物体初值全 1、探针初值 P0 都是确定性的），run_ad 里那句
#   torch.manual_seed 是空转 -> --seed 对 ad/adplain 完全无效，所以 AD 不进 SEEDS 循环。
#   但 AD 会随 --scan-seed(JITTER>0 时) 和 --noise-seed 变，所以那两条轴上每格都跑。
#   （旧的 AD_REPEAT 开关已删除：它假设"AD 各 seed 逐比特相同"，那在 JITTER>0 时不成立。）
# 已经有 *_result.npz 的目录会被跳过，所以中断了直接重跑同一条命令即可续上。

set -u
PY=${PY:-python}
SCRIPT=${SCRIPT:-ptychography_AD_DIP/simulations/ProPtyNet_torch.py}
ROOT=${ROOT:-runs}
ITERS=${ITERS:-2000}
STAGES=${STAGES:-24}
# 三条种子轴，见文件头。全部是空格分隔的列表 —— SEEDS=3 是"一个值 3"，不是"3 个"。
SEEDS=${SEEDS:-"0 1 2"}           # --seed        网络初值（只影响 net）
SCAN_SEEDS=${SCAN_SEEDS:-"0"}     # --scan-seed   扫描抖动实现（JITTER>0 时才有效）
NOISE_SEEDS=${NOISE_SEEDS:-"0"}   # --noise-seed  泊松实现（只在 dose 轴）
SCAN_S0=${SCAN_SEEDS%% *}         # dose 轴把扫描实现固定在这个值
SUP=${SUP:-0.9999}          # 支撑域能量阈值，全程固定，否则各档探针自由度不同
DOSES=${DOSES:-"1e2 3e2 1e3 3e3 1e4"}
# 重叠轴 "点数:步长"。线性/面积重叠(3 seed 均值)与评估区零照明像素占比:
#   9:33.4 -> 35.2%/23.8%  极值39.6  零照明0.1%   <- 下限，受画布 SCAN_LIMIT=48 限制
#  16:26.7 -> 53.8%/43.3%  极值45.4  零照明0
#  25:20   -> 62.9%/53.9%  极值44.0  零照明0
#  36:16   -> 71.3%/63.9%  极值43.2  零照明0
#  49:13.3 -> 76.4%/70.2%  极值42.5  零照明0
# 去掉了原来的 64:11.4 (80.2%) —— 高重叠端信息冗余，且 64 点最耗时。
# 【注意】这条轴上 n 与 step 同向变化(视场恒定~80px)，所以【重叠率与数据量完全混淆】:
#   9 点 vs 49 点是 5.4 倍数据量差。它是"扫描密度轴"，不是纯重叠轴。写论文别叫错。
OVERLAPS=${OVERLAPS:-"9:33.4 16:26.7 25:20 36:16 49:13.3"}
RUN_AD=${RUN_AD:-1}         # RUN_AD=0 则只跑 net
# AD 的优化调度: alternating(=INNM/Keras 移植, 默认) | joint | plain(标准 AD ptychography)
#   plain 用 --ad-iters 计步(全批量), 与 stages 不是同一个刻度, 所以单独放 adplain_* 目录
# 可以给多个, 空格分隔, 一次调用全跑: AD_OPT="alternating plain"
AD_OPT=${AD_OPT:-alternating}
AD_ITERS=${AD_ITERS:-2000}
# plain 的学习率。500 步扫描: 3e-3/1e-2 欠训练, 3e-2 最快但 loss 已接近地板,
# 1e-1 的 loss 反而更高(越过稳定边界)。选 1e-2 作为稳定档。
# 【只作用于 plain】不改 Cfg 的 lr_obj/lr_prb 默认值(3e-2) —— 改了会连带改掉
# alternating 基线, 让已有的 ad_* 结果全部失去可比性。
AD_LR=${AD_LR:-1e-2}

# 样品图。换样品会让所有历史结果失去可比性 -> 同时改 ROOT 换一棵目录树。
IMG_AMP=${IMG_AMP:-}        # 空 = 用 Cfg 默认 cameraman.bmp
IMG_PHASE=${IMG_PHASE:-}    # 空 = 用 Cfg 默认 westconcordorthophoto.bmp
# 物体相位幅度(rad)。空 = 用 Cfg 默认 0.8。改这个等于换样品 -> 同时改 ROOT。
PHASE=${PHASE:-}
# 扫描图案。改这两个等于换了扫描几何 -> 同时改 ROOT。
JITTER=${JITTER:-0}         # --scan-jitter，单位是 step 的比例。0 = 纯周期光栅
SCAN_PATTERN=${SCAN_PATTERN:-}   # 空 = Cfg 默认 raster_jitter；可填 fermat
# 物体输出头初始化。空 = Cfg 默认（初始相位是 U(-pi,pi) 白噪声，实测 RMS 1.80 rad）。
# 给一个数 = 用相位中性初始化，该数是物体头权重的缩放系数：
#   INIT_ALPHA=0    严格中性（第 0 步 O ≡ 1·exp(i0)，与 AD 的初值完全相同）
#   INIT_ALPHA=0.3  部分随机；INIT_ALPHA=1 只把 bias 设成中性点、权重保持默认
# 这等于换了初始点 -> 同时改 ROOT，否则会和已有结果混在一棵树里。
INIT_ALPHA=${INIT_ALPHA:-}
# 探针支撑（模型口径；评估口径恒为二值，不受影响）。改这个 -> 同时改 ROOT。
#   hard(默认,历史) | soft | none
# 【注意】hard 时 simulate() 用完整探针、模型用截断探针，SUP=0.995 下有 6.11%
# 的幅度失配，低重叠时会被补偿进物体 -> 环纹。论文主结果建议 none。
PROBE_SUPPORT=${PROBE_SUPPORT:-hard}
SUPPORT_SOFT=${SUPPORT_SOFT:-6}     # 仅 soft 档有效，过渡带宽度(px)
# 仿真数据用哪个探针: full(默认,物理正确) | model(与模型一致, inverse crime, 只当诊断)
SIM_PROBE=${SIM_PROBE:-full}
COMMON="--support-energy $SUP --lr-cosine --scan-jitter $JITTER"
[ -n "$SCAN_PATTERN" ] && COMMON="$COMMON --scan-pattern $SCAN_PATTERN"
[ -n "$INIT_ALPHA" ]   && COMMON="$COMMON --obj-init neutral --obj-init-alpha $INIT_ALPHA"
COMMON="$COMMON --probe-support $PROBE_SUPPORT --sim-probe $SIM_PROBE"
[ "$PROBE_SUPPORT" = "soft" ] && COMMON="$COMMON --support-soft $SUPPORT_SOFT"
[ -n "$IMG_AMP" ]   && COMMON="$COMMON --obj-amp-img $IMG_AMP"
[ -n "$IMG_PHASE" ] && COMMON="$COMMON --obj-phase-img $IMG_PHASE"
[ -n "$PHASE" ]     && COMMON="$COMMON --obj-phase-span $PHASE"
mkdir -p "$ROOT"

echo "=== run_sweep $(date '+%F %T') ==="
echo "  ROOT=$ROOT  ITERS=$ITERS  SUP=$SUP  AD_OPT='$AD_OPT'  AD_ITERS=$AD_ITERS"
echo "  JITTER=$JITTER  SCAN_PATTERN='${SCAN_PATTERN:-raster_jitter(默认)}'  PHASE='${PHASE:-0.8(默认)}'"
echo "  SEEDS='$SEEDS'  SCAN_SEEDS='$SCAN_SEEDS'  NOISE_SEEDS='$NOISE_SEEDS'"
if [ -n "$INIT_ALPHA" ]; then echo "  物体初值: 相位中性 alpha=$INIT_ALPHA"
else echo "  物体初值: 默认（初始相位是 U(-pi,pi) 白噪声）"; fi
echo "  探针支撑: 模型=$PROBE_SUPPORT$([ "$PROBE_SUPPORT" = soft ] && echo "(过渡带 ${SUPPORT_SOFT}px)")  仿真探针=$SIM_PROBE"
[ "$SIM_PROBE" = "model" ] && echo "  ⚠ SIM_PROBE=model 是 inverse crime，只当诊断用"
[ "$PROBE_SUPPORT" = "hard" ] && [ "$SIM_PROBE" = "full" ] && \
  echo "  ⚠ hard + full: 数据用完整探针、模型用截断探针，SUP=$SUP 下存在幅度失配"
if [ "$JITTER" = "0" ] || [ "$JITTER" = "0.0" ]; then
  echo "  ⚠ JITTER=0：纯周期光栅，存在 raster grid pathology；--scan-seed 此时完全空转。"
  case "$SCAN_SEEDS" in *\ *) echo "  ⚠ JITTER=0 却给了多个 SCAN_SEEDS —— 这些跑会逐比特相同，纯浪费。";; esac
fi

run () {           # run <outdir> <mode> <额外参数...>
  local out="$ROOT/$1"; shift
  local mode="$1"; shift
  if [ -f "$out/${mode}_result.npz" ]; then
    echo "  跳过（已存在） $out"; return
  fi
  mkdir -p "$out"
  echo "  >>> $out   [$(date +%H:%M:%S)]"
  $PY -u "$SCRIPT" "$mode" $COMMON "$@" --outdir "$out" > "$out/log.txt" 2>&1
  if [ $? -ne 0 ]; then
    echo "  !!! 失败，看 $out/log.txt"; tail -5 "$out/log.txt"
  else
    grep -E "^\[scan\]|SSIM 最优步|真值 SSIM|用时" "$out/log.txt" | sed 's/^/      /'
    tail -2 "$out/log.txt" | grep "^  it" | sed 's/^/      /'
  fi
}

run_ad_all () {   # run_ad_all <子目录> <名字后缀> <额外参数...>
  # 对 AD_OPT 里列出的每一种调度各跑一次。目录名: alternating -> ad_*(向后兼容),
  # plain -> adplain_*, 其它 -> ad<名字>_*，互不覆盖，可以并排汇总。
  [ "$RUN_AD" = "1" ] || return 0
  local sub="$1" sfx="$2"; shift 2
  local o tag args
  for o in $AD_OPT; do
    case "$o" in
      plain)       tag=adplain; args="--opt-mode plain --ad-iters $AD_ITERS --lr-obj $AD_LR --lr-prb $AD_LR" ;;
      alternating) tag=ad;      args="--opt-mode alternating --stages $STAGES" ;;
      *)           tag="ad$o";  args="--opt-mode $o --stages $STAGES" ;;
    esac
    run "$sub/$tag$sfx" ad $args "$@"
  done
}

do_smoke () {
  echo "=== 单点验证：干净数据 + lr-cosine，跟你已有的 SSIM 0.954 对比 ==="
  run smoke_clean_cosine net --iters $ITERS
  echo "=== 单点验证：低剂量 ==="
  run smoke_dose1e3 net --iters $ITERS --poisson --peak-photons 1e3 --noise-seed 1
}

do_dose () {
  echo "=== 剂量轴 ==="
  # 扫描实现固定在 SCAN_S0，只让 NOISE_SEEDS × SEEDS 变。
  local SCAN="--scan-seed $SCAN_S0"
  for P in $DOSES; do for W in $NOISE_SEEDS; do
    # AD 随 --noise-seed 变、与 --seed 无关 -> 每个 (剂量, 噪声实现) 跑一次
    run_ad_all dose "_p${P}_w${W}" --poisson --peak-photons $P --noise-seed $W $SCAN
    for S in $SEEDS; do
      run "dose/net_p${P}_w${W}_s${S}" net --iters $ITERS \
          --poisson --peak-photons $P --noise-seed $W $SCAN --seed $S
    done
  done; done
  echo "--- 无噪声参照 ---"
  # 无噪声且扫描固定 -> AD 完全确定性，跑一次即可。
  run_ad_all dose "_clean" $SCAN
  for S in $SEEDS; do
    run "dose/net_clean_s${S}" net --iters $ITERS $SCAN --seed $S
  done
}

do_overlap () {
  echo "=== 重叠轴 ==="
  local NP ST C S SCAN
  for OV in $OVERLAPS; do
    NP=${OV%%:*}; ST=${OV##*:}
    for C in $SCAN_SEEDS; do
      SCAN="--scan-npos $NP --scan-step $ST --scan-seed $C"
      # AD 无随机初始化，只随扫描实现变 -> 每个 (重叠档, 扫描实现) 跑一次，不进 SEEDS 循环
      run_ad_all overlap "_n${NP}_c${C}" $SCAN
      for S in $SEEDS; do
        run "overlap/net_n${NP}_c${C}_s${S}" net --iters $ITERS $SCAN --seed $S
      done
    done
  done
}

case "${1:-smoke}" in
  smoke)   do_smoke ;;
  dose)    do_dose ;;
  overlap) do_overlap ;;
  all)     do_dose; do_overlap ;;
  *) echo "用法: bash run_sweep.sh {smoke|dose|overlap|all}"; exit 1 ;;
esac
echo "=== 完成 $(date) ==="
