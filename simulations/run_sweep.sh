#!/usr/bin/env bash
# ProPtyNet 扫描：DIP 先验 vs 纯 AD
#   bash run_sweep.sh smoke      单点验证（先跑这个）
#   bash run_sweep.sh overlap    重叠/扫描密度轴
#   bash run_sweep.sh dose       剂量（光子数）轴
#   bash run_sweep.sh all        两条轴都跑
#
# 例:
#   SCRIPT=ProPtyNet_torch.py ROOT=runs_base SEEDS="0 1 2" bash run_sweep.sh overlap
#   ROOT=runs_ph0 PHASE=0 AMP_MIN=0.15 bash run_sweep.sh overlap
#   ROOT=runs_upper PROBE_MODE=truth RUN_AD=0 bash run_sweep.sh overlap
#
# 【随机性】只有 net 有随机初始化(--seed)。AD 完全确定性 —— 物体初值恒为 1、
#   探针初值是确定性的圆盘、扫描位置是确定性的栅格，所以 AD 不进 SEEDS 循环，
#   每格只跑一次。AD 仍随 --noise-seed 变，所以剂量轴上每个噪声实现都跑。
#
# 已有 *_result.npz 的目录会被跳过 -> 中断了直接重跑同一条命令即可续上。

set -u
PY=${PY:-python}
SCRIPT=${SCRIPT:-ProPtyNet_torch.py}
ROOT=${ROOT:-runs}

ITERS=${ITERS:-2000}          # net 步数
AD_ITERS=${AD_ITERS:-2000}    # AD 步数（同刻度：两者都是全批量梯度步）
SEEDS=${SEEDS:-"0 1 2"}       # --seed，只影响 net。空格分隔；SEEDS=3 是"一个值 3"
NOISE_SEEDS=${NOISE_SEEDS:-"0"}
RUN_AD=${RUN_AD:-1}
RUN_NET=${RUN_NET:-1}

# 重叠轴 "点数:步长"。视场恒定 ~80px，所以 n 与 step 同向变化 ->
# 【重叠率与数据量完全混淆】: 9 点 vs 49 点是 5.4 倍数据量差。
# 它是"扫描密度轴"，不是纯重叠轴。写论文别叫错。
OVERLAPS=${OVERLAPS:-"9:33.4 16:26.7 25:20 36:16 49:13.3"}
DOSES=${DOSES:-"1e2 3e2 1e3 3e3 1e4"}

# ---- 换了下面任何一项 = 换了实验，必须同时换 ROOT，否则会和已有结果混在一棵树里 ----
PHASE=${PHASE:-}            # --obj-phase-span，空 = Cfg 默认 0.8；0 = 纯振幅物体
AMP_MIN=${AMP_MIN:-}        # --obj-amp-min，空 = 0.4
IMG_AMP=${IMG_AMP:-}
IMG_PHASE=${IMG_PHASE:-}
PROBE_MODE=${PROBE_MODE:-pixel}   # pixel | truth(冻结真值 = 上界对照)
SIGMA=${SIGMA:-}            # --probe-init-sigma，空 = 0.15
ALPHA=${ALPHA:-}            # --obj-init-alpha，空 = 0（严格相位中性，与 AD 同初值）
HOLDOUT=${HOLDOUT:-}        # --holdout-frac，空 = 0（不留出）
BASE_CH=${BASE_CH:-}
LR_NET=${LR_NET:-}
LR_PROBE=${LR_PROBE:-}
LR_OBJ=${LR_OBJ:-}
LR_PRB=${LR_PRB:-}

COMMON="--lr-cosine"
[ -n "$PHASE" ]     && COMMON="$COMMON --obj-phase-span $PHASE"
[ -n "$AMP_MIN" ]   && COMMON="$COMMON --obj-amp-min $AMP_MIN"
[ -n "$IMG_AMP" ]   && COMMON="$COMMON --obj-amp-img $IMG_AMP"
[ -n "$IMG_PHASE" ] && COMMON="$COMMON --obj-phase-img $IMG_PHASE"
[ -n "$SIGMA" ]     && COMMON="$COMMON --probe-init-sigma $SIGMA"
[ -n "$ALPHA" ]     && COMMON="$COMMON --obj-init-alpha $ALPHA"
[ -n "$HOLDOUT" ]   && COMMON="$COMMON --holdout-frac $HOLDOUT"
NET_ONLY="--probe-mode $PROBE_MODE"
[ -n "$BASE_CH" ]   && NET_ONLY="$NET_ONLY --base-ch $BASE_CH"
[ -n "$LR_NET" ]    && NET_ONLY="$NET_ONLY --lr-net $LR_NET"
[ -n "$LR_PROBE" ]  && NET_ONLY="$NET_ONLY --lr-probe $LR_PROBE"
AD_ONLY=""
[ -n "$LR_OBJ" ]    && AD_ONLY="$AD_ONLY --lr-obj $LR_OBJ"
[ -n "$LR_PRB" ]    && AD_ONLY="$AD_ONLY --lr-prb $LR_PRB"

mkdir -p "$ROOT"
echo "=== run_sweep $(date '+%F %T') ==="
echo "  ROOT=$ROOT  SCRIPT=$SCRIPT"
echo "  ITERS=$ITERS  AD_ITERS=$AD_ITERS  SEEDS='$SEEDS'  NOISE_SEEDS='$NOISE_SEEDS'"
echo "  扫描=raster  物体相位跨度=${PHASE:-0.8(默认)}  振幅下限=${AMP_MIN:-0.4(默认)}"
echo "  探针=$PROBE_MODE  初值=平滑圆盘+零相位 sigma=${SIGMA:-0.15(默认)}R"
echo "  物体初值 alpha=${ALPHA:-0(默认,与 AD 同初值)}   留出=${HOLDOUT:-0(不留出)}"
[ "$PROBE_MODE" = "truth" ] && \
  echo "  ⚠ PROBE_MODE=truth：探针冻结在真值上，这是【上界对照】，不是探针重建"

run () {           # run <子目录> <mode> <额外参数...>
  local out="$ROOT/$1"; shift
  local mode="$1"; shift
  if [ -f "$out/${mode}_result.npz" ]; then echo "  跳过（已存在） $out"; return; fi
  mkdir -p "$out"
  echo "  >>> $out   [$(date +%H:%M:%S)]"
  $PY -u "$SCRIPT" "$mode" $COMMON "$@" --outdir "$out" > "$out/log.txt" 2>&1
  if [ $? -ne 0 ]; then
    echo "  !!! 失败，看 $out/log.txt"; tail -5 "$out/log.txt"
  else
    grep -E "^\[scan\]|err_P0|最优步|用时" "$out/log.txt" | sed 's/^/      /'
    tail -3 "$out/log.txt" | grep "^  it" | sed 's/^/      /'
  fi
}

do_smoke () {
  echo "=== 单点验证 ==="
  $PY -u "$SCRIPT" check $COMMON
  [ "$RUN_AD"  = "1" ] && run smoke ad  $AD_ONLY --ad-iters $AD_ITERS
  [ "$RUN_NET" = "1" ] && run smoke net $NET_ONLY --iters $ITERS --seed 0
  return 0
}

do_overlap () {
  echo "=== 重叠 / 扫描密度轴 ==="
  local NP ST S G
  for OV in $OVERLAPS; do
    NP=${OV%%:*}; ST=${OV##*:}
    G="--scan-npos $NP --scan-step $ST"
    # AD 完全确定性 -> 每档只跑一次
    [ "$RUN_AD" = "1" ] && run "overlap/ad_n${NP}" ad $AD_ONLY --ad-iters $AD_ITERS $G
    if [ "$RUN_NET" = "1" ]; then
      for S in $SEEDS; do
        run "overlap/net_n${NP}_s${S}" net $NET_ONLY --iters $ITERS $G --seed $S
      done
    fi
  done
}

do_dose () {
  echo "=== 剂量轴 ==="
  local P W S
  for P in $DOSES; do for W in $NOISE_SEEDS; do
    [ "$RUN_AD" = "1" ] && run "dose/ad_p${P}_w${W}" ad $AD_ONLY \
        --ad-iters $AD_ITERS --poisson --peak-photons $P --noise-seed $W
    if [ "$RUN_NET" = "1" ]; then
      for S in $SEEDS; do
        run "dose/net_p${P}_w${W}_s${S}" net $NET_ONLY --iters $ITERS \
            --poisson --peak-photons $P --noise-seed $W --seed $S
      done
    fi
  done; done
  echo "--- 无噪声参照 ---"
  [ "$RUN_AD" = "1" ] && run "dose/ad_clean" ad $AD_ONLY --ad-iters $AD_ITERS
  if [ "$RUN_NET" = "1" ]; then
    for S in $SEEDS; do
      run "dose/net_clean_s${S}" net $NET_ONLY --iters $ITERS --seed $S
    done
  fi
}

case "${1:-smoke}" in
  smoke)   do_smoke ;;
  overlap) do_overlap ;;
  dose)    do_dose ;;
  all)     do_overlap; do_dose ;;
  *) echo "用法: bash run_sweep.sh {smoke|overlap|dose|all}"; exit 1 ;;
esac
echo "=== 完成 $(date) ==="
