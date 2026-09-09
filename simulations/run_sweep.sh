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
# 已经有 *_result.npz 的目录会被跳过，所以中断了直接重跑同一条命令即可续上。

set -u
PY=${PY:-python}
SCRIPT=${SCRIPT:-ptychography_AD_DIP/simulations/ProPtyNet_torch.py}
ROOT=${ROOT:-runs}
ITERS=${ITERS:-2000}
STAGES=${STAGES:-24}
SEEDS=${SEEDS:-"1 2 3"}
SUP=${SUP:-0.9999}          # 支撑域能量阈值，全程固定，否则各档探针自由度不同
DOSES=${DOSES:-"1e2 3e2 1e3 3e3 1e4"}
OVERLAPS=${OVERLAPS:-"16:26.7 25:20 36:16 49:13.3 64:11.4"}
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
COMMON="--support-energy $SUP --lr-cosine"
[ -n "$IMG_AMP" ]   && COMMON="$COMMON --obj-amp-img $IMG_AMP"
[ -n "$IMG_PHASE" ] && COMMON="$COMMON --obj-phase-img $IMG_PHASE"
mkdir -p "$ROOT"

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
  for P in $DOSES; do for S in $SEEDS; do
    run "dose/net_p${P}_s${S}" net --iters $ITERS --poisson --peak-photons $P --noise-seed $S
    run_ad_all dose "_p${P}_s${S}" --poisson --peak-photons $P --noise-seed $S
  done; done
  echo "--- 无噪声参照 ---"
  run "dose/net_clean" net --iters $ITERS
  run_ad_all dose "_clean" 
}

do_overlap () {
  echo "=== 重叠轴 ==="
  for C in $OVERLAPS; do
    NP=${C%%:*}; ST=${C##*:}
    for S in $SEEDS; do
      run "overlap/net_n${NP}_s${S}" net --iters $ITERS --scan-npos $NP --scan-step $ST --scan-seed $S
      run_ad_all overlap "_n${NP}_s${S}" --scan-npos $NP --scan-step $ST --scan-seed $S
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
