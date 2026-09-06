#!/usr/bin/env bash
# ProPtyNet 剂量轴 / 重叠轴扫描
#   bash run_sweep.sh smoke     单点验证（先跑这个）
#   bash run_sweep.sh dose      剂量轴
#   bash run_sweep.sh overlap   重叠轴
#   bash run_sweep.sh all       两条轴都跑
# 已经有 *_result.npz 的目录会被跳过，所以中断了直接重跑同一条命令即可续上。

set -u
PY=${PY:-python}
SCRIPT=${SCRIPT:-ProPtyNet_torch.py}
ROOT=${ROOT:-runs}
ITERS=${ITERS:-2000}
STAGES=${STAGES:-24}
SEEDS=${SEEDS:-"1 2 3"}
SUP=${SUP:-0.9999}          # 支撑域能量阈值，全程固定，否则各档探针自由度不同
DOSES=${DOSES:-"1e2 3e2 1e3 3e3 1e4"}
OVERLAPS=${OVERLAPS:-"16:26.7 25:20 36:16 49:13.3 64:11.4"}
RUN_AD=${RUN_AD:-1}         # RUN_AD=0 则只跑 net

COMMON="--support-energy $SUP --lr-cosine"
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
    [ "$RUN_AD" = "1" ] && run "dose/ad_p${P}_s${S}" ad --stages $STAGES --poisson --peak-photons $P --noise-seed $S
  done; done
  echo "--- 无噪声参照 ---"
  run "dose/net_clean" net --iters $ITERS
  [ "$RUN_AD" = "1" ] && run "dose/ad_clean" ad --stages $STAGES
}

do_overlap () {
  echo "=== 重叠轴 ==="
  for C in $OVERLAPS; do
    NP=${C%%:*}; ST=${C##*:}
    for S in $SEEDS; do
      run "overlap/net_n${NP}_s${S}" net --iters $ITERS --scan-npos $NP --scan-step $ST --scan-seed $S
      [ "$RUN_AD" = "1" ] && run "overlap/ad_n${NP}_s${S}" ad --stages $STAGES --scan-npos $NP --scan-step $ST --scan-seed $S
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
