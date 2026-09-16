#!/usr/bin/env bash
# 把旧命名的 overlap 结果目录改成新命名，避免白重跑。
#   bash migrate_dirnames.sh runs0.5            # 只打印要做什么（dry run）
#   bash migrate_dirnames.sh runs0.5 --apply    # 真的改
#
# 旧脚本用一个索引同时驱动 --seed 和 --scan-seed，所以旧目录里 seed == scan_seed == S：
#   net_n<NP>_s<S>    -> net_n<NP>_c<S>_s<S>
#   <tag>_n<NP>_s<S>  -> <tag>_n<NP>_c<S>        (ad / adplain / ad<opt>)
# 已经是新命名的目录会被跳过。dose/ 不处理（旧 dose 结果如果有，手工改）。
set -u
ROOT=${1:?用法: bash migrate_dirnames.sh <ROOT> [--apply]}
APPLY=${2:-}
[ -d "$ROOT/overlap" ] || { echo "没有 $ROOT/overlap"; exit 1; }
n=0
for d in "$ROOT"/overlap/*/; do
  b=$(basename "$d")
  case "$b" in
    *_c*_s*|*_c*) continue ;;                      # 已是新命名
  esac
  if [[ "$b" =~ ^net_n([0-9]+)_s([0-9]+)$ ]]; then
    new="net_n${BASH_REMATCH[1]}_c${BASH_REMATCH[2]}_s${BASH_REMATCH[2]}"
  elif [[ "$b" =~ ^(ad[a-z]*)_n([0-9]+)_s([0-9]+)$ ]]; then
    new="${BASH_REMATCH[1]}_n${BASH_REMATCH[2]}_c${BASH_REMATCH[3]}"
  else
    echo "  ? 不认识，跳过: $b"; continue
  fi
  if [ -e "$ROOT/overlap/$new" ]; then
    echo "  ! 目标已存在，跳过: $b -> $new"; continue
  fi
  echo "  $b -> $new"
  [ "$APPLY" = "--apply" ] && mv -n "$ROOT/overlap/$b" "$ROOT/overlap/$new"
  n=$((n+1))
done
if [ "$APPLY" = "--apply" ]; then echo "已改名 $n 个目录。"
else echo "以上是预演，共 $n 个。加 --apply 真正执行。"; fi
