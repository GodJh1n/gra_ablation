#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for dir in *_temp; do
  [[ -d "$dir" ]] || continue
  echo ">>> 进入 $dir 并启动 sub.sh"
  (
    cd "$dir"
    if [[ -f sub.sh ]]; then
      chmod +x sub.sh
      sbatch sub.sh
    else
      echo "    跳过：未找到 sub.sh"
    fi
  )
done

echo "全部目录已处理完毕。"
 
