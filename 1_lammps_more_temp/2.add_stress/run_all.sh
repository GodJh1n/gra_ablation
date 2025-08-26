#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

for dir in *_temp; do
  [[ -d "$dir" ]] || continue
  echo ">>> 进入 $dir 并启动 lammps.sh"
  (
    cd "$dir"
    if [[ -f sub.sh ]]; then
      chmod +x lammps.sh
      ./lammps.sh
    else
      echo "    跳过：未找到 lammps.sh"
    fi
  )
done

echo "全部目录已处理完毕。"

