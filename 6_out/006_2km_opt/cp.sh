#!/usr/bin/env bash
set -euo pipefail

# 要复制的文件（在当前目录）
FILES=(ffield.reax.cho param.qeq hopg_with_O_final.data in.gra_o sub.sh)

# 检查文件是否存在
for f in "${FILES[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "缺少文件：$f" >&2
    exit 1
  fi
done

# 生成 500,700,900,1100,1300,1500 这些温度的目录并复制文件
for T in $(seq 500 200 1500); do
  DIR="${T}_temp"
  mkdir -p "$DIR"
  cp -f "${FILES[@]}" "$DIR/"
  sed -i -E "s/^\s*variable\s+T\s+equal\s+[0-9.]+/variable T equal ${T}.0/" "$DIR/in.gra_o"
  echo "创建并更新：$DIR"
done

echo "全部完成。"

