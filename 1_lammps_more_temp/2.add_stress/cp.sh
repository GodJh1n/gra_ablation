#!/usr/bin/env bash
set -euo pipefail

# 要复制的文件（在当前目录）
FILES=(hopg_init.data ffield.reax.cho in.hopg_T param.qeq sub.sh)

# 检查文件是否存在
for f in "${FILES[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "缺少文件：$f" >&2
    exit 1
  fi
done

# 生成 500,700,900,1100,1300,1500 这些温度的目录并复制文件
for T in $(seq 500 100 2500); do
  DIR="${T}_temp"
  mkdir -p "$DIR"
  cp -f "${FILES[@]}" "$DIR/"
  sed -i -E "s/^\s*variable\s+T\s+equal\s+[0-9.]+/variable T equal ${T}.0/" "$DIR/in.hopg_T"
  echo "创建并更新：$DIR"
done

echo "全部完成。"

