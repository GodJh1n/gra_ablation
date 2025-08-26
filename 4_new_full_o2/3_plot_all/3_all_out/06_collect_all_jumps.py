#!/usr/bin/env python3
# -------------------------------------------------------------
# 递归扫描 *_jump_stats.csv  ➜  all_jump_stats.csv
# 自动保留 6 个应力分量列（若缺失填 NaN）
# -------------------------------------------------------------

# ===== USER CONFIG =====
ROOT_DIR     = "jump_csv"            # 待扫描根目录
GLOB_PATTERN = "[0-9]*jump_stats.csv"     # 文件通配
OUT_FILE     = "jump_csv/all_jump_stats.csv"

# 从文件名提取运行温度标签（500k / 1200K …）
REGEX_TEMP   = r'(\d+)[Kk]'

# 想保留的应力列名（与 extxyz / timeseries 写入保持一致）
STRESS_COLS  = [
    "v_s_xx_gpa", "v_s_yy_gpa", "v_s_zz_gpa",
    "v_s_xy_gpa", "v_s_xz_gpa", "v_s_yz_gpa"
]
# =======================

import os, glob, re, pandas as pd

rows = []
for path in glob.glob(os.path.join(ROOT_DIR, GLOB_PATTERN), recursive=True):
    df = pd.read_csv(path)
    if df.empty:
        continue

    # ---- 解析文件名中的温度标签 ----
    m = re.search(REGEX_TEMP, os.path.basename(path))
    df["run_T_K"] = int(m.group(1)) if m else None
    df["src"]     = os.path.relpath(path, ROOT_DIR)

    # ---- 确保应力 6 列都存在 ----
    for col in STRESS_COLS:
        if col not in df.columns:
            df[col] = pd.NA          # 用 <NA> 占位，方便 concat

    rows.append(df)

# ---- 合并 & 输出 ----
if rows:
    master = pd.concat(rows, ignore_index=True)

    # 按惯例把关键信息放前面（可选）
    front_cols = ["id", "avg_T_K", "t_jump_ps"] + STRESS_COLS
    front_cols = [c for c in front_cols if c in master.columns]
    master = master[ front_cols + [c for c in master.columns if c not in front_cols] ]

    master.to_csv(OUT_FILE, index=False)
    print(f"[✓] 汇总完成 → {OUT_FILE}  (n={len(master)})")
else:
    print("[!] 未找到匹配文件，或全部为空")
