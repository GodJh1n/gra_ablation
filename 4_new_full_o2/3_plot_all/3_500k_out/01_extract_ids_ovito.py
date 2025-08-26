#!/usr/bin/env python3
# -------------------------------------------------------------
# 判断规则：    C 原子 z > Z_THRESH_A  → 记为 escaped
# 输出：        escaped_ids.csv  (一行一个 id)
# -------------------------------------------------------------

# ===== 用户参数 =====
XYZ_PATH   = "./500_temp/4500_500k.xyz"   # 只有 1 帧的 Extended-XYZ
FRAME_IDX  = 0                              # 读第 0 帧
CARBON_TAG = "C"                            # 用 "C" 匹配 species
Z_THRESH_A = 27.0                           # 逃逸阈值 (Å)
OUT_IDS    = "escaped_ids.csv"
# ====================

import numpy as np, ase.io, sys

def get_type_array(at):
    """优先抓 type / species；否则用 chemical symbols。"""
    if "type" in at.arrays:
        return at.arrays["type"]
    if "species" in at.arrays:
        return at.arrays["species"]
    return np.asarray(at.get_chemical_symbols())

# --- 读取指定帧 ---
atoms = ase.io.read(XYZ_PATH, index=FRAME_IDX, format="extxyz")

ids   = atoms.arrays["id"].astype(int)   # id 列
types = get_type_array(atoms)            # 'C' / 'O' / …
z     = atoms.positions[:, 2]            # z 坐标

# --- 判定逃逸：同时满足 (a) 是碳, (b) z > 阈值 ---
mask_escape = (types == CARBON_TAG) & (z > Z_THRESH_A)
escaped_ids = ids[mask_escape]

# --- 写 CSV ---
with open(OUT_IDS, "w") as f:
    f.write(f"# frame_idx,{FRAME_IDX}\n")
    f.write(f"# z_threshold,{Z_THRESH_A}\n# id\n")
    for i in escaped_ids:
        f.write(f"{i}\n")

print(f"[OK] Escaped C atoms = {len(escaped_ids)}  → {OUT_IDS}")
