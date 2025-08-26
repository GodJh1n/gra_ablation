#!/usr/bin/env python3
# -------------------------------------------------------------
# 为后续作图准备数据：在跳点记录中追加等效应力指标
# -------------------------------------------------------------

# ===== USER CONFIG =====
IN_CSV      = "jump_csv/all_jump_stats.csv"
OUT_CSV     = "jump_csv/stress_jump_stats.csv"

# Stress 列在 CSV 中的名称（确保与 escape_timeseries 写入一致）
S_XX_COL, S_YY_COL, S_ZZ_COL = "v_s_xx_gpa", "v_s_yy_gpa", "v_s_zz_gpa"
# 若文件里有剪切分量，再填进去；否则自动跳过 von Mises 部分
TAU_XY_COL, TAU_YZ_COL, TAU_XZ_COL = "v_s_xy_gpa", "v_s_yz_gpa", "v_s_xz_gpa"
# =======================

import pandas as pd, numpy as np, os, sys

if not os.path.exists(IN_CSV):
    sys.exit(f"[ERR] 找不到 {IN_CSV}")

df = pd.read_csv(IN_CSV)
miss = [c for c in [S_XX_COL,S_YY_COL,S_ZZ_COL] if c not in df.columns]
if miss:
    sys.exit(f"[ERR] 缺少应力列 {miss}")

# ---- 体应力 & Deviatoric ----
sxx, syy, szz = [df[c].astype(float) for c in (S_XX_COL,S_YY_COL,S_ZZ_COL)]

df["P_hydro_GPa"]   = (sxx + syy + szz) / 3.0
df["sigma_dev_GPa"] = 0.5 * np.sqrt( (sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 )

# ---- von Mises (若有剪切) ----
have_shear = all(c in df.columns for c in (TAU_XY_COL,TAU_YZ_COL,TAU_XZ_COL))
if have_shear:
    txy, tyz, txz = [df[c].astype(float) for c in (TAU_XY_COL,TAU_YZ_COL,TAU_XZ_COL)]
    df["sigma_vm_GPa"] = np.sqrt(df["sigma_dev_GPa"]**2 + 3*(txy**2 + tyz**2 + txz**2))

df.to_csv(OUT_CSV, index=False)
print(f"[✓] 写出 {OUT_CSV}  (n={len(df)})")
