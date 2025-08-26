#!/usr/bin/env python3
# -------------------------------------------------------------
# 从 escape_timeseries.csv 识别跳点
# 输出 jump_stats.csv :  温度 + 6 应力分量 + σ_eq(Dev) + P_hydro
# -------------------------------------------------------------

# ===== USER CONFIG =====
TS_CSV        = "escape_timeseries.csv"

TIME_START_PS = 15.0      # 分析窗口
TIME_END_PS   = 45.0

T_MIN, T_MAX  =   0, 10000   # 温度合法区间 (过滤坏值)

# ---- 跳点判据 ----
BASE_WIN_PS   = 0.05       # 前 0.05 ps 做基线
Z_ABS_TH      = 5.0        # z − z0 ≥ TH Å → 记为跳点
DT_AVG_PS     = 0.05       # 跳点 ±Δt 求局部均值

# ---- 应力列名（与 timeseries 保持一致）----
S_XX, S_YY, S_ZZ = "v_s_xx_gpa", "v_s_yy_gpa", "v_s_zz_gpa"
S_XY, S_XZ, S_YZ = "v_s_xy_gpa", "v_s_xz_gpa", "v_s_yz_gpa"
STRESS_COLS = [S_XX, S_YY, S_ZZ, S_XY, S_XZ, S_YZ]
# =======================

import numpy as np, pandas as pd, math, sys, os

if not os.path.exists(TS_CSV):
    sys.exit(f"[ERR] 找不到 {TS_CSV}")

df = pd.read_csv(TS_CSV)

# ---- 时间窗裁剪 ----
df = df[df["time_ps"].between(TIME_START_PS, TIME_END_PS)]
if df.empty:
    sys.exit("[ERR] 时间窗内无数据")

# ---------- (1) 全局平均 T (500–3000 K) ----------
(df[df["T"].between(500, 3000)]
   .groupby("time_ps", sort=True)["T"]
   .mean()
   .reset_index()
   .to_csv("all_T.csv", index=False))
print("[OK] all_T.csv 已保存")

# ---------- (2) 跳点检测 + 应力统计 ----------
rows = []
for aid, g in df.groupby("id", sort=False):
    g = g.sort_values("time_ps").reset_index(drop=True)

    # —— 基线 z0 ——
    z0_mask = g["time_ps"] <= (TIME_START_PS + BASE_WIN_PS)
    if not z0_mask.any(): continue
    z0 = g.loc[z0_mask, "z"].mean()

    # —— 首帧抬升 ≥ Z_ABS_TH ——
    idx_jump = np.where((g["z"] - z0) >= Z_ABS_TH)[0]
    if idx_jump.size == 0: continue
    i_jump  = idx_jump[0]
    t_jump  = g.loc[i_jump, "time_ps"]

    # —— 跳点 ±Δt 窗口 ——
    win = g["time_ps"].between(t_jump - DT_AVG_PS, t_jump + DT_AVG_PS)
    wdf = g.loc[win]

    # —— 平均温度 ——
    T_local = wdf["T"].mask(~wdf["T"].between(T_MIN, T_MAX))
    if T_local.notna().empty: continue
    avg_T = T_local.mean()

    # —— 平均 6 应力分量 (若列不存在则 NaN) ——
    stress_avg = {col: wdf[col].mean() if col in wdf else math.nan
                  for col in STRESS_COLS}

    # —— 等效应力与体应力 ——
    sxx, syy, szz = [stress_avg[c] for c in (S_XX,S_YY,S_ZZ)]
    if not math.isnan(sxx):
        P_hydro = (sxx + syy + szz) / 3.0
        sigma_dev = 0.5*math.sqrt((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2)
    else:
        P_hydro = sigma_dev = math.nan

    rows.append(
        [aid, t_jump,
         t_jump-DT_AVG_PS, t_jump+DT_AVG_PS,
         avg_T,
         stress_avg[S_XX], stress_avg[S_YY], stress_avg[S_ZZ],
         stress_avg[S_XY], stress_avg[S_XZ], stress_avg[S_YZ],
         sigma_dev, P_hydro]
    )

cols = ["id","t_jump_ps","win_start_ps","win_end_ps","avg_T_K",
        S_XX,S_YY,S_ZZ,S_XY,S_XZ,S_YZ,
        "sigma_dev_GPa","P_hydro_GPa"]

pd.DataFrame(rows, columns=cols).to_csv("jump_stats.csv", index=False)
print(f"[OK] jump_stats.csv 已保存  (n={len(rows)})")
