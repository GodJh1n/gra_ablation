#!/usr/bin/env python3
# -------------------------------------------------------------
# 1) 计算过滤平均 T  → all_T.csv   (500–3000 K)
# 2) 基线抬升 + 回溯找跳点 → jump_stats.csv
#    * 若该 ID 在跳点窗口内无合法温度 → 丢弃
# -------------------------------------------------------------

# ===== USER CONFIG =====
TS_CSV        = "escape_timeseries.csv"
TIME_START_PS = 5.0
TIME_END_PS   = 35.0
T_MIN, T_MAX  = 500, 3000      # 温度合法区间
BASE_WIN_PS   = 0.05           # 基线窗口
Z_ABS_TH      = 5.0            # 抬升阈值 (Å)
DT_AVG_PS     = 0.05           # 跳点局部均温 ±Δt
# =======================

import numpy as np, pandas as pd, math, sys

df = pd.read_csv(TS_CSV)

# ---- 时间窗裁剪 ----
df = df[df["time_ps"].between(TIME_START_PS, TIME_END_PS)]
if df.empty:
    sys.exit("时间窗内无数据")

# ---------- (1) 全局平均 T ----------
(df[df["T"].between(T_MIN, T_MAX)]
   .groupby("time_ps", sort=True)["T"]
   .mean()
   .reset_index()
   .to_csv("all_T.csv", index=False))
print("[OK] all_T.csv 已保存")

# ---------- (2) 跳点检测 ----------
rows = []
for aid, g in df.groupby("id", sort=False):
    g = g.sort_values("time_ps").reset_index(drop=True)

    # 基线 z0
    z0_mask = g["time_ps"] <= (TIME_START_PS + BASE_WIN_PS)
    if not z0_mask.any():
        continue
    z0 = g.loc[z0_mask, "z"].mean()

    # 首帧满足抬升阈值
    idx_jump = np.where((g["z"] - z0) >= Z_ABS_TH)[0]
    if idx_jump.size == 0:
        continue
    i_jump = idx_jump[0]
    t_jump = g.loc[i_jump, "time_ps"]

    # 跳点局部温度
    win = g["time_ps"].between(t_jump - DT_AVG_PS, t_jump + DT_AVG_PS)
    T_local = g.loc[win, "T"]
    T_local = T_local[T_local.between(T_MIN, T_MAX)]
    avg_T = T_local.mean() if not T_local.empty else math.nan

    if math.isnan(avg_T):          # ★ 无有效温度 → 跳过此 ID
        continue

    rows.append([aid, t_jump,
                 t_jump - DT_AVG_PS, t_jump + DT_AVG_PS,
                 avg_T])

pd.DataFrame(rows,
    columns=["id","t_jump_ps","win_start_ps","win_end_ps","avg_T_K"]
    ).to_csv("jump_stats.csv", index=False)
print("[OK] jump_stats.csv 已保存")
