#!/usr/bin/env python3
# -------------------------------------------------------------
# 逐帧收集指定 ID 的 z、温度、(可选应力)
# 计算 t_escape / T_escape / t_peak / T_peak
# -------------------------------------------------------------

# ===== USER CONFIG =====
XYZ_PATH      = "./500_temp/1500_4500_500k.xyz"   # 多帧 extxyz
IDS_CSV       = "escaped_ids.csv"                # 前一步生成
START_FRAME   = 0
END_FRAME     = 3000        # exclusive
DT_FS         = 0.1         # fs per MD step
TIME_ZERO_PS  = 15         # 把 START_FRAME 对应到 15 ps
Z_THRESH_A    = 27.0
TEMP_MIN, TEMP_MAX = 0, 100000
OUT_TS_CSV    = "escape_timeseries.csv"
OUT_SUM_CSV   = "escape_summary.csv"
# 如果想导出应力, 列出 extxyz 中的列名 (可留空 [])
STRESS_COLS   = ["v_s_xx_gpa","v_s_yy_gpa","v_s_zz_gpa","v_s_xy_gpa","v_s_xz_gpa","v_s_yz_gpa"]
# =======================

import numpy as np, pandas as pd, ase.io, math, sys, csv

# ---------- 读取待追踪 ID ----------
try:
    # 跳过注释 (#) + 表头 (id)
    ids = np.loadtxt(IDS_CSV, comments="#", dtype=int, skiprows=1, ndmin=1)
except ValueError as e:        # 文件只有一行 id 时 numpy 会返回标量
    sys.exit(f"读取 {IDS_CSV} 失败: {e}")
if ids.size == 0:
    sys.exit(f"{IDS_CSV} 中没有任何 id")

# --------- 结果容器 ---------
rows_ts  = []    # time-series : 每帧 × 每 id
rows_sum = []    # summary    : 每 id

# 预先算好实际的时间 (ps)
time_ps = (np.arange(START_FRAME, END_FRAME) * DT_FS * 100 / 1000.0) + TIME_ZERO_PS

# --------- 逐帧读取 extxyz ---------
for k, at in enumerate(
        ase.io.iread(XYZ_PATH,
                     index=slice(START_FRAME, END_FRAME),
                     format="extxyz")):
    pid = at.arrays["id"].astype(int)      # id:I:1
    z   = at.positions[:, 2]

    # ---- 选择温度列 ----
    if "v_mytemp" in at.arrays:            # 首选 v_mytemp
        T_arr = at.arrays["v_mytemp"]
    elif "c_myke" in at.arrays:            # 退而求其次 c_myke
        T_arr = (2/3) * at.arrays["c_myke"] / 0.0019872041
    else:
        T_arr = np.full_like(z, np.nan)    # 都没有 → NaN

    # id → index 映射
    id2idx = {int(a): i for i, a in enumerate(pid)}

    # ---- 写入 time-series 行 ----
    for a_id in ids:
        idx = id2idx.get(int(a_id))
        if idx is None:
            continue                       # 本帧里可能已删除
        row = {
            "id":       int(a_id),
            "frame":    START_FRAME + k,
            "time_ps":  time_ps[k],
            "z":        float(z[idx]),
            "T":        np.nan if
                        (T_arr[idx] < TEMP_MIN or T_arr[idx] > TEMP_MAX)
                        else float(T_arr[idx]),
        }
        # 应力列 (可选)
        for sc in STRESS_COLS:
            row[sc] = ( float(at.arrays[sc][idx])
                        if sc in at.arrays else np.nan )
        rows_ts.append(row)

# --------- 保存 time-series CSV ---------
df_ts = pd.DataFrame(rows_ts)
df_ts.to_csv(OUT_TS_CSV, index=False)
print(f"[COLLECT] time-series  →  {OUT_TS_CSV}")

# --------- 计算 summary ---------
for a_id, g in df_ts.groupby("id", sort=False):
    # t_escape = 第一帧 z > Z_THRESH_A
    esc_idx = np.where(g["z"] > Z_THRESH_A)[0]
    if esc_idx.size:
        ie  = esc_idx[0]
        t_e = float(g.iloc[ie]["time_ps"])
        T_e = float(g.iloc[ie]["T"])
    else:
        t_e = T_e = math.nan

    # 峰值温度 (忽略 NaN)
    if np.isfinite(g["T"]).any():
        ip   = g["T"].idxmax()
        T_p  = float(g.loc[ip, "T"])
        t_p  = float(g.loc[ip, "time_ps"])
    else:
        T_p = t_p = math.nan

    rows_sum.append([int(a_id), t_e, T_e, t_p, T_p])

pd.DataFrame(rows_sum,
             columns=["id", "t_escape_ps", "T_escape_K",
                      "t_peak_ps", "T_peak_K"]
             ).to_csv(OUT_SUM_CSV, index=False)
print(f"[COLLECT] summary      →  {OUT_SUM_CSV}")
