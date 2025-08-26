#!/usr/bin/env python3
# -------------------------------------------------------------
#   生成 train_escape_map.csv  (T_K, sigma_GPa, escape)
#   正样本: jump_stats.csv 里各 id 的 t_jump
#   负样本: 每个 id 在 (t_jump-3 ps , t_jump) 内均匀挑 10 帧
# -------------------------------------------------------------

# ===== USER CONFIG =====
JUMP_CSV        = "jump_csv/stress_jump_stats.csv"    # 正样本
TS_CSV_PATTERN  = "escape_csv/*escape_timeseries*.csv"
PRE_WINDOW_PS   = 3.0            # 跳点前多远作为负样本池
NEG_POINTS_PER  = 10             # 每个 id 均匀取多少帧
T_RANGE         = (500, 3000)    # 过滤极端温度
SIG_TARGET      = "sigma_dev_signed_GPa"
OUT_TRAIN       = "train_escape_map.csv"
REGEX_TEMP      = r'(\d+)[Kk]'   # 文件名里提 run_T_K
# =======================

import os, re, glob, math
import numpy as np, pandas as pd

# ---------- util: 确保有 sigma_dev_signed_GPa ----------
def add_sigma_signed(df):
    if SIG_TARGET in df.columns:
        return df
    alt = [c for c in ["sigma_dev_GPa", "sigma_vm_GPa"] if c in df.columns]
    if alt:                     # 直接复用单列
        df[SIG_TARGET] = df[alt[0]]
        return df

    need = ["v_s_xx_gpa","v_s_yy_gpa","v_s_zz_gpa",
            "v_s_xy_gpa","v_s_xz_gpa","v_s_yz_gpa"]
    if all(c in df.columns for c in need):
        sx, sy, sz = (df[c] for c in need[:3])
        txy, txz, tyz = (df[c] for c in need[3:])
        sigma_eq = np.sqrt(((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 +
                            6*(txy**2 + txz**2 + tyz**2))/2.0)
        df[SIG_TARGET] = np.sign(sz) * sigma_eq
        return df
    raise KeyError(f"无法构造 {SIG_TARGET}")

# ---------- 1) 读正样本 jump_stats ----------
jump_df = pd.read_csv(JUMP_CSV)
jump_df = add_sigma_signed(jump_df)

jump_df = jump_df.rename(columns={"avg_T_K":"T_K",
                                  SIG_TARGET:"sigma_GPa"})
jump_df["escape"] = 1
print(f"[INFO] 正样本 {len(jump_df)} 条")

# 按 id -> t_jump 映射，便于后面查找窗口
id2tjump = jump_df.set_index("id")["t_jump_ps"].to_dict()

# ---------- 2) 为每个 id 抽负样本 ----------
neg_frames = []
for ts_path in glob.glob(TS_CSV_PATTERN):
    ts = pd.read_csv(ts_path)
    if ts.empty:
        continue

    ts = add_sigma_signed(ts)
    ts = ts[ts["T"].between(*T_RANGE)]

    # Parse run_T_K from file name
    m = re.search(REGEX_TEMP, os.path.basename(ts_path))
    runT = int(m.group(1)) if m else None
    ts["run_T_K"] = runT

    # 只保留 jump_df 中出现过的 id
    ts = ts[ts["id"].isin(id2tjump)]
    if ts.empty:
        continue

    # 对每个 id 生成局部负样本
    for aid, g in ts.groupby("id"):
        t_jump = id2tjump[aid]
        win = g[g["time_ps"].between(t_jump-PRE_WINDOW_PS, t_jump, inclusive="left")]
        if win.empty:
            continue
        # 均匀抽 NEG_POINTS_PER 个索引
        idxs = np.linspace(0, len(win)-1, NEG_POINTS_PER, dtype=int)
        sub = win.iloc[idxs][["id","T","run_T_K", SIG_TARGET]].copy()
        neg_frames.append(sub)

neg_df = (pd.concat(neg_frames, ignore_index=True)
            .rename(columns={"T":"T_K", SIG_TARGET:"sigma_GPa"}))
neg_df["escape"] = 0
print(f"[INFO] 负样本 {len(neg_df)} 条")

# ---------- 3) 合并 & 导出 ----------
out = pd.concat([jump_df[["id","T_K","run_T_K","sigma_GPa","escape"]],
                 neg_df], ignore_index=True)
out.to_csv(OUT_TRAIN, index=False)
print(f"[✓] 训练集写出 → {OUT_TRAIN}  (total={len(out)})")
