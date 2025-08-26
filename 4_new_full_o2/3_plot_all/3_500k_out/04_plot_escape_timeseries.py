#!/usr/bin/env python3
# -------------------------------------------------------------
# 对 escape_timeseries.csv 画 (T vs t) / (z vs t)
# -------------------------------------------------------------

# ===== USER CONFIG =====
TS_CSV      = "escape_timeseries.csv"
TIME_START  = 15.0         # ps
TIME_END    = 45.0         # ps
OUT_DIR     = "plots_ts_split"   # 统一放图的文件夹
PLOT_MODE   = "split"      # split | both | none
# =======================

import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, sys

df = pd.read_csv(TS_CSV)
t_min, t_max = df["time_ps"].min(), df["time_ps"].max()
print(f"[INFO] time_ps in file: {t_min:.3g} – {t_max:.3g} ps")

# ---- 筛选时间窗口 ----
df_w = df[(df["time_ps"] >= TIME_START) & (df["time_ps"] <= TIME_END)]
if df_w.empty:
    sys.exit(f"[WARN] 在 {TIME_START}-{TIME_END} ps 区间内没有数据，"
             f"请调整 TIME_START / TIME_END")

# ---- 输出目录 ----
os.makedirs(OUT_DIR, exist_ok=True)

sns.set(style="whitegrid"); plt.rcParams["font.size"] = 11
palette = sns.color_palette("husl", n_colors=max(df_w["id"].nunique(), 3))

for i, (a_id, g) in enumerate(df_w.groupby("id", sort=False)):
    g = g.sort_values("time_ps")

    if PLOT_MODE in ("both", "split"):
        # T–t
        plt.figure(figsize=(4,3))
        plt.plot(g["time_ps"], g["T"], color=palette[i], lw=1.2)
        plt.xlabel("time (ps)"); plt.ylabel("T (K)"); plt.title(f"id {a_id}")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/T_id{a_id}.png", dpi=300); plt.close()

        # z–t
        plt.figure(figsize=(4,3))
        plt.plot(g["time_ps"], g["z"], color=palette[i], lw=1.2)
        plt.xlabel("time (ps)"); plt.ylabel("z (Å)"); plt.title(f"id {a_id}")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/z_id{a_id}.png", dpi=300); plt.close()

    if PLOT_MODE == "both":
        fig, ax1 = plt.subplots(figsize=(5,3))
        ax1.plot(g["time_ps"], g["T"], color="tab:red", lw=1.2)
        ax1.set_xlabel("time (ps)"); ax1.set_ylabel("T (K)", color="tab:red")
        ax1.tick_params(axis='y', labelcolor="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(g["time_ps"], g["z"], color="tab:blue", lw=1.2)
        ax2.set_ylabel("z (Å)", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")
        plt.title(f"id {a_id}")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/both_id{a_id}.png", dpi=300); plt.close()

print(f"[OK] 图像已保存到 ./{OUT_DIR}/ 目录下")
