#!/usr/bin/env python3
# 读取 escape_summary.csv 画直方图/散点图
# ---------------------------------------------

SUM_CSV   = "escape_summary.csv"
OUT_PREFIX = "escape"

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv(SUM_CSV)

sns.set(style="whitegrid"); plt.rcParams["font.size"] = 11

# 1) 直方图 (T_escape vs T_peak)
plt.figure(figsize=(7,3))
sns.histplot(df["T_escape_K"].dropna(), color="b", kde=True, label="T_escape")
sns.histplot(df["T_peak_K"].dropna(),   color="orange", alpha=.6, kde=True, label="T_peak")
plt.legend(); plt.xlabel("Temperature (K)")
plt.tight_layout(); plt.savefig(f"{OUT_PREFIX}_hist.png", dpi=300)

# 2) t_peak vs t_escape
plt.figure(figsize=(4,4))
sns.scatterplot(df, x="t_escape_ps", y="t_peak_ps")
mn,mx = df[["t_escape_ps","t_peak_ps"]].min().min(), df.max().max()
plt.plot([mn,mx],[mn,mx],"--",c="k")
plt.xlabel("t_escape (ps)"); plt.ylabel("t_peak (ps)")
plt.tight_layout(); plt.savefig(f"{OUT_PREFIX}_tpeak_vs_escape.png", dpi=300)

# 3) T_peak vs t_escape
plt.figure(figsize=(4,4))
sns.scatterplot(df, x="t_escape_ps", y="T_peak_K")
plt.xlabel("t_escape (ps)"); plt.ylabel("T_peak (K)")
plt.tight_layout(); plt.savefig(f"{OUT_PREFIX}_Tpeak_vs_escape.png", dpi=300)

print(f"[PLOT] 图像输出 → {OUT_PREFIX}_*.png")
