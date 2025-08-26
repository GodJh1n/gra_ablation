#!/usr/bin/env python3
# -------------------------------------------------------------
#   A) 单文件散点  : 均线 + P-percent 线
#   B) 汇总直方图 : 正态拟合 + 三条参考线
#                   (实红 = 样本均值, 实橙 = 样本P分位,
#                    虚黑 = 正态P分位)
# -------------------------------------------------------------

# ===== USER CONFIG =====
JUMP_FILES = [
    "jump_csv/500k_jump_stats.csv",
    "jump_csv/1000k_jump_stats.csv"
]
ALL_FILE   = "jump_csv/all_jump_stats.csv"
P_PERCENT  = 0.20      # 0.20 → 20 %    0.80 → 80 %
BIN_SIZE_K = 100       # 直方图 bin 宽 (K)
OUT_DIR    = "jump_csv/plots"
# =======================

import os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import norm

os.makedirs(OUT_DIR, exist_ok=True)
sns.set(style="whitegrid"); plt.rcParams["font.size"] = 11

# ───────────────────────────  A) 散点  ───────────────────────────
for f in JUMP_FILES:
    if not os.path.exists(f):
        print(f"[!] 跳过不存在文件 {f}")
        continue
    df = (pd.read_csv(f)
            .dropna(subset=["avg_T_K"])
            .sort_values("id"))
    if df.empty:
        print(f"[!] {f} 为空")
        continue

    mean_T = df["avg_T_K"].mean()
    perc_T = df["avg_T_K"].quantile(P_PERCENT)

    plt.figure(figsize=(6,3))
    plt.scatter(df["id"], df["avg_T_K"], s=28)
    plt.axhline(mean_T,  color="red",    lw=1.4,
                label=f"mean {mean_T:.1f} K")
    plt.axhline(perc_T, color="orange", lw=1.4,
                label=f"{int(P_PERCENT*100)} % ≤ {perc_T:.1f} K")

    plt.xlabel("id"); plt.ylabel("avg T near jump (K)")
    plt.legend(); plt.tight_layout()
    name = os.path.splitext(os.path.basename(f))[0]
    plt.savefig(f"{OUT_DIR}/{name}_scatter.png", dpi=300); plt.close()
    print(f"[✓] {name}_scatter.png")

# ─────────────────────────  B) 直方图  ─────────────────────────
if os.path.exists(ALL_FILE):
    all_df = (pd.read_csv(ALL_FILE)
                .dropna(subset=["avg_T_K"]))
    if not all_df.empty:
        data = all_df["avg_T_K"].values
        mean_sample  = data.mean()
        perc_sample  = all_df["avg_T_K"].quantile(P_PERCENT)

        # —— 直方图 —— (密度归一化)
        plt.figure(figsize=(6,4))
        bins = np.arange(0, data.max()+BIN_SIZE_K, BIN_SIZE_K)
        plt.hist(data, bins=bins, color="skyblue",
                 edgecolor="k", alpha=.7, density=True)

        # —— 正态拟合 ——
        mu, sigma = norm.fit(data)
        xs = np.linspace(0, data.max()*1.05, 400)
        plt.plot(xs, norm.pdf(xs, mu, sigma),
                 'k-', lw=2,
                 label=f"Normal fit  μ={mu:.1f}, σ={sigma:.1f}")

        # —— 样本均值 & 分位 ——
        plt.axvline(mean_sample,  color="red",   lw=1.6,
                    label=f"mean {mean_sample:.1f} K")
        plt.axvline(perc_sample,  color="orange",lw=1.6,
                    label=f"{int(P_PERCENT*100)} % ≤ {perc_sample:.1f} K")

        # —— 正态 P-percent 分位 (黑虚线) ——
        perc_norm = norm.ppf(P_PERCENT, mu, sigma)
        plt.axvline(perc_norm, color="black", ls="--", lw=1.6,
                    label=f"Normal {int(P_PERCENT*100)} % = {perc_norm:.1f} K")

        plt.xlabel("avg T near jump (K)")
        plt.ylabel("density")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/hist_all_jump.png", dpi=300); plt.close()
        print(f"[✓] hist_all_jump.png  |  μ={mu:.1f} σ={sigma:.1f}  "
              f"|  Norm-{int(P_PERCENT*100)} %={perc_norm:.1f} K")
    else:
        print("[!] all_jump_stats.csv 为空")
else:
    print("[!] 未找到 all_jump_stats.csv，跳过直方图")
