#!/usr/bin/env python3
# -------------------------------------------------------------
# 应力-温度耦合图集
# -------------------------------------------------------------

# ===== USER CONFIG =====
DATA_CSV      = "jump_csv/stress_jump_stats.csv"
OUT_DIR       = "jump_csv/plots_stress"
# SIG_EQ_COL    = "sigma_vm_GPa"       # 可改 "sigma_vm_GPa"
SIG_EQ_COL = "sigma_dev_signed_GPa"


P_PERCENT     = 0.20                  # 20 %
BIN_K         = 0.5                   # 直方图 bin 宽 (GPa)
KDE_GRIDSIZE  = 100
# =======================

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import norm

os.makedirs(OUT_DIR, exist_ok=True)
sns.set(style="whitegrid"); plt.rcParams["font.size"] = 11

df = pd.read_csv(DATA_CSV).dropna(subset=["avg_T_K", SIG_EQ_COL])
if df.empty:
    raise SystemExit("[ERR] 数据为空，检查路径或列名")

# ───────── 1. T vs σeq ─────────
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="avg_T_K", y=SIG_EQ_COL, s=30, alpha=.7)
plt.xlabel("avg T near jump (K)")
plt.ylabel(f"{SIG_EQ_COL} (GPa)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/scatter_T_sigma.png", dpi=300); plt.close()

# ───────── 2. σxy vs σzz （若有） ─────────
if all(c in df.columns for c in ["v_s_xy_gpa","v_s_zz_gpa"]):
    plt.figure(figsize=(5,4))
    sns.scatterplot(data=df, x="v_s_zz_gpa", y="v_s_xy_gpa", s=30, alpha=.7)
    plt.xlabel("σ_zz  (GPa)"); plt.ylabel("τ_xy  (GPa)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/scatter_szz_txy.png", dpi=300); plt.close()

# ───────── 3. σeq 直方图 + 正态拟合 ─────────
sig = df[SIG_EQ_COL].values
mu, sigma = norm.fit(sig)
sig_min, sig_max = sig.min(), sig.max()
bins = np.arange(np.floor(sig_min) - BIN_K, sig_max + BIN_K, BIN_K)

plt.figure(figsize=(6,4))
plt.hist(sig, bins=bins, color="skyblue", edgecolor="k",
         alpha=.7, density=True, label="hist")
xs = np.linspace(sig.min()*1.05, sig.max()*1.05, 400)
plt.plot(xs, norm.pdf(xs, mu, sigma), 'k-', lw=2,
         label=f"Normal μ={mu:.2f}, σ={sigma:.2f} GPa")
perc = np.quantile(sig, P_PERCENT)
plt.axvline(sig.mean(), color="red",   lw=1.6, label=f"mean {sig.mean():.2f}")
plt.axvline(perc,       color="orange",lw=1.6,
            label=f"{int(P_PERCENT*100)} % ≤ {perc:.2f}")
plt.axvline(norm.ppf(P_PERCENT, mu, sigma), ls="--", color="black", lw=1.6,
            label=f"N({P_PERCENT*100:.0f}%) {norm.ppf(P_PERCENT,mu,sigma):.2f}")
plt.xlabel(f"{SIG_EQ_COL} (GPa)"); plt.ylabel("density")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/hist_{SIG_EQ_COL}.png", dpi=300); plt.close()

# ───────── 4. T-σeq 二维 KDE (修正 x= y=) ─────────
plt.figure(figsize=(6,5))
sns.kdeplot(
    data=df, x="avg_T_K", y=SIG_EQ_COL,
    cmap="rocket", fill=True, thresh=0.02, levels=100,
    gridsize=KDE_GRIDSIZE
)
plt.xlabel("avg T near jump (K)")
plt.ylabel(f"{SIG_EQ_COL} (GPa)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/kde_T_sigma.png", dpi=300); plt.close()

print("[✓] 全部图像已输出到", OUT_DIR)
