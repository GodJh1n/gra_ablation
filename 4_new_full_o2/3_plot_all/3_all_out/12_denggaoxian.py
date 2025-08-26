#!/usr/bin/env python3
# -------------------------------------------------------------
#   逃逸成功点 → 2-D 核密度 → 累积概率等高线
# -------------------------------------------------------------

# ===== USER CONFIG =====
JUMP_CSV     = "./jump_csv/all_jump_stats.csv"           # 只需正样本
X_COL        = "avg_T_K"                  # 横轴列名    (温度)
Y_COL        = "v_s_zz_gpa"     # 纵轴列名    (应力可换成 v_s_zz_gpa 等)
LEVELS_CDF   = (0.3, 0.5, 0.8)            # 累积概率等高线 (0<C<1)
GRID_T       = (500, 3000, 200)           # (min, max, N)  – 温度轴网格
GRID_S       = (-30,   30, 200)           # (min, max, N)  – 应力轴网格
BANDWIDTH    = None                       # KDE bandwidth; None=Scott
OUT_PNG      = "escape_isoContours_z.png"
OUT_TXT      = "isoContour_ranges_z.txt"
# =======================

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
from scipy.stats import gaussian_kde

# ---------- 1. 读取正样本 ----------
df = pd.read_csv(JUMP_CSV, usecols=[X_COL, Y_COL]).dropna()
X = df[X_COL].to_numpy()
Y = df[Y_COL].to_numpy()
print(f"[INFO] 正样本点数 = {len(df)}")

# ---------- 2. KDE ----------
kde = gaussian_kde([X, Y], bw_method=BANDWIDTH)

T_lin = np.linspace(*GRID_T)
S_lin = np.linspace(*GRID_S)
TT, SS = np.meshgrid(T_lin, S_lin)
coords = np.vstack([TT.ravel(), SS.ravel()])
PDF = kde(coords).reshape(TT.shape)           # 概率密度 f(T,σ)

# ---------- 3. 由 PDF → CDF 等高线阈值 ----------
pdf_flat  = PDF.ravel()
idx_sort  = np.argsort(pdf_flat)[::-1]        # 降序（大→小）
pdf_sort  = pdf_flat[idx_sort]
area      = (T_lin[1]-T_lin[0]) * (S_lin[1]-S_lin[0])
cdf_sort  = np.cumsum(pdf_sort) * area

thr_raw = [pdf_sort[np.searchsorted(cdf_sort, p)]
           for p in LEVELS_CDF]

# 升序 + 同步百分比
thr, LEVELS_CDF = zip(*sorted(zip(thr_raw, LEVELS_CDF)))

print("[INFO] 阈值(pdf) =", ["%.3e" % t for t in thr])


# ---------- 4. 画图 ----------
sns.set(style="whitegrid"); plt.rcParams["font.size"] = 11
plt.figure(figsize=(7,5))

# 背景填色：概率密度
plt.contourf(TT, SS, PDF, levels=30, cmap="rocket_r", alpha=.85)

# 轮廓线：累积概率阈值
CS = plt.contour(TT, SS, PDF, levels=thr, colors=["orange","cyan","blue"],
                 linewidths=[2,2,2], linestyles=["-","--","-."])
fmt = {thr[i]:f"CDF={LEVELS_CDF[i]*100:.0f}%" for i in range(len(thr))}
plt.clabel(CS, CS.levels, inline=True, fontsize=9, fmt=fmt)

# 散点
plt.scatter(X, Y, s=12, c="white", edgecolor="k", linewidth=.3, alpha=.6)

plt.xlabel("Temperature (K)")
plt.ylabel("signed σ_dev (GPa)")
plt.title("Escape iso-probability contours  (positive samples only)")
plt.tight_layout(); plt.savefig(OUT_PNG, dpi=300)
plt.close()
print(f"[✓] 图像保存 → {OUT_PNG}")

# ---------- 5. 每条封闭曲线的 (T,σ) 范围 ----------
with open(OUT_TXT, "w") as fout:
    for lev, p in zip(CS.levels, LEVELS_CDF):
        # 找对应 PDF ≥ lev 的格子
        mask = PDF >= lev
        if not mask.any(): continue
        Tmin, Tmax = T_lin[mask.any(0)].min(), T_lin[mask.any(0)].max()
        Smin, Smax = S_lin[mask.any(1)].min(), S_lin[mask.any(1)].max()
        fout.write(f"CDF {p:5.2f}  :  T [{Tmin:6.0f}, {Tmax:6.0f}] K   "
                   f"σ_dev [{Smin:6.1f}, {Smax:6.1f}] GPa\n")
print(f"[✓] 等高线范围写入 → {OUT_TXT}")
