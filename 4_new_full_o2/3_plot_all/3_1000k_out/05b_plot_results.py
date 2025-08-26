#!/usr/bin/env python3
# -------------------------------------------------------------
# all_T.png : 所有 ID 温度曲线 + 过滤后均线
# avgT_vs_id.png : 跳点局部均温散点 + 全局均温水平线 + 线性拟合
# -------------------------------------------------------------

# ===== USER CONFIG =====
TS_CSV      = "escape_timeseries.csv"
ALL_T_CSV   = "all_T.csv"       # 已是 500–3000 K 过滤后的均值
JUMP_CSV    = "jump_stats.csv"
TIME_START  = 5.0
TIME_END    = 35.0
T_MIN, T_MAX = 500, 3000        # <—— 统一温度滤波阈值
OUT_DIR     = "plots"
# =======================

import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────  (1) all_T.png  ────────────────────────────────
df = pd.read_csv(TS_CSV)
df = df[(df["time_ps"].between(TIME_START, TIME_END)) &
        (df["T"].between(T_MIN, T_MAX))]            # ★ 温度过滤

sns.set(style="whitegrid"); plt.rcParams["font.size"] = 11
palette = sns.color_palette("husl", n_colors=df["id"].nunique())

fig, ax = plt.subplots(figsize=(6,4))
for i, (aid, g) in enumerate(df.groupby("id", sort=False)):
    ax.plot(g["time_ps"], g["T"], lw=.8, color=palette[i], alpha=.7)

df_mean = pd.read_csv(ALL_T_CSV)                    # 已是 500–3000 K
ax.plot(df_mean["time_ps"], df_mean["T"],
        color="red", lw=2,
        label=f"mean T ({T_MIN}–{T_MAX} K)")

ax.set_xlabel("time (ps)"); ax.set_ylabel("T (K)")
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_T.png", dpi=300); plt.close()
print("[✓] plots/all_T.png 保存")

# ───────────── (2) avgT_vs_id.png ─────────────
jump_df = (pd.read_csv(JUMP_CSV)
             .dropna(subset=["avg_T_K"])
             .query(f"{T_MIN} <= avg_T_K <= {T_MAX}"))   # 温度过滤

if not jump_df.empty:
    plt.figure(figsize=(6,3))
    plt.scatter(jump_df["id"], jump_df["avg_T_K"], s=28,
                label="per-id avg T")

    # —— 全体平均 ——
    global_avg = jump_df["avg_T_K"].mean()
    plt.axhline(global_avg, color="red", lw=1.5,
                label=f"overall mean = {global_avg:.1f} K")

    plt.xlabel("id"); plt.ylabel("avg T near jump (K)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/avgT_vs_id.png", dpi=300); plt.close()
    print("[✓] plots/avgT_vs_id.png 保存")
else:
    print("[!] jump_stats.csv 为空或全部超出温度阈值，未绘 avgT_vs_id")
