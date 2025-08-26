#!/usr/bin/env python3
# -------------------------------------------------------------
#  (T , σ_dev_signed) → P_escape 的 2-D 判逃面
#  支持 LogisticRegression 或 SVM-RBF
# -------------------------------------------------------------

# ===== USER CONFIG =====
TRAIN_CSV = "train_escape_map.csv"   # 由 10_prepare_escape_map_v2.py 生成
MODEL_TYPE = "logit"                 # "logit" | "svm"
GRID_T   = (500, 3000, 150)          # 温度网格  (min, max, N)
GRID_S   = (-20,   25, 150)          # σ_dev 网格 (GPa)
OUT_DIR  = "jump_csv/escape_map"     # 输出目录
# =======================

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import make_pipeline
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.metrics       import roc_auc_score

os.makedirs(OUT_DIR, exist_ok=True)
sns.set(style="whitegrid"); plt.rcParams["font.size"] = 11

# ---------- 1. 读训练集 ----------
df = (pd.read_csv(TRAIN_CSV)
        .dropna(subset=["T_K", "sigma_GPa", "escape"]))   # 去掉 NaN
print(f"[INFO] 样本量 = {len(df)}  (正 {df['escape'].sum()} / 负 {len(df)-df['escape'].sum()})")

X = df[["T_K", "sigma_GPa"]].to_numpy()
y = df["escape"].astype(int).to_numpy()

# ---------- 2. 拟合模型 ----------
if MODEL_TYPE == "logit":
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000))
elif MODEL_TYPE == "svm":
    model = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", probability=True, gamma="scale"))
else:
    raise ValueError("MODEL_TYPE 只能是 'logit' 或 'svm'")
model.fit(X, y)
auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
print(f"[INFO] 训练 AUC = {auc:.3f}")

# ---------- 3. 网格预测 ----------
T_lin = np.linspace(*GRID_T)
S_lin = np.linspace(*GRID_S)
TT, SS = np.meshgrid(T_lin, S_lin)
grid   = np.c_[TT.ravel(), SS.ravel()]
PP     = model.predict_proba(grid)[:,1].reshape(TT.shape)

# ---------- 4. 画判逃面 ----------
plt.figure(figsize=(7,5))
cf = plt.contourf(TT, SS, PP, levels=np.linspace(0,1,21),
                  cmap="inferno", alpha=.85)
plt.colorbar(cf, label="P_escape")

# P = 0.5 等势线
plt.contour(TT, SS, PP, levels=[0.5], colors="cyan",
            linewidths=2, linestyles="--")

# 训练散点
sns.scatterplot(data=df, x="T_K", y="sigma_GPa", hue="escape",
                palette={0:"#1f77b4", 1:"#ff7f0e"},
                edgecolor="k", linewidth=.3, s=26, alpha=.6)

plt.xlabel("Temperature (K)")
plt.ylabel("signed σ_dev (GPa)")
plt.title(f"Escape probability map  ({MODEL_TYPE},  AUC={auc:.3f})")
plt.legend(title="escape", loc="upper left")
plt.tight_layout()
out_png = f"{OUT_DIR}/escape_map_{MODEL_TYPE}.png"
plt.savefig(out_png, dpi=300)
plt.close()
print(f"[✓] 判逃面保存 → {out_png}")
