#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot & poly-fit:
  - Time series: stress(unit-selectable), virial(kcal/mol), rate(1/ps)
  - Scatter & polynomial fits: rate vs stress(unit-selectable), and rate vs virial
"""

# ============================ PARAMS (EDIT HERE) ============================
CSV_MAIN    = "stress_rate_binned.csv"
CSV_SIGMA   = "stress_rate_binned_sorted_by_sigma.csv"
CSV_VIRIAL  = "stress_rate_binned_sorted_by_virial.csv"

# columns in CSV
X_SIGMA_GPA_COL = "sigma_vm_mean_gpa"      # 原始是 GPa
X_VIR_COL       = "vm_virial_mean_kcalmol" # kcal/mol
Y_COL           = "rate_mean_per_ps"       # 1/ps

# 选择要显示/拟合的“应力单位”：'GPa' | 'MPa' | 'kPa'
STRESS_UNIT = "MPa"

# 多项式阶数
POLY_DEGREE = 4

# 可选：仅绘制/拟合这个时间窗口（单位 ps）
T_MIN_PS = None     # 例如 55.0
T_MAX_PS = None

# 可选：应力坐标筛选（按照 STRESS_UNIT 解释）
X_SIGMA_MIN = None  # 例如 0
X_SIGMA_MAX = None  # 例如 2.0e9  # (kPa)

# 输出文件（会自动带上单位后缀）
OUT_STRESS_TIME = None  # 若为 None，自动命名
OUT_VIR_TIME    = "virial_vs_time_kcalmol.png"
OUT_RATE_TIME   = "rate_vs_time.png"
OUT_FIT_SIGMA_PNG  = None  # 若为 None，自动命名
OUT_FIT_VIR_PNG    = "stress_rate_fit_virial.png"
OUT_FITS_JSON      = "stress_rate_fit_coeffs.json"
OUT_CURVE_SIGMA_CSV = None
OUT_CURVE_VIR_CSV   = "stress_rate_fit_curve_virial.csv"
# ===========================================================================

import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

_SCALE = {"GPa": 1.0, "MPa": 1e3, "kPa": 1e6}  # 从 GPa → 目标单位 的倍率
_LABEL = {"GPa": "GPa", "MPa": "MPa", "kPa": "kPa"}

def _load_csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else None

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def poly_fit(x, y, deg):
    coefs = np.polyfit(x, y, deg=deg)
    poly  = np.poly1d(coefs)
    yhat  = poly(x)
    return coefs, poly, float(r2_score(y, yhat))

def _apply_time_crop(df, tmin, tmax):
    if df is None or df.empty:
        return df
    out = df.copy()
    if tmin is not None: out = out[out['t_center_ps'] >= tmin]
    if tmax is not None: out = out[out['t_center_ps'] <= tmax]
    return out

def main():
    if STRESS_UNIT not in _SCALE:
        raise SystemExit(f"STRESS_UNIT must be one of {_SCALE.keys()}")

    # 自动输出名
    unit_tag = STRESS_UNIT.lower()
    stress_label = f"Mean escape von Mises stress ({_LABEL[STRESS_UNIT]})"
    out_stress_time = OUT_STRESS_TIME or f"stress_vs_time_{unit_tag}.png"
    out_fit_sigma   = OUT_FIT_SIGMA_PNG or f"stress_rate_fit_sigma_{unit_tag}.png"
    out_curve_sigma = OUT_CURVE_SIGMA_CSV or f"stress_rate_fit_curve_sigma_{unit_tag}.csv"

    # 载入主表并裁剪时间
    df = _load_csv(CSV_MAIN)
    if df is None or df.empty:
        raise SystemExit(f"CSV not found or empty: {CSV_MAIN}")
    df = _apply_time_crop(df, T_MIN_PS, T_MAX_PS)

    # 把 GPa 列换算到目标单位（仅用于作图/拟合，不改源文件）
    sigma_scale = _SCALE[STRESS_UNIT]
    df['sigma_plot'] = df[X_SIGMA_GPA_COL].to_numpy(float) * sigma_scale

    # ---------- time series ----------
    # stress vs time（目标单位）
    plt.figure()
    plt.plot(df['t_center_ps'], df['sigma_plot'], marker='o', lw=1)
    plt.xlabel('Time (ps)'); plt.ylabel(stress_label)
    plt.title('Mean escape stress vs time'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_stress_time, dpi=300); plt.close()

    # virial vs time（kcal/mol）
    plt.figure()
    plt.plot(df['t_center_ps'], df[X_VIR_COL], marker='o', lw=1)
    plt.xlabel('Time (ps)'); plt.ylabel('Mean escape “von Mises” virial (kcal/mol)')
    plt.title('Mean escape virial vs time'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_VIR_TIME, dpi=300); plt.close()

    # rate vs time（1/ps）
    plt.figure()
    plt.plot(df['t_center_ps'], df[Y_COL], marker='o', lw=1)
    plt.xlabel('Time (ps)'); plt.ylabel('Formation rate (CO+CO2) [1/ps]')
    plt.title('Formation rate vs time'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_RATE_TIME, dpi=300); plt.close()

    # ---------- scatter & poly-fit: rate vs stress(目标单位) ----------
    ds = _load_csv(CSV_SIGMA)
    if ds is None or ds.empty:
        ds = df.dropna(subset=[X_SIGMA_GPA_COL, Y_COL]).copy().sort_values(X_SIGMA_GPA_COL)
    ds = _apply_time_crop(ds, T_MIN_PS, T_MAX_PS)
    ds['sigma_plot'] = ds[X_SIGMA_GPA_COL].to_numpy(float) * sigma_scale

    if X_SIGMA_MIN is not None: ds = ds[ds['sigma_plot'] >= X_SIGMA_MIN]
    if X_SIGMA_MAX is not None: ds = ds[ds['sigma_plot'] <= X_SIGMA_MAX]

    xs = ds['sigma_plot'].to_numpy(float)
    ys = ds[Y_COL].to_numpy(float)

    coefs_s, poly_s, r2_s = poly_fit(xs, ys, POLY_DEGREE)
    xs_plot = np.linspace(np.min(xs), np.max(xs), 400)
    ys_plot = poly_s(xs_plot)

    plt.figure()
    plt.scatter(xs, ys, s=26, label='Binned points')
    plt.plot(xs_plot, ys_plot, label=f'Poly deg={POLY_DEGREE}, R²={r2_s:.3f}')
    plt.xlabel(stress_label); plt.ylabel('Formation rate (CO+CO2) [1/ps]')
    plt.title(f'Rate vs stress ({_LABEL[STRESS_UNIT]})'); plt.grid(True, alpha=0.3); plt.legend()
    if (X_SIGMA_MIN is not None) or (X_SIGMA_MAX is not None):
        xmin = X_SIGMA_MIN if X_SIGMA_MIN is not None else np.min(xs)
        xmax = X_SIGMA_MAX if X_SIGMA_MAX is not None else np.max(xs)
        plt.xlim(xmin, xmax)
    plt.tight_layout(); plt.savefig(out_fit_sigma, dpi=300); plt.close()

    pd.DataFrame({"sigma_"+unit_tag: xs_plot, "y_poly": ys_plot}).to_csv(out_curve_sigma, index=False)

    # ---------- scatter & poly-fit: rate vs virial(kcal/mol) ----------
    dv = _load_csv(CSV_VIRIAL)
    if dv is None or dv.empty:
        dv = df.dropna(subset=[X_VIR_COL, Y_COL]).copy().sort_values(X_VIR_COL)
    dv = _apply_time_crop(dv, T_MIN_PS, T_MAX_PS)
    xv = dv[X_VIR_COL].to_numpy(float); yv = dv[Y_COL].to_numpy(float)
    coefs_v, poly_v, r2_v = poly_fit(xv, yv, POLY_DEGREE)
    xv_plot = np.linspace(np.min(xv), np.max(xv), 400)
    yv_plot = poly_v(xv_plot)

    plt.figure()
    plt.scatter(xv, yv, s=26, label='Binned points')
    plt.plot(xv_plot, yv_plot, label=f'Poly deg={POLY_DEGREE}, R²={r2_v:.3f}')
    plt.xlabel('Mean escape “von Mises” virial (kcal/mol)')
    plt.ylabel('Formation rate (CO+CO2) [1/ps]')
    plt.title('Rate vs virial (kcal/mol)'); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT_FIT_VIR_PNG, dpi=300); plt.close()

    pd.DataFrame({X_VIR_COL: xv_plot, "y_poly": yv_plot}).to_csv(OUT_CURVE_VIR_CSV, index=False)

    # 存储拟合系数
    Path(OUT_FITS_JSON).write_text(json.dumps({
        "y_unit": "1/ps",
        "stress_unit": _LABEL[STRESS_UNIT],
        "poly_degree": POLY_DEGREE,
        "sigma_fit": {"coeffs_high_to_low": list(map(float, coefs_s)), "R2": r2_s},
        "virial_fit": {"coeffs_high_to_low": list(map(float, coefs_v)), "R2": r2_v},
    }, indent=2), encoding='utf-8')

    print(f"[OK] wrote images & fits: {out_stress_time}, {OUT_VIR_TIME}, {OUT_RATE_TIME}, "
          f"{out_fit_sigma}, {OUT_FIT_VIR_PNG}, {OUT_FITS_JSON}, {out_curve_sigma}, {OUT_CURVE_VIR_CSV}")

if __name__ == "__main__":
    main()
