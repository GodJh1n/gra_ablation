#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract binned (stress, rate) pairs from LAMMPS outputs.
- Keeps BOTH raw per-atom virial (kcal/mol) and converted stress (GPa).
- Fixed volume per atom: V_atom = (Lx*Ly/N_layer_atoms) * t_eff, all in Å units (box fixed, no change_box).
"""

# ============================ PARAMS (EDIT HERE) ============================
# Inputs
SPECIES_PATH      = "species.out"
ABLATEDUMP_PATH   = "ablate.lammpstrj"

# LAMMPS time step (fs) and analysis knobs
DT_FS             = 0.1        # LAMMPS 'timestep' in fs
BIN_PS            = 2.0        # time-bin width for pairing stress & rate [ps]
ESCAPE_DELTAT_PS  = 0.05       # ±Δt window for per-atom escape stress averaging [ps]
MIN_ESCAPES_PER_BIN = 1        # keep bins with >= this many escaped atoms

# Box/geometry constants for per-atom "volume" (Å, Å^2, Å^3)
LX_A              = 62.35382907247957
LY_A              = 64.800000000000011
N_LAYER_ATOMS     = 1500        # atoms per single C layer
T_EFF_A           = 3.35         # nominal single-layer thickness (Å)
V_ATOM_A3         = (LX_A * LY_A / N_LAYER_ATOMS) * T_EFF_A   # constant Å^3 per atom

# Conversion: (kcal/mol) / Å^3  ->  GPa
KCALMOL_A3_TO_GPA = 6.947695

# Smoothing of rate (points of moving average; 1 = no smooth)
SMOOTH_RATE_POINTS = 1

# Optional time filter for final table (use None to disable)
T_MIN_PS = None
T_MAX_PS = None

# Outputs
OUT_CHRONO_CSV           = "stress_rate_binned.csv"
OUT_SORTED_SIGMA_CSV     = "stress_rate_binned_sorted_by_sigma.csv"
OUT_SORTED_VIRIAL_CSV    = "stress_rate_binned_sorted_by_virial.csv"
OUT_META_JSON            = "stress_rate_meta.json"
# ===========================================================================

import re, json, numpy as np, pandas as pd
from pathlib import Path


def _safe_num(x):
    try:
        return float(x)
    except Exception:
        try:
            return int(x)
        except Exception:
            return x


def parse_species(species_path: str, dt_fs: float) -> pd.DataFrame:
    rows, cols = [], None
    with open(species_path, 'r', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                cols = re.sub(r'^#+\s*', '', s).split()
                continue
            if cols is None:
                continue
            vals = s.split()
            if len(vals) < len(cols):
                continue
            rows.append({c: _safe_num(v) for c, v in zip(cols, vals)})
    if not rows:
        raise RuntimeError(f"No data parsed from {species_path}")

    df = pd.DataFrame(rows)
    if 'Timestep' not in df.columns:
        ts = next((c for c in df.columns if c.lower() == 'timestep'), None)
        if ts is None:
            raise RuntimeError("No 'Timestep' column in species file")
        df = df.rename(columns={ts: 'Timestep'})

    # identify CO & CO2 columns robustly
    lowers = {c: c.lower() for c in df.columns}
    co_cols  = [c for c, l in lowers.items() if ('co' in l and 'co2' not in l)]
    co2_cols = [c for c, l in lowers.items() if 'co2' in l]
    if not co_cols and 'CO' in df.columns:   co_cols = ['CO']
    if not co2_cols and 'CO2' in df.columns: co2_cols = ['CO2']

    df['CO_CO2_total'] = 0.0
    if co_cols:  df['CO_CO2_total'] += df[co_cols].astype(float).sum(axis=1)
    if co2_cols: df['CO_CO2_total'] += df[co2_cols].astype(float).sum(axis=1)

    dt_ps = dt_fs * 1e-3
    df['time_ps'] = df['Timestep'].astype(float) * dt_ps
    df = df.sort_values('time_ps').drop_duplicates('time_ps', keep='last').reset_index(drop=True)

    # rate = dN/dt (per ps)
    t = df['time_ps'].to_numpy()
    y = df['CO_CO2_total'].to_numpy()
    if len(t) >= 3 and np.ptp(t) > 0:
        rate = np.gradient(y, t)
    else:
        rate = np.zeros_like(y)
    if SMOOTH_RATE_POINTS and SMOOTH_RATE_POINTS > 1:
        k = int(SMOOTH_RATE_POINTS)
        rate = np.convolve(rate, np.ones(k)/k, mode='same')

    df['rate_per_ps'] = rate
    return df[['time_ps','Timestep','CO_CO2_total','rate_per_ps']]


def parse_ablate_dump(ablate_path: str, dt_fs: float):
    """
    Read ablate.lammpstrj with columns:
      id ... c_MyStress[1..6]
    Returns dict[atom_id] -> DataFrame(['time_ps','vxx','vyy','vzz','vxy','vxz','vyz'])
    NOTE: 'v' prefixes denote virial components (kcal/mol).
    """
    atom = {}
    with open(ablate_path, 'r', errors='ignore') as fh:
        tps, cols, reading_atoms = None, None, False
        while True:
            line = fh.readline()
            if not line:
                break
            s = line.strip()
            if s.startswith('ITEM:'):
                reading_atoms = False
                if 'TIMESTEP' in s:
                    ts = fh.readline().strip()
                    tps = int(ts) * dt_fs * 1e-3
                elif 'ATOMS' in s:
                    cols = s.replace('ITEM: ATOMS', '').strip().split()
                    reading_atoms = True
                continue
            if reading_atoms and cols is not None:
                toks = s.split()
                if not toks:
                    continue
                rec = dict(zip(cols, toks))
                try:
                    aid = int(rec.get('id', toks[0]))
                    vxx = float(rec['c_MyStress[1]']); vyy = float(rec['c_MyStress[2]']); vzz = float(rec['c_MyStress[3]'])
                    vxy = float(rec['c_MyStress[4]']); vxz = float(rec['c_MyStress[5]']); vyz = float(rec['c_MyStress[6]'])
                except Exception:
                    continue
                atom.setdefault(aid, []).append((tps, vxx, vyy, vzz, vxy, vxz, vyz))

    out = {}
    for aid, lst in atom.items():
        arr = np.array(lst, dtype=float)
        df = pd.DataFrame(arr, columns=['time_ps','vxx','vyy','vzz','vxy','vxz','vyz']).sort_values('time_ps')
        out[aid] = df.reset_index(drop=True)
    return out


def von_mises(a11, a22, a33, a12, a13, a23):
    term1 = ((a11 - a22)**2 + (a22 - a33)**2 + (a33 - a11)**2) / 2.0
    term2 = 3.0 * (a12**2 + a13**2 + a23**2)
    return np.sqrt(np.maximum(term1 + term2, 0.0))


def compute_escape_metrics(atom_traj: dict,
                           escape_deltat_ps: float,
                           v_atom_a3: float,
                           kconv: float):
    """
    For each atom, window-average virial components (kcal/mol) around its last time,
    and also convert to stress (GPa) using constant per-atom volume.
    Returns DataFrame:
      ['atom_id','escape_time_ps',
       'vm_virial_kcalmol', 'sigma_vm_gpa',
       'n_frames_used',
       'vxx','vyy','vzz','vxy','vxz','vyz']  # window mean per component (kcal/mol)
    """
    rows = []
    for aid, df in atom_traj.items():
        if df.empty:
            continue
        te = float(df['time_ps'].max())
        win = df[(df['time_ps'] >= te - escape_deltat_ps) & (df['time_ps'] <= te + escape_deltat_ps)]
        if win.empty:
            win = df.iloc[[-1]]

        # window means of virial components (kcal/mol)
        vxx = win['vxx'].mean(); vyy = win['vyy'].mean(); vzz = win['vzz'].mean()
        vxy = win['vxy'].mean(); vxz = win['vxz'].mean(); vyz = win['vyz'].mean()

        # virial "von Mises" (same algebra, still kcal/mol)
        vm_v_kcal = float(von_mises(vxx, vyy, vzz, vxy, vxz, vyz))

        # convert to stresses: sigma = -(virial / V) * factor
        sxx = -(vxx / v_atom_a3) * kconv
        syy = -(vyy / v_atom_a3) * kconv
        szz = -(vzz / v_atom_a3) * kconv
        sxy = -(vxy / v_atom_a3) * kconv
        sxz = -(vxz / v_atom_a3) * kconv
        syz = -(vyz / v_atom_a3) * kconv

        vm_sig_gpa = float(von_mises(sxx, syy, szz, sxy, sxz, syz))

        rows.append({
            'atom_id': aid,
            'escape_time_ps': te,
            'vm_virial_kcalmol': vm_v_kcal,
            'sigma_vm_gpa': vm_sig_gpa,
            'n_frames_used': int(len(win)),
            'vxx': vxx, 'vyy': vyy, 'vzz': vzz, 'vxy': vxy, 'vxz': vxz, 'vyz': vyz
        })
    return pd.DataFrame(rows)


def time_bin_aggregate(esc_df: pd.DataFrame, sp_df: pd.DataFrame,
                       bin_ps: float, min_escapes: int = 1) -> pd.DataFrame:
    # time span
    tmin = min(esc_df['escape_time_ps'].min() if not esc_df.empty else np.inf,
               sp_df['time_ps'].min() if not sp_df.empty else np.inf)
    tmax = max(esc_df['escape_time_ps'].max() if not esc_df.empty else -np.inf,
               sp_df['time_ps'].max() if not sp_df.empty else -np.inf)
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        return pd.DataFrame()

    edges = np.arange(tmin, tmax + bin_ps*0.5, bin_ps)
    if len(edges) < 2:
        edges = np.array([tmin, tmax])

    # bin & stats for escape virial/stress
    e2 = esc_df.copy()
    e2['bin'] = pd.cut(e2['escape_time_ps'], bins=edges, right=False, include_lowest=True)
    g_e = e2.groupby('bin', observed=False).agg(
        n_escaped=('atom_id','count'),
        sigma_vm_mean_gpa=('sigma_vm_gpa','mean'),
        sigma_vm_median_gpa=('sigma_vm_gpa','median'),
        sigma_vm_std_gpa=('sigma_vm_gpa','std'),
        vm_virial_mean_kcalmol=('vm_virial_kcalmol','mean'),
        vm_virial_median_kcalmol=('vm_virial_kcalmol','median'),
        vm_virial_std_kcalmol=('vm_virial_kcalmol','std')
    )

    # bin & stats for rate
    s2 = sp_df.copy()
    s2['bin'] = pd.cut(s2['time_ps'], bins=edges, right=False, include_lowest=True)
    g_r = s2.groupby('bin', observed=False)['rate_per_ps'].agg(
        n_rate_samples='count',
        rate_mean_per_ps='mean',
        rate_median_per_ps='median',
        rate_std_per_ps='std'
    )

    merged = g_e.join(g_r, how='outer').reset_index()

    # extract left/right safely
    b = merged['bin'].astype('object')
    merged['bin_start_ps'] = b.apply(lambda iv: float(iv.left) if isinstance(iv, pd.Interval) else np.nan)
    merged['bin_end_ps']   = b.apply(lambda iv: float(iv.right) if isinstance(iv, pd.Interval) else np.nan)
    merged.drop(columns=['bin'], inplace=True)
    merged['t_center_ps']  = 0.5*(merged['bin_start_ps'] + merged['bin_end_ps'])

    merged = merged.fillna({'n_escaped':0,'n_rate_samples':0})
    merged = merged[(merged['n_escaped'] >= min_escapes) & (merged['n_rate_samples'] > 0)]
    merged = merged.sort_values('t_center_ps').reset_index(drop=True)

    # optional time crop for final output
    if T_MIN_PS is not None:
        merged = merged[merged['t_center_ps'] >= T_MIN_PS]
    if T_MAX_PS is not None:
        merged = merged[merged['t_center_ps'] <= T_MAX_PS]

    return merged


def main():
    spec_p = Path(SPECIES_PATH); abl_p = Path(ABLATEDUMP_PATH)
    if not spec_p.exists(): raise FileNotFoundError(spec_p)
    if not abl_p.exists():  raise FileNotFoundError(abl_p)

    sp_df = parse_species(str(spec_p), DT_FS)
    atom_traj = parse_ablate_dump(str(abl_p), DT_FS)

    if not atom_traj:
        pd.DataFrame().to_csv(OUT_CHRONO_CSV, index=False)
        pd.DataFrame().to_csv(OUT_SORTED_SIGMA_CSV, index=False)
        pd.DataFrame().to_csv(OUT_SORTED_VIRIAL_CSV, index=False)
        Path(OUT_META_JSON).write_text(json.dumps({"warn":"no atoms"}, indent=2))
        print("[WARN] No atoms found in ablate.lammpstrj")
        return

    esc_df = compute_escape_metrics(atom_traj, ESCAPE_DELTAT_PS, V_ATOM_A3, KCALMOL_A3_TO_GPA)
    out_df = time_bin_aggregate(esc_df, sp_df, BIN_PS, MIN_ESCAPES_PER_BIN)

    out_df.to_csv(OUT_CHRONO_CSV, index=False)
    print(f"[OK] wrote {OUT_CHRONO_CSV}  rows={len(out_df)}")

    # sorted versions (for plotting/fits)
    if not out_df.empty:
        out_df.sort_values('sigma_vm_mean_gpa').to_csv(OUT_SORTED_SIGMA_CSV, index=False)
        out_df.sort_values('vm_virial_mean_kcalmol').to_csv(OUT_SORTED_VIRIAL_CSV, index=False)

    meta = {
        "params":{
            "DT_FS":DT_FS, "BIN_PS":BIN_PS, "ESCAPE_DELTAT_PS":ESCAPE_DELTAT_PS,
            "MIN_ESCAPES_PER_BIN":MIN_ESCAPES_PER_BIN, "SMOOTH_RATE_POINTS":SMOOTH_RATE_POINTS
        },
        "volume_model":{
            "LX_A":LX_A, "LY_A":LY_A, "N_LAYER_ATOMS":N_LAYER_ATOMS, "T_EFF_A":T_EFF_A,
            "V_ATOM_A3":V_ATOM_A3, "KCALMOL_A3_TO_GPA":KCALMOL_A3_TO_GPA
        },
        "input":{"species":str(spec_p), "ablate":str(abl_p)},
        "rows": int(len(out_df))
    }
    Path(OUT_META_JSON).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"[OK] wrote {OUT_META_JSON}")


if __name__ == "__main__":
    main()
