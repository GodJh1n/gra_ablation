#!/usr/bin/env python3
# track_ids_plot.py
# 读取 IDs 列表，从 start:end 帧跟踪它们的 z 和温度，出两张图（均值±std 或 逐原子）

import argparse, csv, math
import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread

def load_ids_from_csv(path):
    ids = []
    with open(path, "r") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            try:
                ids.append(int(row[0]))
            except ValueError:
                # 跳过表头行
                if row[0] == "id":
                    continue
    return np.array(sorted(set(ids)), dtype=int)

def main():
    ap = argparse.ArgumentParser(description="Track detached atom IDs across frames and plot Z/T vs time.")
    ap.add_argument("--dump", required=True, help="LAMMPS dump file (text)")
    ap.add_argument("--ids-csv", required=True, help="CSV from step 1 (first column must be id)")
    ap.add_argument("--start-frame", type=int, required=True, help="0-based start frame index (where IDs were selected)")
    ap.add_argument("--end-frame", type=int, default=None, help="0-based exclusive end frame (default: to the end)")
    ap.add_argument("--dt-fs", type=float, default=0.1, help="timestep in fs for time axis")
    ap.add_argument("--per-atom", action="store_true", help="plot per-atom traces instead of mean±std")
    ap.add_argument("--estimate-T-from-KE", action="store_true",
                    help="if v_MyTemp not present, estimate T from c_MyKE (T = 2/3 KE / kB)")
    ap.add_argument("--kB-kcal", type=float, default=0.0019872041, help="Boltzmann constant in kcal/mol/K")
    ap.add_argument("--figsize", default="6,3.2", help="figure size WxH in inches")
    ap.add_argument("--font", type=int, default=11)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--prefix", default="track", help="output prefix")
    args = ap.parse_args()

    track_ids = load_ids_from_csv(args.ids_csv)
    if track_ids.size == 0:
        raise SystemExit("No IDs found in CSV.")

    W, H = (float(x) for x in args.figsize.split(","))
    plt.rcParams.update({
        "font.size": args.font,
        "axes.labelsize": args.font,
        "axes.titlesize": args.font+1,
        "xtick.labelsize": args.font,
        "ytick.labelsize": args.font,
        "legend.fontsize": args.font,
    })

    # 用流式读取，避免一次性载入
    frames = []
    Z_list = []
    T_list = []
    idx0 = args.start_frame
    idx1 = args.end_frame

    # 建议：dump 很大时，iread 的 index=slice(idx0, idx1, 1)
    for i, atoms in enumerate(iread(args.dump, index=slice(idx0, idx1, 1), format="lammps-dump-text")):
        frame_idx = idx0 + i
        frames.append(frame_idx)

        # id→index 映射
        if "id" not in atoms.arrays:
            raise SystemExit("dump 缺少 'id' 列，无法跟踪。")
        ids = atoms.arrays["id"].astype(int)
        id2ind = {int(i): j for j, i in enumerate(ids)}

        # 位置（z）
        pos = atoms.positions
        z = np.full(track_ids.shape, np.nan)
        for k, a_id in enumerate(track_ids):
            j = id2ind.get(int(a_id), None)
            if j is not None:
                z[k] = pos[j, 2]
        Z_list.append(z)

        # 温度（优先用 v_MyTemp；没有时可选从 c_MyKE 估算）
        T_arr = np.full(track_ids.shape, np.nan)
        if "v_MyTemp" in atoms.arrays:
            arr = atoms.arrays["v_MyTemp"]
            for k, a_id in enumerate(track_ids):
                j = id2ind.get(int(a_id), None)
                if j is not None:
                    T_arr[k] = float(arr[j])
        elif args.estimate_T_from_KE and "c_MyKE" in atoms.arrays:
            # c_MyKE 是每原子的动能（kcal/mol），用 3 自由度近似：T = 2/3 KE / kB
            ke = atoms.arrays["c_MyKE"]
            for k, a_id in enumerate(track_ids):
                j = id2ind.get(int(a_id), None)
                if j is not None:
                    T_arr[k] = (2.0/3.0) * float(ke[j]) / args.kB_kcal
        else:
            # 找不到温度
            pass
        T_list.append(T_arr)

    frames = np.array(frames, dtype=int)
    time_ps = frames * args.dt_fs / 1000.0
    Z = np.vstack(Z_list)  # shape: (n_frames, n_ids)
    T = np.vstack(T_list)  # same shape (可能为 NaN)

    # 统计（逐帧）
    Z_mean = np.nanmean(Z, axis=1)
    Z_std  = np.nanstd(Z, axis=1)
    T_mean = np.nanmean(T, axis=1) if np.isfinite(T).any() else None
    T_std  = np.nanstd(T, axis=1)  if np.isfinite(T).any() else None

    # ---- 图 1：Z vs time ----
    fig1, ax1 = plt.subplots(figsize=(W, H), constrained_layout=True)
    if args.per_atom:
        for j in range(Z.shape[1]):
            ax1.plot(time_ps, Z[:, j], linewidth=1.0, alpha=0.6)
        ax1.set_title(f"Tracked C (N={Z.shape[1]}) — z vs time (per atom)")
    else:
        ax1.plot(time_ps, Z_mean, linewidth=1.8, label="mean z")
        ax1.fill_between(time_ps, Z_mean-Z_std, Z_mean+Z_std, alpha=0.25, label="±1σ")
        ax1.set_title(f"Tracked C (N={Z.shape[1]}) — z vs time (mean ± std)")
        ax1.legend(frameon=False)
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("z (Å)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    fig1.savefig(f"{args.prefix}_z_vs_time.png", dpi=args.dpi)

    # ---- 图 2：Temp vs time ----
    fig2, ax2 = plt.subplots(figsize=(W, H), constrained_layout=True)
    if T_mean is None:
        ax2.text(0.5, 0.5, "No per-atom temperature available\n(v_MyTemp / c_MyKE not found)",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()
    else:
        if args.per_atom:
            for j in range(T.shape[1]):
                ax2.plot(time_ps, T[:, j], linewidth=1.0, alpha=0.6)
            ax2.set_title(f"Tracked C — temperature vs time (per atom)")
        else:
            ax2.plot(time_ps, T_mean, linewidth=1.8, label="mean T")
            ax2.fill_between(time_ps, T_mean-T_std, T_mean+T_std, alpha=0.25, label="±1σ")
            ax2.set_title(f"Tracked C — temperature vs time (mean ± std)")
            ax2.legend(frameon=False)
        ax2.set_xlabel("Time (ps)")
        ax2.set_ylabel("Temperature (K)")
        ax2.grid(True, linestyle="--", alpha=0.4)
    fig2.savefig(f"{args.prefix}_temp_vs_time.png", dpi=args.dpi)

    print(f"[OK] Saved: {args.prefix}_z_vs_time.png, {args.prefix}_temp_vs_time.png")

if __name__ == "__main__":
    main()
