#!/usr/bin/env python3
# extract_detached_ids_ovito_pycharm.py
# ---------------------------------------------
# 从 LAMMPS dump 中读取指定帧，用 z_top + 阈值 d_th 判定“脱离 C 原子”，仅输出 CSV
# 需要: pip install ovito numpy

# ========== USER CONFIG ==========
DUMP_PATH   = "trajectory.T_300_v7.76.lammpstrj"  # 你的轨迹
FRAME_IDX   = 120                                  # 0-based 帧序号
CARBON_TYPE = 1                                    # C 的粒子类型号
D_THRESH_A  = 25.0                                 # 脱离判据: d = z - z_top > D_THRESH_A
CORE_WIDTH  = 5.0                                  # 片层核心: |z - z_mode| <= CORE_WIDTH (Å)
OUT_CSV     = None                                 # 输出文件名; None=自动命名
# 可选：若你固定层 id<=1500 想作为“片面基准”，设为 True（多数情况保持 False）
USE_FIXED_LAYER = False
FIXED_LAYER_MAX_ID = 1500
# 附带导出的可选列（若 dump 里有这些字段会自动写出）
EXTRA_COLS = ["v_MyTemp","c_MyKE","vx","vy","vz",
              "v_s_xx_gpa","v_s_yy_gpa","v_s_zz_gpa",
              "v_s_xy_gpa","v_s_xz_gpa","v_s_yz_gpa"]
# =================================

import sys, numpy as np
from ovito.io import import_file

def robust_top_surface_z(z_c, core_width=5.0, hist_bins=120):
    counts, edges = np.histogram(z_c, bins=hist_bins)
    b = np.argmax(counts)
    z_mode = 0.5 * (edges[b] + edges[b+1])
    core = z_c[np.abs(z_c - z_mode) <= core_width]
    if core.size < 10:
        z_top = np.percentile(z_c, 75.0)
    else:
        z_top = np.percentile(core, 95.0)
    return float(z_top), float(z_mode)

def main():
    pipe = import_file(DUMP_PATH)
    if FRAME_IDX < 0 or FRAME_IDX >= pipe.source.num_frames:
        sys.exit(f"[ERROR] FRAME_IDX 超界: 0..{pipe.source.num_frames-1}")
    data = pipe.compute(FRAME_IDX)
    P = data.particles

    # 基础列
    pid  = P['Particle Identifier'].array.astype(int)
    ptyp = P['Particle Type'].array.astype(int)
    pos  = P.positions  # (N,3)

    # 择基准片面
    if USE_FIXED_LAYER:
        base_mask = (pid <= FIXED_LAYER_MAX_ID)
        if not np.any(base_mask):
            sys.exit("[ERROR] 未找到 fixed layer（id<=FIXED_LAYER_MAX_ID）")
        z_base = pos[base_mask, 2]
        z_top = float(np.percentile(z_base, 95.0))
        z_mode = float(np.median(z_base))
    else:
        maskC = (ptyp == CARBON_TYPE)
        z_c = pos[maskC, 2]
        z_top, z_mode = robust_top_surface_z(z_c, CORE_WIDTH)

    # 判定脱离（仅对 C）
    maskC = (ptyp == CARBON_TYPE)
    ids_c = pid[maskC]
    pos_c = pos[maskC]
    d_above = pos_c[:, 2] - z_top
    det_mask = d_above > D_THRESH_A

    # 需要导出的可选列
    opt_names = [n for n in EXTRA_COLS if n in P]

    rows = []
    for i, ok in enumerate(det_mask):
        if not ok: continue
        row = [int(ids_c[i]),
               float(pos_c[i,0]), float(pos_c[i,1]), float(pos_c[i,2]),
               float(d_above[i])]
        for n in opt_names:
            row.append(float(P[n].array[maskC][i]))
        rows.append(row)

    out = OUT_CSV or f"detached_ids_frame{FRAME_IDX}.csv"
    with open(out, "w") as f:
        header = ["id","x","y","z","d_above(Å)"] + opt_names
        ts = data.attributes.get('Timestep', None)
        f.write(f"# frame_idx,{FRAME_IDX}\n")
        if ts is not None: f.write(f"# Timestep,{ts}\n")
        f.write(f"# z_mode,{z_mode:.6f}\n# z_top,{z_top:.6f}\n")
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"[OK] 帧 {FRAME_IDX}: z_top={z_top:.3f} Å, 脱离数={len(rows)} → {out}")

if __name__ == "__main__":
    main()
