#!/usr/bin/env python3
# 过滤 LAMMPS dump：按帧区间、类型、id 列表、以及“脱离”阈值（相对于 z_top）筛选粒子，
# 输出瘦身后的 dump（仅包含需要的列）。

import argparse, csv
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import PythonModifier, DeleteSelectedModifier

# ---------- CLI ----------
ap = argparse.ArgumentParser(description="OVITO pre-filter for LAMMPS dump")
ap.add_argument("--in", dest="in_dump", required=True)
ap.add_argument("--out", dest="out_dump", required=True)
ap.add_argument("--start", type=int, required=True, help="start frame (inclusive)")
ap.add_argument("--end",   type=int, required=True, help="end frame (exclusive)")
ap.add_argument("--types", type=str, default=None, help='keep Particle Types, e.g. "1,2"')
ap.add_argument("--ids-csv", type=str, default=None, help="CSV with first column = id to keep")
ap.add_argument("--keep-only-ids", action="store_true", help="only keep IDs from --ids-csv (not just intersect)")
ap.add_argument("--detach-thresh", type=float, default=None, help="keep carbons with d = z - z_top > thresh")
ap.add_argument("--carbon-type", type=int, default=1, help="type id for carbon (for z_top calc)")
ap.add_argument("--columns", type=str, default="id,type,x,y,z,v_MyTemp,c_MyKE,v_s_xx_gpa,v_s_yy_gpa,v_s_zz_gpa",
                help="export columns, comma-separated (non-existent columns are ignored)")
args = ap.parse_args()

types_keep = None
if args.types:
    types_keep = np.array([int(t.strip()) for t in args.types.split(",") if t.strip()!=''], dtype=int)

ids_keep = None
if args.ids_csv:
    ids = []
    with open(args.ids_csv, "r") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"): continue
            if row[0].lower() == "id": continue
            try: ids.append(int(row[0]))
            except: pass
    ids_keep = np.array(sorted(set(ids)), dtype=int)

def robust_top_surface_z(z_c, core_width=5.0, hist_bins=120):
    if z_c.size == 0:
        return None
    counts, edges = np.histogram(z_c, bins=hist_bins)
    b = np.argmax(counts)
    z_mode = 0.5 * (edges[b] + edges[b+1])
    core = z_c[np.abs(z_c - z_mode) <= 5.0]
    if core.size < 10:
        z_top = np.percentile(z_c, 75.0)
    else:
        z_top = np.percentile(core, 95.0)
    return float(z_top)


# particle_filter 函数内的修正片段

def particle_filter(frame, data):
    P = data.particles
    N = P.count
    # 默认全不保留，只有满足条件的才设为True，逻辑更清晰
    final_keep = np.zeros(N, dtype=bool)

    # 基础列
    pid = P['Particle Identifier'].array.astype(int)
    ptyp = P['Particle Type'].array.astype(int)
    z = P.positions[:, 2]

    # ----- 新的逻辑流程 -----

    # 1) 如果指定了 --keep-only-ids，则它具有最高优先级
    if args.keep_only_ids and ids_keep is not None:
        # 直接将ID列表作为唯一的筛选条件
        final_keep = np.isin(pid, ids_keep)

    # 2) 否则，应用多个条件的组合（交集）
    else:
        # 先应用类型筛选
        type_mask = np.ones(N, dtype=bool)
        if types_keep is not None:
            type_mask = np.isin(ptyp, types_keep)

        # 再应用ID筛选
        id_mask = np.ones(N, dtype=bool)
        if ids_keep is not None:
            id_mask = np.isin(pid, ids_keep)

        # 求交集
        current_keep = type_mask & id_mask

        # 3) “脱离”筛选：这个筛选是特殊的，它是在当前保留的C原子基础上，剔除掉未脱离的
        if args.detach_thresh is not None:
            isC = (ptyp == args.carbon_type)
            # 只在当前已经确定要保留的C原子中计算z_top
            z_top = robust_top_surface_z(z[current_keep & isC])
            if z_top is not None:
                # 在当前保留的原子中，剔除那些是“未脱离的碳”
                is_non_detached_carbon = isC & ((z - z_top) <= args.detach_thresh)
                current_keep &= ~is_non_detached_carbon

        final_keep = current_keep

    # 在 OVITO 里，“Selection=1”表示将被 DeleteSelectedModifier 删除
    sel = data.particles_.create_property('Selection')
    sel.marray[:] = (~final_keep).astype(np.uint8)

# 构建管线并应用 Python 过滤 + 删除
pipe = import_file(args.in_dump)
if args.start < 0 or args.end <= args.start or args.end > pipe.source.num_frames:
    raise SystemExit(f"帧范围非法: 0..{pipe.source.num_frames-1}, 给定 [{args.start}, {args.end})")

pipe.modifiers.append(PythonModifier(function=particle_filter))
pipe.modifiers.append(DeleteSelectedModifier())

# 组织导出列（存在才导）
probe = pipe.compute(args.start)
P = probe.particles
col_req = [c.strip() for c in args.columns.split(",") if c.strip()!='']
columns = []
name_map = {
    "id":"Particle Identifier", "type":"Particle Type",
    "x":"Position.X", "y":"Position.Y", "z":"Position.Z",
    "vx":"Velocity.X", "vy":"Velocity.Y", "vz":"Velocity.Z",
}
for c in col_req:
    if c in name_map:
        columns.append(name_map[c])
    elif c in P:
        columns.append(c)
    else:
        # 静默忽略不存在列
        pass

if not columns:
    # 至少要导出 id,type,x,y,z
    columns = ["Particle Identifier","Particle Type",
               "Position.X","Position.Y","Position.Z"]

# 导出瘦身后的 dump（仅 [start,end) 帧）
export_file(pipe, args.out_dump, 'lammps/dump',
            columns=columns, start_frame=args.start, end_frame=args.end-1)

print(f"[OK] Exported filtered dump → {args.out_dump}")
