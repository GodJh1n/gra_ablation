#!/usr/bin/env python3
from ase.build import graphene_nanoribbon
from ase import Atoms
from ase.io import write
import numpy as np

# --- 1. User-Configurable Parameters ---
N_WIDTH_UNITS   = 50     # Units along armchair edge (~6 nm)
N_LENGTH_UNITS  = 30      # Units along zigzag edge (~6.4 nm)
C_C_BOND        = 1.44    # C–C bond length in Å
NUM_LAYERS      = 10       # Number of graphene layers
INTERLAYER_DZ   = 3.35    # Interlayer spacing in Å
BOTTOM_OFFSET   = 10.0    # 底层最低 z 偏移 (对应 LAMMPS 固定层阈值 12 Å)
VAC             = 300.0   # 真空厚度 in Å
OUTPUT_FILENAME = "hopg_init_big.data"

# --- 2. Create Single-Layer Template via Robust Axis-Swap ---
# 在 x–z 平面生成 sheet，并在厚度方向留出半层间隔的真空
gnr = graphene_nanoribbon(
    n=N_WIDTH_UNITS,
    m=N_LENGTH_UNITS,
    type='armchair',
    sheet=True,
    C_C=C_C_BOND,
    vacuum=INTERLAYER_DZ/2
)
# 旋转到 x–y 平面
gnr.euler_rotate(theta=90)
l = gnr.cell.lengths()
gnr.cell = gnr.cell.new((l[0], l[2], l[1]))
# 居中并开启三向周期
gnr.center(axis=2)
gnr.pbc = [True, True, True]

# --- 3. Stack Layers with 指定层间距 ---
stack = gnr.repeat((1, 1, NUM_LAYERS))
# 将所有层整体上移，使底层在 z ≥ BOTTOM_OFFSET
stack.positions[:, 2] += BOTTOM_OFFSET

# --- 4. Add Vacuum in z Direction and Wrap Atoms ---
cell = stack.get_cell()
cell[2, 2] += VAC
stack.set_cell(cell, scale_atoms=False)
stack.wrap()

# --- 5. Write Out LAMMPS Data File ---
write(
    OUTPUT_FILENAME,
    stack,
    format='lammps-data',
    atom_style='charge',
    specorder=['C']
)
print(f"✅ Wrote '{OUTPUT_FILENAME}', total atoms = {len(stack)}")

# --- 6. 简单诊断：z 范围 & 晶胞高度 ---
zs = stack.positions[:, 2]
cell_z = stack.get_cell()[2, 2]
print(f"  • z_min = {zs.min():.3f} Å, z_max = {zs.max():.3f} Å")
print(f"  • cell_z (含真空) = {cell_z:.3f} Å")