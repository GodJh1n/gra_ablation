#!/usr/bin/env python3
from ase.build import graphene_nanoribbon, bulk
from ase.io import write
import numpy as np

# --- 1. User-Configurable Parameters ---
N_WIDTH_UNITS   = 25     # Units along armchair edge
N_LENGTH_UNITS  = 15      # Units along zigzag edge
C_C_BOND        = 1.44   # C–C bond length in Å
NUM_LAYERS      = 5      # Number of graphene layers
INTERLAYER_DZ   = 3.35   # Interlayer spacing in Å
BOTTOM_OFFSET   = 10.0   # Target z-coordinate for the bottom layer
VAC             = 260.0   # Vacuum thickness in Å

# Oxygen FCC packing parameters:
O_CELL_CONST    = 18.0    # FCC lattice constant for O atoms (Å)
O_REP_XY        = 3    # Repetitions in x & y directions for O FCC
O_REP_Z         = 11     # Repetitions in z direction for O FCC
O_Z_CENTER      = 160.0   # Target z-coordinate for the CENTER of the O block (Å)

OUTPUT_FILENAME = "hopg_with_O_final.data"

# --- 2. Create Single-Layer Template ---
print("1. Creating graphene template...")
gnr = graphene_nanoribbon(
    n=N_WIDTH_UNITS,
    m=N_LENGTH_UNITS,
    type='armchair',
    sheet=True,
    C_C=C_C_BOND,
    vacuum=INTERLAYER_DZ/2
)
gnr.euler_rotate(theta=90)
l = gnr.cell.lengths()
gnr.cell = gnr.cell.new((l[0], l[2], l[1]))
gnr.center(axis=2)
gnr.pbc = [True, True, True]

# --- 3. Stack Layers ---
print("2. Stacking graphene layers...")
stack = gnr.repeat((1, 1, NUM_LAYERS))
# FIX: Robustly move the stack so the bottom layer is exactly at BOTTOM_OFFSET
min_z = stack.positions[:, 2].min()
stack.positions[:, 2] += (BOTTOM_OFFSET - min_z)

# --- 4. Add Vacuum to Define Final Simulation Box ---
print("3. Defining final simulation box with vacuum...")
cell = stack.get_cell()
cell[2, 2] += VAC
stack.set_cell(cell, scale_atoms=False)
stack.wrap()

# --- 5. Build and Position Oxygen Supercell ---
print("4. Creating and positioning oxygen block...")
o_bulk = bulk('O', 'fcc', a=O_CELL_CONST, cubic=True)
o_super = o_bulk.repeat((O_REP_XY, O_REP_XY, O_REP_Z))

# CRITICAL FIX: Assign the final simulation cell to the oxygen object
# This must be done BEFORE combining with the '+' operator.
o_super.set_cell(stack.get_cell(), scale_atoms=False)

# SUGGESTION: Use the cleaner, built-in 'center' method for XY
o_super.center(axis=(0, 1), vacuum=0)

# FIX: Robustly move the block to the desired Z center
current_o_center_z = o_super.positions[:, 2].mean()
o_super.positions[:, 2] += (O_Z_CENTER - current_o_center_z)
o_super.center(axis=(0, 1), vacuum=0)

# --- 6. Combine Graphene + Oxygen and Write Data File ---
print("5. Combining systems and writing LAMMPS data file...")
# Now this combination is safe because both objects have the same cell
combined = stack + o_super

# FIX: The final combined system should be wrapped to handle any atoms
# that were shifted outside the periodic boundaries.
combined.wrap()



# CRITICAL FIX: Use 'charge' atom style for ReaxFF
write(
    OUTPUT_FILENAME,
    combined,
    format='lammps-data',
    atom_style='charge',
    specorder=['C', 'O'] # Ensures C=type 1, O=type 2
)

# --- 7. Diagnostics ---
zs = combined.positions[:, 2]
cell_z = combined.get_cell()[2, 2]
print(f"\n✅ Wrote '{OUTPUT_FILENAME}', total atoms = {len(combined)}")
print(f"  • Atom Counts: C={len(stack)}, O={len(o_super)}")
print(f"  • z_min = {zs.min():.3f} Å, z_max = {zs.max():.3f} Å")
print(f"  • cell_z (with vacuum) = {cell_z:.3f} Å")
