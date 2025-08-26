# make_o2_mol.py
with open("o2.mol", "w") as f:
    f.write("# O2 molecule\n\n")
    f.write("2 atoms\n1 bonds\n\n")
    f.write("Coords\n")
    f.write("0.0 0.0 -0.604\n")   # O1  (Å)
    f.write("0.0 0.0  0.604\n")   # O2
    f.write("\nTypes\n")
    f.write("2 2\n")              # 两个原子均为 type 2 (O)
    f.write("\nBonds\n")
    f.write("1 1 1 2\n")          # bond‑id  bond‑type  atom1 atom2
print("o2.mol generated")
