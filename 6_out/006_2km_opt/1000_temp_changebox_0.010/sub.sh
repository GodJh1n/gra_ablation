#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 64
source  /public1/soft/modules/module.sh 
module load mpi/oneAPI/2022.1 gcc/10.2.0
export PATH=/public1/home/sch4430/soft/lammps-patch_27Jun2024/src:$PATH
mpirun -np 64 lmp_intel_cpu_intelmpi  -i in.gra_o
