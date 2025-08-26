#!/bin/bash
#SBATCH -p v6_384
#SBATCH -N 1
#SBATCH -n 64
source  /public1/soft/modules/module.sh 
module load lammps/oneAPI.2022.1/patch_27Jun2024-para
mpirun -np 64 lmp_intel_cpu_intelmpi  -i in.gra_o
