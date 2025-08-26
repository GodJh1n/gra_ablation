#!/bin/bash
#SBATCH -N 4  
#SBATCH -n 169     
#SBATCH -p CLUSTER   
#SBATCH -t 1000:00:00  
#SBATCH -J JHY

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

srun hostname -s | sort -n > slurm.hosts

mpirun -np 169 -machinefile slurm.hosts /share/apps/lammps/lammps-2Aug2023/src/lmp_mpi -in in.gra_o > out