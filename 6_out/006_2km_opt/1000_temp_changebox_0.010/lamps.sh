#!/bin/bash
#SBATCH -N 1  
#SBATCH -n 64     
#SBATCH -w compute-0-12   
#SBATCH -p CLUSTER   
#SBATCH -t 1000:00:00  
#SBATCH -J JHY

source ~/.bashrc

cd $SLURM_SUBMIT_DIR

srun hostname -s | sort -n > slurm.hosts

mpirun -np 64 -machinefile slurm.hosts /share/apps/lammps/lammps-2Aug2023/src/lmp_mpi -in in.gra_o > out