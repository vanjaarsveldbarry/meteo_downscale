#!/bin/bash
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --time=72:00:00
#SBATCH -o /eejit/home/7006713/projects/1km_forcing/run_ds.out

source ${HOME}/.bashrc
mamba activate meteo_ds

taskset -c 0-95 python /eejit/home/7006713/projects/1km_forcing/downscale_forcing.py