#!/bin/bash
#
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=xhu85@jhu.edu
#SBATCH -o slurm.%N.%J.%u.out
#SBATCH -e slurm.%N.%J.%u.err

conda init

source  ~/.bashrc

conda activate qamodel

python3 distilBert.py

conda deactivate
