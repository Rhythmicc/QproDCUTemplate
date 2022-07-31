#!/bin/bash
#SBATCH -J QproDCUTemplate
#SBATCH -p your queue name
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -x
#SBATCH --gres=dcu:1
#SBATCH -o log/%j.loop
#SBATCH -e log/%j.loop

srun dist/QproDCUTemplate
