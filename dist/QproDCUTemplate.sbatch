#!/bin/bash
#SBATCH -J QproDCUTemplate
#SBATCH -p 你的任务队列名
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -o log/%j.loop
#SBATCH -e log/%j.loop

srun dist/QproDCUTemplate
