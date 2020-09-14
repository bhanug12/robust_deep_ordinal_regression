#!/bin/bash
#SBATCH -n 5                    
#SBATCH --mem=25000
#SBATCH --gres=gpu:1
#SBATCH --job-name=syn_l
#SBATCH --mail-user=bhanugarg05@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_%j.out

#SBATCH -A research
#SBATCH -p long
#SBATCH -t 03-15:00:00
#SBATCH --export=ALL

module add cuda/10.0
module add cudnn/7-cuda-8.0
module add cudnn/7-cuda-9.0

source ~/env/bin/activate

python3  lhat.py  -lr  0.001 -h_l 10  -n 3
#python3  l.py  -lr  0.001 -h_l 10  -n 3 
 
#python3  l.py  -lr  0.001 -h_l 1 -n 1 
#python3  l.py  -lr  0.001 -h_l 1 -n 2 
#python3  l.py  -lr  0.001 -h_l 1  -n 3  
#python3  l.py  -lr  0.001 -h_l 1  -n 4  
#python3  l.py  -lr  0.001 -h_l 1 -n 5  
#python3  l.py  -lr  0.001 -h_l 1 -n 6 
#
#python3  lhat.py  -lr  0.001 -h_l 1 -n 1 
#python3  lhat.py  -lr  0.001 -h_l 1 -n 2  
#python3  lhat.py  -lr  0.001 -h_l 1  -n 3 
#python3 lhat.py  -lr  0.001 -h_l 9  -n 4 
#python3 lhat.py  -lr  0.001 -h_l 4 -n 5 
#python3 lhat.py  -lr  0.001 -h_l 2 -n 6 
#
#
#
#python3 lhes.py  -lr  0.001 -h_l 9  -n 0 
#python3 lhes.py  -lr  0.001 -h_l 9 -n 1 
#python3 lhes.py  -lr  0.01 -h_l 4 -n 2 
#python3 lhes.py  -lr  0.001 -h_l 5  -n 3 
#python3 lhes.py  -lr  0.001 -h_l 9  -n 4 
#python3 lhes.py  -lr  0.001 -h_l 4 -n 5 
#python3 lhes.py  -lr  0.001 -h_l 2 -n 6 
