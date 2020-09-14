#!/bin/bash
#SBATCH -n 2                    
#SBATCH --mem=20000
#SBATCH --gres=gpu:1
#SBATCH --job-name=ab_ce_nt
#SBATCH --mail-user=bhanugarg05@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=slurm_%j.out

#SBATCH -A research
#SBATCH -p long
#SBATCH -t 03-15:00:00
#SBATCH --export=ALL

module add cuda/10.0
module add cuda/8.0
module add cuda/9.0
module add cuda/9.1
module add cudnn/6-cuda-8.0
module add cudnn/7-cuda-8.0
module add cudnn/7-cuda-9.0


source ~/env/bin/activate
i=0
cd $i
cp ../s.py .
python3 s.py  -lr  0.001 -h_l   6 7 8 9 10  -n $i
cd ..
i=3
cd $i
cp ../s.py .
python3 s.py  -lr  0.001 -h_l   6 7 8 9 10  -n $i
cd ..
