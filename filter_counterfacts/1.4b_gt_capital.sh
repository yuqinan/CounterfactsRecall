#!/bin/sh
#$ -cwd


#SBATCH --job-name=1.4b_gt_capital
#SBATCH --time=24:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --mem=60G

# Ensures all allocated cores are on the same node
#SBATCH -N 1
#SBATCH --mail-type=END
#SBATCH --mail-user=qinan_yu@brown.edu

#SBATCH -o /gpfs/data/epavlick/overwrite/1.4b_gt_capital.txt
#SBATCH -e /gpfs/data/epavlick/overwrite/1.4b_gt_capital.txt


# load the limber2 environment
source /gpfs/data/epavlick/limber2/myenvlimber/bin/activate
module load python/3.7.4

python ../multiple_scale.py --model pythia-1.4b --scale1 3.5 --scale2 -6.0 