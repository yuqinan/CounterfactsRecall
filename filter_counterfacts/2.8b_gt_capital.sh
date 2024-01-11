#!/bin/sh
#$ -cwd


#SBATCH --job-name=2.8b_gt_capital
#SBATCH --time=24:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=60G

# Ensures all allocated cores are on the same node
#SBATCH -N 1
#SBATCH --mail-type=END
#SBATCH --mail-user=qinan_yu@brown.edu

#SBATCH -o /gpfs/data/epavlick/overwrite/2.8b_gt_capital.txt
#SBATCH -e /gpfs/data/epavlick/overwrite/2.8b_gt_capital.txt


# load the limber2 environment
source /gpfs/data/epavlick/limber2/myenvlimber/bin/activate
module load python/3.7.4

python ../multiple_scale.py --model pythia-2.8b --scale1 2.6 --scale2 -1.5