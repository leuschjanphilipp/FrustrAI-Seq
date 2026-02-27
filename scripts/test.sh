#!/bin/bash
### SLURM submit

#SBATCH -J "it6_test_debug"
#SBATCH -o logs/it6_test_debug_%j.out
#SBATCH -e logs/it6_test_debug_%j.err
#SBATCH -t 02:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

#source ~/.bash_profile
#source /lustre/groups/shared/hei/leusch.jan/FrustrAI-Seq/venv/bin/activate

cd /lustre/groups/shared/hei/leusch.jan/FrustrAI-Seq
source .venv/bin/activate
wandb login
mkdir -p logs

srun --nice=10000 python src/frustraiseq/eval/test.py --config ./it6_train_debug/config.yaml