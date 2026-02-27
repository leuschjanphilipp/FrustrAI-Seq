#!/bin/bash
### SLURM submit

#SBATCH -J "it6_train_debug"
#SBATCH -o logs/it6_train_debug_%j.out
#SBATCH -e logs/it6_train_debug_%j.err
#SBATCH -t 24:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

#source ~/.bash_profile
#source /lustre/groups/shared/hei/leusch.jan/FrustrAI-Seq/venv/bin/activate

cd /lustre/groups/shared/hei/leusch.jan/FrustrAI-Seq
source .venv/bin/activate
wandb login
mkdir -p logs

srun --nice=10000 python src/frustraiseq/train/train.py --experiment_name it6_train_debug --fit_dataset data/v10_Funstration_Dataset.parquet.gzip --plm_model data/prot_t5_xl_half_uniref50-enc --split_key split_0 --batch_size 16 --num_workers 10 --cath_sampling_n 10

