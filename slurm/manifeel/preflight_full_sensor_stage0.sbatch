#!/usr/bin/env bash
#SBATCH --job-name=full_sensor_stage0_batch8_preflight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
cd /public/home/wangzihao/projects/AdaViP
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
echo "CPU affinity: $(taskset -pc $$); batch=8; train_workers=2; val_workers=0"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
run_dir="/data/wangzihao/outputs/manifeel/full_sensor_stage0_batch8_preflight_${SLURM_JOB_ID}"
bash scripts/manifeel/train_full_sensor_reference.sh \
    "output_dir=${run_dir}" training.num_epochs=1 training.max_train_steps=2 \
    training.max_val_steps=1 dataloader.num_workers=2 val_dataloader.num_workers=0 \
    training.checkpoint_every=1
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
bash scripts/manifeel/train_full_sensor_reference.sh \
    "output_dir=${run_dir}" "resume=${run_dir}/checkpoints/latest.ckpt" \
    training.num_epochs=1 training.max_train_steps=2 training.max_val_steps=1 \
    dataloader.num_workers=2 val_dataloader.num_workers=0 training.checkpoint_every=1
