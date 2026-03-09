#!/bin/bash
#SBATCH --job-name=ReProSeg_NNI_HPO
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

cd "${SLURM_SUBMIT_DIR}"
uv run python src/scripts/run.py