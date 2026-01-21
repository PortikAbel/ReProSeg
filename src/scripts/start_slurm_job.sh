#!/bin/bash
#SBATCH --job-name=ReProSeg_NNI_HPO
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

# Use SLURM_SUBMIT_DIR which is set to where sbatch was called from
cd "${SLURM_SUBMIT_DIR}"
echo "Working directory: $(pwd)"
uv run python src/scripts/run.py