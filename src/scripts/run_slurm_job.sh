#!/bin/bash
#SBATCH --job-name=ReProSeg_NNI_HPO
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/%u/logs/slurm-%j.out

cd "${SLURM_SUBMIT_DIR:-${PWD}}"
uv run python src/scripts/run.py "$@"