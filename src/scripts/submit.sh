#!/bin/bash
#SBATCH --job-name=ReProSeg
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/%u/logs/slurm-%j.out
#
# Generic Slurm job: forwards everything after the script name to `uv run python`.
# Submit directly for one-off runs, e.g.:
#   sbatch src/scripts/submit.sh -m evaluate model.checkpoint=<path> evaluate=consistency
# Override resources on the sbatch command line (later flags win), e.g.:
#   sbatch --gres=gpu:0 --mem=8G src/scripts/submit.sh -m evaluate ...
# Prefer the train.sh / visualize.sh / evaluate.sh wrappers for the common cases.

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
uv run python "$@"
