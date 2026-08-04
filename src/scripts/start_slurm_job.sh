#!/bin/bash
#SBATCH --job-name=ReProSeg_NNI_HPO
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/%u/logs/slurm-%j.out

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
# uv run python src/scripts/run.py "$@"

# uv run python src/scripts/run.py training.skip_training=true +evaluation.consistency_score.calculate=true +evaluation.consistency_score.threshold=0.8

# for q in 0.7 0.8 0.9; do
#   uv run python -m visualize.consistency \
#     /home/annamari/ProtoSeg-checkpoints/cityscapes_kld_imnet_4_16v2/checkpoints/push_best.pth \
#     --data-path /data/datasets/Cityscapes \
#     --quantile "$q" \
#     --threshold 0.8 \
#     --output-dir consistency_results_protoseg
# done

for q in 0.7 0.8 0.9; do
  uv run python -m visualize.consistency \
    /home/annamari/ReProSeg-checkpoints/net_trained_best_miou \
    --data-path /data/datasets/Cityscapes \
    --quantile "$q" \
    --threshold 0.8 \
    --output-dir consistency_results_reproseg
done
