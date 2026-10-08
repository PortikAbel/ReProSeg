#!/bin/bash
# Submit an interpretability-metric evaluation job for a trained checkpoint,
# logging into the same directory as the training run (so e.g. the consistency
# score lands alongside that run's TensorBoard hparams):
#   src/scripts/evaluate.sh <run_dir> [extra hydra overrides...]
#   src/scripts/evaluate.sh <run_dir> evaluate.consistency.quantile=0.7 data=pascal_voc
# <run_dir> is a training run's log directory (contains checkpoints/).
# Defaults to the best-mIoU checkpoint; override with model.checkpoint=<path>.
# Pass the same data=<dataset>/model=<...> overrides used for training if they
# weren't the defaults.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source src/scripts/_lib.sh

run_dir="${1:?Usage: evaluate.sh <run_dir> [hydra overrides...]}"
shift

sbatch_submit --job-name=ReProSeg_evaluate --gres=gpu:1 src/scripts/_submit.sh \
  -m evaluate "model.checkpoint=${run_dir}/checkpoints/net_trained_best_miou" "logging.path=${run_dir}" "$@"
