#!/bin/bash
# Submit a visualization job for a trained checkpoint, logging into the same
# directory as the training run (so e.g. cached top-k activations and
# TensorBoard hparams are shared with it):
#   src/scripts/visualize.sh <run_dir> [extra hydra overrides...]
# <run_dir> is a training run's log directory (contains checkpoints/).
# Defaults to the best-mIoU checkpoint; override with model.checkpoint=<path>.
# Pass the same data=<dataset>/model=<...> overrides used for training if they
# weren't the defaults.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source src/scripts/_lib.sh

run_dir="${1:?Usage: visualize.sh <run_dir> [hydra overrides...]}"
shift

sbatch_submit --job-name=ReProSeg_visualize --gres=gpu:1 src/scripts/_submit.sh \
  -m visualize "model.checkpoint=${run_dir}/checkpoints/net_trained_best_miou" "logging.path=${run_dir}" "$@"
