#!/bin/bash
# Submit a visualization job that reuses an existing run's exact config:
#   src/scripts/visualize.sh <run_dir> [extra hydra overrides...]
# <run_dir> is a training run's log directory (contains .hydra/ and checkpoints/).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

run_dir="${1:?Usage: visualize.sh <run_dir> [hydra overrides...]}"
shift

sbatch --job-name=ReProSeg_visualize --gres=gpu:1 src/scripts/_submit.sh \
  -m visualize --config-path="${run_dir}/.hydra" --config-name=config "$@"
