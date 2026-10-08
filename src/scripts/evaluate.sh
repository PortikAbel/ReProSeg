#!/bin/bash
# Submit an interpretability-metric evaluation job for a trained checkpoint:
#   src/scripts/evaluate.sh <checkpoint> [hydra overrides...]
#   src/scripts/evaluate.sh <checkpoint> evaluate=consistency data=pascal_voc
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

checkpoint="${1:?Usage: evaluate.sh <checkpoint> [hydra overrides...]}"
shift

sbatch --job-name=ReProSeg_evaluate --gres=gpu:1 src/scripts/_submit.sh \
  -m evaluate "model.checkpoint=${checkpoint}" "$@"
