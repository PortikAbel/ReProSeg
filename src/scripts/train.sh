#!/bin/bash
# Submit a training job: src/scripts/train.sh [hydra overrides...]
#   src/scripts/train.sh training=fast data=cityscapes
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

sbatch --job-name=ReProSeg_train --gres=gpu:1 src/scripts/submit.sh -m train "$@"
