#!/bin/bash
# Shared helpers for the sbatch wrapper scripts.

# `--nodelist` has no SBATCH_* environment variable equivalent, so it must be
# passed explicitly on every sbatch invocation; wrap sbatch here to avoid repeating that.
sbatch_submit() {
  sbatch ${SLURM_NODE_LIST:+--nodelist="${SLURM_NODE_LIST}"} "$@"
}
