#!/bin/bash
# Shared helpers for the sbatch wrapper scripts.

# Honors SLURM_NODE_LIST, then SBATCH_NODELIST, as --nodelist.
sbatch_submit() {
  local nodelist="${SLURM_NODE_LIST:-${SBATCH_NODELIST:-}}"
  sbatch ${nodelist:+--nodelist="${nodelist}"} "$@"
}
