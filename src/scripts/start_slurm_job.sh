#!/bin/bash

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
script_dir="$(dirname "${script_path}")"
repo_root="$(cd "${script_dir}/../.." && pwd)"
job_script="${script_dir}/run_slurm_job.sh"

sbatch_args=(--chdir="${repo_root}")

if [[ -n "${SLURM_NODE_LIST:-}" ]]; then
	sbatch_args+=(--nodelist="${SLURM_NODE_LIST}")
fi

exec sbatch "${sbatch_args[@]}" "$@" "${job_script}"