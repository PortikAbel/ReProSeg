#!/usr/bin/env bash

: "${NNI_OUTPUT_DIR:?NNI_OUTPUT_DIR is not set (NNI should set it)}"
: "${NNI_SYS_DIR:?NNI_SYS_DIR is not set (NNI should set it)}"
: "${NNI_TRIAL_JOB_ID:?NNI_TRIAL_JOB_ID is not set (NNI should set it)}"

EXPORT_VARS="ALL,NNI_OUTPUT_DIR=${NNI_OUTPUT_DIR},NNI_SYS_DIR=${NNI_SYS_DIR},NNI_TRIAL_JOB_ID=${NNI_TRIAL_JOB_ID}"

echo "Submitting Slurm job for NNI trial..."
echo "Exported variables: ${EXPORT_VARS}"

jobid="$(sbatch --parsable --export="${EXPORT_VARS}" --output="${NNI_OUTPUT_DIR}/slurm-%j.log" src/scripts/start_slurm_job.sh)"

echo "Submitted Slurm JobID: ${jobid}"

# Wait until the Slurm job finishes
while [[ -n "$(squeue -j "${jobid}" -h -o '%i' 2>/dev/null)" ]]; do
  sleep 10
done