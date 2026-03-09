#!/usr/bin/env bash

: "${NNI_OUTPUT_DIR:?NNI_OUTPUT_DIR is not set (NNI should set it)}"
: "${NNI_SYS_DIR:?NNI_SYS_DIR is not set (NNI should set it)}"
: "${NNI_TRIAL_JOB_ID:?NNI_TRIAL_JOB_ID is not set (NNI should set it)}"

EXPORT_VARS="ALL,NNI_OUTPUT_DIR=${NNI_OUTPUT_DIR},NNI_SYS_DIR=${NNI_SYS_DIR},NNI_TRIAL_JOB_ID=${NNI_TRIAL_JOB_ID}"

echo "Submitting Slurm job for NNI trial..."
echo "Exported variables: ${EXPORT_VARS}"

jobid=""
submit_attempt=1
while (( submit_attempt <= SBATCH_RETRIES )); do
  submit_output="$(sbatch --parsable --job-name="ReProSeg_NNI_${NNI_TRIAL_JOB_ID}" --export="${EXPORT_VARS}" --output="${NNI_OUTPUT_DIR}/slurm-%j.log" src/scripts/start_slurm_job.sh 2>&1)"
  submit_rc=$?

  if [[ ${submit_rc} -eq 0 ]]; then
    parsed_jobid="${submit_output%%;*}"
    if [[ "${parsed_jobid}" =~ ^[0-9]+$ ]]; then
      jobid="${parsed_jobid}"
      break
    fi
    echo "ERROR: sbatch returned success but job id is invalid: ${submit_output}"
  else
    echo "WARNING: sbatch attempt ${submit_attempt}/${SBATCH_RETRIES} failed: ${submit_output}"
  fi

  if (( submit_attempt < SBATCH_RETRIES )); then
    echo "Retrying sbatch in ${SBATCH_RETRY_DELAY_SEC}s..."
    sleep "${SBATCH_RETRY_DELAY_SEC}"
  fi
  ((submit_attempt++))
done

if [[ -z "${jobid}" ]]; then
  echo "ERROR: failed to submit Slurm job after ${SBATCH_RETRIES} attempts"
  exit 1
fi

cleanup_on_signal() {
  if [[ -n "${jobid}" ]]; then
    echo "Received termination signal, cancelling Slurm JobID: ${jobid}"
    scancel "${jobid}" >/dev/null 2>&1 || true
  fi
  exit 143
}

trap cleanup_on_signal INT TERM

echo "Submitted Slurm JobID: ${jobid}"

# Wait until the Slurm job finishes
while [[ -n "$(squeue -j "${jobid}" -h -o '%i' 2>/dev/null)" ]]; do
  sleep 1
done