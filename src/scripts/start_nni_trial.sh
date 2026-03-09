#!/usr/bin/env bash

: "${NNI_OUTPUT_DIR:?NNI_OUTPUT_DIR is not set (NNI should set it)}"
: "${NNI_SYS_DIR:?NNI_SYS_DIR is not set (NNI should set it)}"
: "${NNI_TRIAL_JOB_ID:?NNI_TRIAL_JOB_ID is not set (NNI should set it)}"

set -u
set -o pipefail

EXPORT_VARS="ALL,NNI_OUTPUT_DIR=${NNI_OUTPUT_DIR},NNI_SYS_DIR=${NNI_SYS_DIR},NNI_TRIAL_JOB_ID=${NNI_TRIAL_JOB_ID}"
SBATCH_RETRIES="${SBATCH_RETRIES:-5}"
SBATCH_RETRY_DELAY_SEC="${SBATCH_RETRY_DELAY_SEC:-3}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-2}"
STATUS_QUERY_RETRIES="${STATUS_QUERY_RETRIES:-5}"

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

status_query_failures=0
while true; do
  queue_state_output="$(squeue -j "${jobid}" -h -o '%T' 2>&1)"
  queue_rc=$?

  if [[ ${queue_rc} -eq 0 ]]; then
    if [[ -n "${queue_state_output}" ]]; then
      status_query_failures=0
      queue_state="$(echo "${queue_state_output}" | head -n1 | xargs)"
      case "${queue_state}" in
        PENDING|CONFIGURING|RUNNING|COMPLETING|STAGE_OUT|SUSPENDED)
          sleep "${POLL_INTERVAL_SEC}"
          continue
          ;;
      esac
    fi
  else
    echo "WARNING: squeue query failed for job ${jobid}: ${queue_state_output}"
  fi

  job_info_output="$(scontrol show jobid "${jobid}" 2>&1)"
  job_info_rc=$?
  if [[ ${job_info_rc} -ne 0 ]]; then
    ((status_query_failures++))
    echo "WARNING: scontrol query failed for job ${jobid} (${status_query_failures}/${STATUS_QUERY_RETRIES}): ${job_info_output}"
    if (( status_query_failures >= STATUS_QUERY_RETRIES )); then
      echo "ERROR: unable to determine terminal state for Slurm job ${jobid}"
      exit 1
    fi
    sleep "${POLL_INTERVAL_SEC}"
    continue
  fi

  status_query_failures=0
  job_state="$(awk -F'JobState=' 'NF>1 {print $2; exit}' <<< "${job_info_output}" | awk '{print $1}')"

  case "${job_state}" in
    PENDING|CONFIGURING|RUNNING|COMPLETING|STAGE_OUT|SUSPENDED)
      sleep "${POLL_INTERVAL_SEC}"
      ;;
    COMPLETED)
      echo "Slurm JobID ${jobid} completed successfully"
      exit 0
      ;;
    CANCELLED|CANCELLED+|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
      echo "Slurm JobID ${jobid} finished with state ${job_state}"
      exit 1
      ;;
    "")
      echo "WARNING: empty Slurm state for job ${jobid}, continuing to poll"
      sleep "${POLL_INTERVAL_SEC}"
      ;;
    *)
      echo "WARNING: unknown Slurm state '${job_state}' for job ${jobid}, continuing to poll"
      sleep "${POLL_INTERVAL_SEC}"
      ;;
  esac
done