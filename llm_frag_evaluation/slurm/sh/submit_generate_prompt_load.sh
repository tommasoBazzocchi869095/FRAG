#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PRIVATE_ENV="${HPC_PRIVATE_ENV:-${PROJECT_ROOT}/llm_frag_evaluation/slurm/hpc.private.env}"

if [[ ! -f "${PRIVATE_ENV}" ]]; then
  echo "Missing private env file: ${PRIVATE_ENV}" >&2
  echo "Copy llm_frag_evaluation/slurm/hpc.private.env.example or use a model-specific env profile." >&2
  exit 2
fi

source "${PRIVATE_ENV}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <prompt_load_path> [prompt_limit]" >&2
  exit 2
fi

PROMPT_LOAD_PATH="$1"
PROMPT_LIMIT="${2:-}"

cd "${PROJECT_ROOT}"
mkdir -p llm_frag_evaluation/outputs/logs

SBATCH_ARGS=(
  --account="${HPC_ACCOUNT}"
  --qos="${HPC_QOS}"
  --partition="${HPC_PARTITION}"
  --time="${GENERATE_TIME_LIMIT:-10:00:00}"
  --cpus-per-task="${GENERATE_CPUS_PER_TASK:-4}"
  --mem="${GENERATE_MEM:-100G}"
  --gres="gpu:${GENERATE_GPUS:-4}"
  --export=ALL,PROMPT_LOAD_PATH="${PROMPT_LOAD_PATH}",PROMPT_LIMIT="${PROMPT_LIMIT}"
)

sbatch "${SBATCH_ARGS[@]}" llm_frag_evaluation/slurm/generate_prompt_load.slurm
