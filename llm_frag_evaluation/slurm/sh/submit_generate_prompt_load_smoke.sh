#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_ALIAS=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-alias)
      MODEL_ALIAS="${2:-}"
      shift 2
      ;;
    --model-alias=*)
      MODEL_ALIAS="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--model-alias <alias>] <prompt_load_path>"
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}"

if [[ -n "${MODEL_ALIAS}" ]]; then
  PRIVATE_ENV="${PROJECT_ROOT}/llm_frag_evaluation/slurm/model_profiles/${MODEL_ALIAS}.env"
else
  PRIVATE_ENV="${HPC_PRIVATE_ENV:-${PROJECT_ROOT}/llm_frag_evaluation/slurm/hpc.private.env}"
fi

if [[ ! -f "${PRIVATE_ENV}" ]]; then
  echo "Missing private env file: ${PRIVATE_ENV}" >&2
  echo "Create model profiles with plan_model_sweep.py --write-profiles or copy hpc.private.env.example." >&2
  exit 2
fi

source "${PRIVATE_ENV}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--model-alias <alias>] <prompt_load_path>" >&2
  exit 2
fi

SUBMIT_ARGS=()
if [[ -n "${MODEL_ALIAS}" ]]; then
  SUBMIT_ARGS+=(--model-alias "${MODEL_ALIAS}")
fi

bash "${SCRIPT_DIR}/submit_generate_prompt_load.sh" "${SUBMIT_ARGS[@]}" "$1" "${SMOKE_PROMPT_LIMIT:-5}"
