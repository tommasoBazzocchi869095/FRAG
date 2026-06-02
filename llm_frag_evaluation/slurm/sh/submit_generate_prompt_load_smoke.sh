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
  echo "Usage: $0 <prompt_load_path>" >&2
  exit 2
fi

"${SCRIPT_DIR}/submit_generate_prompt_load.sh" "$1" "${SMOKE_PROMPT_LIMIT:-5}"
