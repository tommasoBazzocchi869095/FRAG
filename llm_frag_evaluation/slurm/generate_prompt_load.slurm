#!/bin/bash
#SBATCH --job-name=frag-vllm
#SBATCH --output=llm_frag_evaluation/outputs/logs/%x_%j.out
#SBATCH --error=llm_frag_evaluation/outputs/logs/%x_%j.err

set -euo pipefail

if [[ -z "${PROMPT_LOAD_PATH:-}" ]]; then
  echo "PROMPT_LOAD_PATH is required" >&2
  exit 2
fi

if [[ -z "${HPC_MODEL_PATH:-}" ]]; then
  echo "HPC_MODEL_PATH is required" >&2
  exit 2
fi

eval "${ENV_INIT_COMMAND:-module purge; module load python/3.11.7}"

if [[ -n "${VLLM_VENV_ACTIVATE:-}" ]]; then
  source "${VLLM_VENV_ACTIVATE}"
fi

cd "${SLURM_SUBMIT_DIR}"

LIMIT_ARGS=()
if [[ -n "${PROMPT_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${PROMPT_LIMIT}")
fi

python llm_frag_evaluation/scripts/run_vllm_prompt_load.py \
  --prompt-load "${PROMPT_LOAD_PATH}" \
  --model-path "${HPC_MODEL_PATH}" \
  --tensor-parallel-size "${GENERATE_TENSOR_PARALLEL_SIZE:-4}" \
  --gpu-memory-utilization "${GENERATE_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${GENERATE_MAX_MODEL_LEN:-4096}" \
  --max-tokens "${GENERATE_MAX_TOKENS:-512}" \
  --batch-size "${GENERATE_BATCH_SIZE:-5}" \
  "${LIMIT_ARGS[@]}"
