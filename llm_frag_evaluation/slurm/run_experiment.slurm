#!/bin/bash
#SBATCH --job-name=frag-rerank
#SBATCH --output=llm_frag_evaluation/outputs/logs/%x_%j.out
#SBATCH --error=llm_frag_evaluation/outputs/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

# TODO: set the correct CINECA account and partition for the allocation.
##SBATCH --account=<account>
##SBATCH --partition=<partition>

# TODO: load the required modules for the selected cluster.
# module purge
# module load python
# module load cuda

# TODO: activate the project environment.
# source /path/to/venv/bin/activate

cd "${SLURM_SUBMIT_DIR}"

python llm_frag_evaluation/scripts/run_experiment.py \
  --config llm_frag_evaluation/configs/default_config.json \
  --dry-run
