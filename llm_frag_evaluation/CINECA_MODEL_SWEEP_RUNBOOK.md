# CINECA Model Sweep Runbook

This runbook is the reproducible path for the multi-model FRAG/RAG/zero-shot sweep on CINECA.

## Goal

Run the same prompt-load and vLLM generation workflow across all models in `llm_frag_evaluation/configs/model_sweep.json`, without overwriting completed outputs or manually editing Slurm settings per model.

## Key Files

| File | Purpose |
|---|---|
| `llm_frag_evaluation/configs/model_sweep.json` | Canonical model list, Hugging Face IDs, local snapshot paths, and launch profiles. |
| `llm_frag_evaluation/scripts/print_model_download_commands.py` | Prints Hugging Face download commands from the canonical model list. |
| `llm_frag_evaluation/scripts/plan_model_sweep.py` | Prints or writes profile env files, prompt-load commands, smoke commands, and full submit commands. |
| `llm_frag_evaluation/slurm/model_profiles/<alias>.env` | Generated per-model Slurm/vLLM settings. Ignored by Git. |
| `llm_frag_evaluation/slurm/sh/submit_generate_prompt_load_smoke.sh` | Submits a smoke generation job. Use `--model-alias`. |
| `llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh` | Submits a full prompt-load generation job. Use `--model-alias`. |

## CINECA Baseline Settings

Known working settings:

```bash
export HPC_ACCOUNT=IscrC_SpecDLM
export HPC_QOS=normal
export HPC_PARTITION=boost_usr_prod
export VLLM_VENV_ACTIVATE=/leonardo_work/IscrC_SpecDLM/FRAG/.venv_frag_vllm/bin/activate
export HF_HOME=/leonardo_work/IscrC_SpecDLM/.cache/huggingface
```

Do not use `boost_qos_bprod` for this sweep. Its minimum allocation is too large for single-node smoke and generation jobs.

## Step 1: Update The Repo

```bash
cd /leonardo_work/IscrC_SpecDLM/FRAG
git status --short
git pull --ff-only
git log -1 --oneline
source .venv_frag_vllm/bin/activate
```

## Step 2: Verify Model Snapshots

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.load(open("llm_frag_evaluation/configs/model_sweep.json"))
root = Path(config["model_root"])

for model in config["models"]:
    path = root / model["local_dir"]
    print(("OK" if path.is_dir() else "MISSING"), model["alias"], path)
PY
```

If models are missing, print download commands:

```bash
python llm_frag_evaluation/scripts/print_model_download_commands.py --family qwen
python llm_frag_evaluation/scripts/print_model_download_commands.py --family llama
python llm_frag_evaluation/scripts/print_model_download_commands.py --family mistral
python llm_frag_evaluation/scripts/print_model_download_commands.py --family biomedical
```

For gated models, authenticate first:

```bash
hf auth login
```

## Step 3: Generate Per-Model Profiles

Use Wikipedia profiles first because Wikipedia prompts are shorter and validated with `GENERATE_MAX_MODEL_LEN=12288`.

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --write-profiles \
  --overwrite-profiles
```

For PubMed profiles, regenerate later with:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_pubmed \
  --write-profiles \
  --overwrite-profiles
```

## Step 4: Create Prompt Loads

Wikipedia first:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --section prompt-loads > /tmp/frag_create_wiki_prompt_loads.sh

wc -l /tmp/frag_create_wiki_prompt_loads.sh
bash /tmp/frag_create_wiki_prompt_loads.sh
```

Verify:

```bash
find llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki -mindepth 4 -maxdepth 4 -type d -printf '%f\n' | sort -u | wc -l
find llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki -name prompts.jsonl | wc -l
```

Expected for 14 models: `14` model directories and `280` prompt files.

## Step 5: Run Smoke Tests

Start with the smallest model and a short prompt class:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load_smoke.sh \
  --model-alias Qwen2.5-1.5B-Instruct \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki/medqa/bm25/zero_shot/Qwen2.5-1.5B-Instruct/prompts.jsonl
```

Then run a harder retrieval prompt class:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load_smoke.sh \
  --model-alias Qwen2.5-1.5B-Instruct \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki/medqa/contriever/frag/Qwen2.5-1.5B-Instruct/prompts.jsonl
```

Monitor:

```bash
squeue -j <JOB_ID>
cat llm_frag_evaluation/outputs/logs/frag-vllm_<JOB_ID>.out
cat llm_frag_evaluation/outputs/logs/frag-vllm_<JOB_ID>.err
```

Inspect summary:

```bash
cat llm_frag_evaluation/outputs/predictions/source_collection_wiki/medqa/contriever/frag/Qwen2.5-1.5B-Instruct/run_summary.json
```

A clean smoke test has:

```text
preflight_error_count = 0
parsed_record_count = prompt_count
error_record_count = 0
```

## Step 6: Expand Smoke Coverage

Before full jobs, smoke one representative model per profile:

| Profile | Representative | GPUs/job |
|---|---|---:|
| `small_1gpu` | `Qwen2.5-1.5B-Instruct` | 1 |
| `medium_1gpu` | `Qwen2.5-7B-Instruct` or `Mistral-7B-Instruct-v0.3` | 1 |
| `large_2gpu` | `Qwen2.5-32B-Instruct` or `medgemma-27b-it` | 2 |

Generate smoke commands:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --section smoke
```

## Step 7: Launch Full Wikipedia Jobs In Waves

Generate submit commands:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --section submit > /tmp/frag_submit_wiki.sh
```

Launch in waves:

```text
small_1gpu: up to 16 concurrent jobs
medium_1gpu: up to 16 concurrent jobs after smoke validation
large_2gpu: up to 8 concurrent jobs
```

Use `squeue -u $USER` to keep within the 16-GPU target.

## Step 8: Repeat For PubMed

After Wikipedia works:

1. Regenerate profiles with `--collection source_collection_pubmed`.
2. Create PubMed prompt loads.
3. Run worst-prompt PubMed smoke tests first.
4. Launch full PubMed jobs in waves.

PubMed uses longer prompts and starts from `GENERATE_MAX_MODEL_LEN=22528`.

## Reproducibility Rules

1. Never hand-edit prompt-load or prediction output paths.
2. Always use `--model-alias` when submitting generation jobs.
3. Treat `model_sweep.json` as the source of truth for model IDs, aliases, local paths, and profiles.
4. Use `git pull --ff-only` on CINECA to avoid accidental merge commits.
5. Keep `llm_frag_evaluation/slurm/model_profiles/*.env` local and ignored by Git.
6. Record any profile changes that prove necessary back into `model_sweep.json`.
7. Count missing or invalid predictions as incorrect, matching the Llama 3.1 70B baseline policy.
