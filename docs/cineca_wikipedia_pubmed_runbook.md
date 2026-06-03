# CINECA Wikipedia And PubMed Runbook

This runbook is the copy-paste path for running the full `llm_frag_evaluation` sweep on CINECA for:

- `source_collection_wiki`
- `source_collection_pubmed`

It covers:

1. Model/profile setup.
2. Prompt-load creation.
3. Smoke and full generation submission.
4. Prediction validation.
5. Metric computation.
6. Manual handoff into the LaTeX table.

The commands below assume the repository lives at `/leonardo_work/IscrC_SpecDLM/FRAG` and the model snapshots live under `/leonardo_work/IscrC_SpecDLM/models`.

## Current Campaign State

This is the current manual tracker for the small-model sweep.

### Wikipedia

| Model | Status | Note |
|---|---|---|
| `Qwen2.5-1.5B-Instruct` | Done | 20 experiments completed; 2 prompt errors due to generation size. |
| `Llama-3.2-1B-Instruct` | Running | 20 experiments in progress. |
| `Qwen2.5-3B-Instruct` | Todo | 20 experiments pending. |
| `Llama-3.2-3B-Instruct` | Todo | 20 experiments pending. |
| `medgemma-4b-it` | Todo | 20 experiments pending. |
| `Qwen2.5-7B-Instruct` | Todo | 20 experiments pending. |
| `Mistral-7B-Instruct-v0.3` | Todo | 20 experiments pending. |
| `Qwen2.5-14B-Instruct` | Todo | 20 experiments pending. |
| `Llama-3.1-8B-Instruct` | Todo | 20 experiments pending. |
| `BioMistral-7B` | Todo | 20 experiments pending. |
| `Llama3-OpenBioLLM-8B` | Todo | 20 experiments pending. |
| `PMC_LLaMA_13B` | Todo | 20 experiments pending. |
| `medgemma-27b-it` | Todo | 20 experiments pending. |
| `Qwen2.5-32B-Instruct` | Todo | 20 experiments pending. |

### PubMed

| Model | Status | Note |
|---|---|---|
| `Qwen2.5-1.5B-Instruct` | Blocked | Prompt size issues on PubMed. |
| `Llama-3.2-1B-Instruct` | Todo | 20 experiments pending. |
| `Qwen2.5-3B-Instruct` | Todo | 20 experiments pending. |
| `Llama-3.2-3B-Instruct` | Todo | 20 experiments pending. |
| `medgemma-4b-it` | Todo | 20 experiments pending. |
| `Qwen2.5-7B-Instruct` | Todo | 20 experiments pending. |
| `Mistral-7B-Instruct-v0.3` | Todo | 20 experiments pending. |
| `BioMistral-7B` | Todo | 20 experiments pending. |
| `Llama-3.1-8B-Instruct` | Todo | 20 experiments pending. |
| `Llama3-OpenBioLLM-8B` | Todo | 20 experiments pending. |
| `PMC_LLaMA_13B` | Todo | 20 experiments pending. |
| `Qwen2.5-14B-Instruct` | Todo | 20 experiments pending. |
| `medgemma-27b-it` | Todo | 20 experiments pending. |
| `Qwen2.5-32B-Instruct` | Todo | 20 experiments pending. |

### Operational Note

For the remaining runs, keep the execution order consistent:

1. Create prompt loads.
2. Validate prompt loads.
3. Run one smoke prompt load per model.
4. Run the full 20-experiment matrix.
5. Validate predictions.
6. Compute metrics.
7. Copy the metrics into the LaTeX table manually.

If a model fails because the prompt is too long, reduce the batch size only after a worst-case smoke run succeeds and confirm the `GENERATE_MAX_MODEL_LEN` value in the per-model profile before relaunching the full matrix.

## 0. Base Environment

Use the shared CINECA settings that are already known to work for this project:

```bash
export HPC_ACCOUNT=IscrC_SpecDLM
export HPC_QOS=normal
export HPC_PARTITION=boost_usr_prod
export VLLM_VENV_ACTIVATE=/leonardo_work/IscrC_SpecDLM/FRAG/.venv_frag_vllm/bin/activate
export HF_HOME=/leonardo_work/IscrC_SpecDLM/.cache/huggingface
```

If you need to authenticate with gated Hugging Face models:

```bash
hf auth login
```

If some model snapshots are missing, print the download commands from the canonical model list:

```bash
python llm_frag_evaluation/scripts/print_model_download_commands.py --family qwen
python llm_frag_evaluation/scripts/print_model_download_commands.py --family llama
python llm_frag_evaluation/scripts/print_model_download_commands.py --family mistral
python llm_frag_evaluation/scripts/print_model_download_commands.py --family biomedical
```

## 1. Update The Repo

Run this once at the start of the campaign:

```bash
cd /leonardo_work/IscrC_SpecDLM/FRAG
git status --short
git pull --ff-only
source .venv_frag_vllm/bin/activate
```

## 2. Wikipedia Campaign

The Wikipedia collection uses the files under `llm_frag_evaluation/data/inputs/source_collection_wiki/`.

### 2.1 Write The Per-Model Profiles

Wikipedia profiles should be written before prompt creation:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --write-profiles \
  --overwrite-profiles
```

### 2.2 Create Prompt Loads

Generate the prompt-load command file and run it:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --section prompt-loads > /tmp/frag_create_wiki_prompt_loads.sh

bash /tmp/frag_create_wiki_prompt_loads.sh
```

### 2.3 Validate The Wikipedia Prompt Loads

```bash
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/wiki_config.json \
  --all-input-files
```

### 2.4 Run Smoke Jobs

Generate the smoke command file and run it:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --section smoke > /tmp/frag_smoke_wiki.sh

bash /tmp/frag_smoke_wiki.sh
```

### 2.5 Run The Full Wikipedia Matrix

Generate the full submit script and run it:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --section submit > /tmp/frag_submit_wiki.sh

bash /tmp/frag_submit_wiki.sh
```

Monitor the jobs:

```bash
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %R"
```

Inspect a job log if needed:

```bash
cat llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.out
cat llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.err
```

### 2.6 Validate The Wikipedia Predictions

This checks that every generated prediction file exists and has a valid answer choice.

```bash
find llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki -name prompts.jsonl | sort > /tmp/frag_wiki_prompt_loads.txt

while IFS= read -r prompt_load; do
  out_dir="${prompt_load%/prompts.jsonl}"
  python llm_frag_evaluation/scripts/validate_predictions.py \
    --prompt-load "$prompt_load" \
    > "${out_dir}/validation_report.json"
done < /tmp/frag_wiki_prompt_loads.txt
```

### 2.7 Evaluate The Wikipedia Predictions

This writes one `metrics.json` file per run, which you can later copy into the paper tables.

```bash
wiki_input_file() {
  dataset="$1"
  retriever="$2"

  case "${dataset}:${retriever}" in
    mmlu:bm25) echo "source_collection_wiki/cache_step2_mmlu_scored_bm25.json" ;;
    mmlu:contriever) echo "source_collection_wiki/cache_step2_mmlu_scored_contriever.json" ;;
    medqa:bm25) echo "source_collection_wiki/cache_step2_medqa_scored_bm25.json" ;;
    medqa:contriever) echo "source_collection_wiki/cache_step2_medqa_scored_contriever.json" ;;
    pubmedqa:bm25) echo "source_collection_wiki/cache_step2_pubmedqa_scored_bm25.json" ;;
    pubmedqa:contriever) echo "source_collection_wiki/cache_step2_pubmedqa_scored_contriever.json" ;;
    bioasq:bm25) echo "source_collection_wiki/cache_step2_bioasq_scored_bm25 (1).json" ;;
    bioasq:contriever) echo "source_collection_wiki/cache_step2_bioasq_scored_contriever (1).json" ;;
    *) return 1 ;;
  esac
}

while IFS= read -r prompt_load; do
  rel="${prompt_load#llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki/}"
  dataset="${rel%%/*}"
  rest="${rel#*/}"
  retriever="${rest%%/*}"
  rest="${rest#*/}"
  experiment="${rest%%/*}"
  llm_dir="${rest#*/}"
  llm="${llm_dir%/prompts.jsonl}"
  out_dir="${prompt_load%/prompts.jsonl}"
  input_file="$(wiki_input_file "$dataset" "$retriever")"

  python llm_frag_evaluation/scripts/evaluate_predictions.py \
    --config llm_frag_evaluation/configs/wiki_config.json \
    --input-file "$input_file" \
    --dataset "$dataset" \
    --retriever "$retriever" \
    --experiment "$experiment" \
    --llm "$llm" \
    > "${out_dir}/metrics.json"
done < /tmp/frag_wiki_prompt_loads.txt
```

## 3. PubMed Campaign

The PubMed collection uses the files under `llm_frag_evaluation/data/inputs/source_collection_pubmed/`.

### 3.1 Rewrite The Per-Model Profiles

PubMed needs a separate profile pass because the context window is larger:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_pubmed \
  --write-profiles \
  --overwrite-profiles
```

### 3.2 Create PubMed Prompt Loads

Generate the PubMed prompt-load command file and run it:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_pubmed \
  --section prompt-loads > /tmp/frag_create_pubmed_prompt_loads.sh

bash /tmp/frag_create_pubmed_prompt_loads.sh
```

### 3.3 Validate The PubMed Prompt Loads

```bash
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files
```

### 3.4 Run PubMed Smoke Jobs

Generate the PubMed smoke command file and run it:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_pubmed \
  --section smoke > /tmp/frag_smoke_pubmed.sh

bash /tmp/frag_smoke_pubmed.sh
```

### 3.5 Run The Full PubMed Matrix

Generate the full PubMed submit script and run it:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_pubmed \
  --section submit > /tmp/frag_submit_pubmed.sh

bash /tmp/frag_submit_pubmed.sh
```

Monitor the jobs:

```bash
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %R"
```

Inspect a job log if needed:

```bash
cat llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.out
cat llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.err
```

### 3.6 Validate The PubMed Predictions

```bash
find llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed -name prompts.jsonl | sort > /tmp/frag_pubmed_prompt_loads.txt

while IFS= read -r prompt_load; do
  out_dir="${prompt_load%/prompts.jsonl}"
  python llm_frag_evaluation/scripts/validate_predictions.py \
    --prompt-load "$prompt_load" \
    > "${out_dir}/validation_report.json"
done < /tmp/frag_pubmed_prompt_loads.txt
```

### 3.7 Evaluate The PubMed Predictions

This writes one `metrics.json` file per run.

```bash
pubmed_input_file() {
  dataset="$1"
  retriever="$2"

  case "$retriever" in
    bm25) retriever_tag="bm25" ;;
    contriever) retriever_tag="Contriever" ;;
    *) return 1 ;;
  esac

  echo "source_collection_pubmed/cache_step2_${dataset}_scored_PubMed_${retriever_tag}.json"
}

while IFS= read -r prompt_load; do
  rel="${prompt_load#llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/}"
  dataset="${rel%%/*}"
  rest="${rel#*/}"
  retriever="${rest%%/*}"
  rest="${rest#*/}"
  experiment="${rest%%/*}"
  llm_dir="${rest#*/}"
  llm="${llm_dir%/prompts.jsonl}"
  out_dir="${prompt_load%/prompts.jsonl}"
  input_file="$(pubmed_input_file "$dataset" "$retriever")"

  python llm_frag_evaluation/scripts/evaluate_predictions.py \
    --config llm_frag_evaluation/configs/pubmed_config.json \
    --input-file "$input_file" \
    --dataset "$dataset" \
    --retriever "$retriever" \
    --experiment "$experiment" \
    --llm "$llm" \
    > "${out_dir}/metrics.json"
done < /tmp/frag_pubmed_prompt_loads.txt
```

## 4. Manual LaTeX Update

The result files you need for the paper are the `metrics.json` files written next to each prediction directory. The directory pattern is:

```text
llm_frag_evaluation/outputs/predictions/<source_collection>/<dataset>/<retriever>/<experiment>/<llm>/metrics.json
```

Example:

```bash
cat llm_frag_evaluation/outputs/predictions/source_collection_wiki/medqa/bm25/frag/<MODEL>/metrics.json
```

For the final tables, copy the `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `n`, and `missing_predictions` values into:

```text
llm_frag_evaluation/RESULTS_TABLE.md
```

If you are updating the LaTeX manually, keep the missing-prediction policy unchanged: missing or invalid predictions count as incorrect.
