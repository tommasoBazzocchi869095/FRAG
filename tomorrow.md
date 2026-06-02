# CINECA `llm_frag_evaluation` Next Steps

Current state as of May 28, 2026:

- Prompt code is aligned with previous MedRAG templates and pushed to GitHub.
- The completed experiments used Wikipedia as the retrieval resource, so their Step 2 inputs are the files under `llm_frag_evaluation/data/inputs/source_collection_wiki/`.
- PubMed-backed reruns should use `llm_frag_evaluation/configs/pubmed_config.json`, which selects the matching files under `llm_frag_evaluation/data/inputs/source_collection_pubmed/` and writes collection-qualified prompt loads/predictions.
- Zero-shot is not retriever-specific; run it only once per dataset using `bm25/zero_shot`.
- Prompt loads should now be 20 files after regeneration, not 24.
- The earlier Wikipedia prompt length report showed 0 prompts over 12288 tokens; that finding does not apply to PubMed-backed RAG/FRAG prompt loads.
- PubMed-backed prompt loads must be tokenized before full generation. Use `llm_frag_evaluation/scripts/report_prompt_lengths.py` for all prompt loads and `llm_frag_evaluation/tests/diagnostics/diagnose_vllm_run.py` for failed or smoke runs.
- PubMed MedQA Contriever FRAG at `GENERATE_MAX_MODEL_LEN=12288` produced 406 `PromptTooLong` records and 1 invalid answer. The longest prompt was 20280 tokens.
- With `GENERATE_MAX_TOKENS=1024` and a 512-token safety buffer, the measured recommendation for PubMed MedQA Contriever FRAG is `GENERATE_MAX_MODEL_LEN=22528`.
- A longest-prompt smoke run for PubMed MedQA Contriever FRAG completed 7/7 predictions with `GENERATE_MAX_MODEL_LEN=22528`, `GENERATE_BATCH_SIZE=1`, and `GENERATE_GPU_MEMORY_UTILIZATION=0.90`.
- The vLLM runner logs individual prompt errors and continues instead of aborting the whole job.
- The PubMed-resource campaign is complete with metrics recorded for MMLU, MedQA, PubMedQA, and BioASQ. MedQA BM25 RAG has 9 missing predictions and BM25 FRAG has 11 missing predictions; MMLU BM25 RAG and BM25 FRAG each have 4 missing predictions; PubMedQA RAG/FRAG has 1 missing or invalid prediction per run. Missing predictions are counted as incorrect.
- First task tomorrow: prepare the next model sweep before launching new generation. Download/check model snapshots, define HPC launch profiles, and create experiment launch commands for the general-purpose, Qwen, and biomedical models listed below.
- Then: use the completed Llama 3.1 70B Wikipedia/PubMed results as the baseline for the next model sweep.

## First Task Tomorrow: Next Model Sweep

Prepare the next generation campaign across model families and scales. The goal is to reuse the existing PubMed/Wikipedia prompt-load workflow while choosing GPU counts, tensor parallel sizes, context windows, and batch sizes appropriate to each model size.

The canonical model list for the sweep is stored in `llm_frag_evaluation/configs/model_sweep.json`. That file includes the Hugging Face URL, repo ID, local CINECA snapshot directory, model alias, and initial GPU profile for each model.

### Models To Download Or Verify

General-purpose Llama instruction models:

```text
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.1-70B-Instruct - Done for all except the mix evidence retrieval with the 2 collections.
```

`Llama-3.1-8B-Instruct` is a useful reference point because it has been used in recent medical RAG work.

Qwen instruction models:

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-32B-Instruct
Qwen/Qwen2.5-72B-Instruct
```

Mistral instruction baseline:

```text
mistralai/Mistral-7B-Instruct-v0.3
```

Medical and biomedical models:

```text
google/medgemma-4b-it
google/medgemma-27b-it
aaditya/Llama3-OpenBioLLM-8B
BioMistral/BioMistral-7B
axiong/PMC_LLaMA_13B
```

Confirm exact Hugging Face IDs and access requirements before downloading. Some models may require license acceptance, authentication, or slightly different repository names.

### Model Links And Initial Profiles

| Model | Hugging Face URL | Initial profile |
|---|---|---|
| `axiong/PMC_LLaMA_13B` | `https://huggingface.co/axiong/PMC_LLaMA_13B` | `medium_1gpu` |
| `aaditya/Llama3-OpenBioLLM-8B` | `https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B` | `medium_1gpu` |
| `google/medgemma-4b-it` | `https://huggingface.co/google/medgemma-4b-it` | `small_1gpu` |
| `Qwen/Qwen2.5-32B-Instruct` | `https://huggingface.co/Qwen/Qwen2.5-32B-Instruct` | `large_2gpu` |
| `Qwen/Qwen2.5-14B-Instruct` | `https://huggingface.co/Qwen/Qwen2.5-14B-Instruct` | `medium_1gpu` |
| `Qwen/Qwen2.5-7B-Instruct` | `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct` | `medium_1gpu` |
| `Qwen/Qwen2.5-3B-Instruct` | `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct` | `small_1gpu` |
| `Qwen/Qwen2.5-1.5B-Instruct` | `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct` | `small_1gpu` |
| `meta-llama/Llama-3.1-8B-Instruct` | `https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct` | `medium_1gpu` |
| `meta-llama/Llama-3.2-3B-Instruct` | `https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct` | `small_1gpu` |
| `meta-llama/Llama-3.2-1B-Instruct` | `https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct` | `small_1gpu` |
| `mistralai/Mistral-7B-Instruct-v0.3` | `https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3` | `medium_1gpu` |
| `BioMistral/BioMistral-7B` | `https://huggingface.co/BioMistral/BioMistral-7B` | `medium_1gpu` |
| `google/medgemma-27b-it` | `https://huggingface.co/google/medgemma-27b-it` | `large_2gpu` |

With 16 A100 64GB GPUs, the first scheduling assumption is:

| Profile | GPUs per job | Max concurrent jobs on 16 GPUs | Intended models |
|---|---:|---:|---|
| `small_1gpu` | 1 | 16 | 1B-4B models |
| `medium_1gpu` | 1 | 16 | 7B-14B models if the long PubMed prompt profile fits |
| `large_2gpu` | 2 | 8 | 27B-32B models with long PubMed prompts |

Do not immediately launch all 16 one-GPU jobs. First run one worst-prompt smoke job per model family and inspect vLLM memory use, prompt errors, and answer-format validity. If the 22528-token PubMed profile fits on one A100 for 7B-14B models, then 16 concurrent one-GPU jobs is a reasonable scheduling target for those models.

### Download Plan

Use local snapshots under the shared CINECA model directory, for example:

```text
/leonardo_work/IscrC_SpecDLM/models/
```

Download each model with `huggingface_hub.snapshot_download`, limiting workers to avoid login-node pressure:

```bash
source .venv_frag_vllm/bin/activate
export HF_TOKEN="..."
export MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
export MODEL_PATH="/leonardo_work/IscrC_SpecDLM/models/Llama-3.1-8B-Instruct"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_ID', local_dir='$MODEL_PATH', token='$HF_TOKEN', max_workers=2)"
```

Repeat with a stable local folder name for each model.

To print the full set of download commands from the canonical model list:

```bash
python llm_frag_evaluation/scripts/print_model_download_commands.py
```

Use `--family qwen`, `--family llama`, `--family biomedical`, or `--alias <model-alias>` to print a smaller batch.
Use `--family mistral` for `mistralai/Mistral-7B-Instruct-v0.3`.

### CINECA Sweep Planning Commands

Use the sweep planner to print the exact profile, prompt-load, smoke-test, and full submission commands:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py --section summary
python llm_frag_evaluation/scripts/plan_model_sweep.py --family qwen --section profiles
python llm_frag_evaluation/scripts/plan_model_sweep.py --family qwen --collection source_collection_pubmed --section prompt-loads
python llm_frag_evaluation/scripts/plan_model_sweep.py --family qwen --collection source_collection_pubmed --section smoke
python llm_frag_evaluation/scripts/plan_model_sweep.py --family qwen --collection source_collection_pubmed --section submit
```

Create model-specific env files under `llm_frag_evaluation/slurm/model_profiles/<alias>.env` with:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py --family qwen --collection source_collection_pubmed --write-profiles
```

On CINECA, pass the real Slurm account if the default placeholder has not been replaced:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --family qwen \
  --collection source_collection_pubmed \
  --hpc-account iscrc_specdlm \
  --hpc-qos boost_qos_bprod \
  --hpc-partition boost_usr_prod \
  --vllm-venv-activate /leonardo_work/IscrC_SpecDLM/FRAG/.venv_frag_vllm/bin/activate \
  --write-profiles
```

These files are ignored by Git because they may contain local CINECA account and environment paths. Submit commands pass the intended profile with `HPC_PRIVATE_ENV=...`. Existing env files are not overwritten unless `--overwrite-profiles` is passed.

Recommended launch order:

1. Download or verify local snapshots for one family, starting with Qwen small models because they are ungated and should exercise the pipeline quickly.
2. Create prompt loads for that family and one collection.
3. Run one worst-prompt smoke job for one small, one medium, and one large profile.
4. Inspect `run_summary.json`, `generation_errors.jsonl`, and Slurm logs for memory, prompt-length, and answer-format issues.
5. If smoke tests are clean, launch full jobs in waves: up to 16 concurrent `small_1gpu` or `medium_1gpu` jobs, and up to 8 concurrent `large_2gpu` jobs on 16 A100 64GB GPUs.

### Development Plan

Ultimate goal: build a repeatable CINECA workflow that downloads all target models, generates model-specific prompt loads, runs smoke tests, and launches the full FRAG/RAG/zero-shot matrix quickly without overwriting outputs or wasting GPU time.

1. Commit the current sweep setup.
   How: review the current diff, then commit `model_sweep.json`, planner scripts, Slurm profile support, and this planning document so the baseline is fixed before further changes.

2. Add a profile materialization command.
   How: use `plan_model_sweep.py --write-profiles` to write `llm_frag_evaluation/slurm/model_profiles/<alias>.env` files directly from `model_sweep.json`; this reduces manual copy errors from printed profile blocks.

3. Add a CINECA dry-run checklist.
   How: document or generate checks for `HF_TOKEN`, account/QOS/partition, environment activation, model snapshot paths, prompt-load existence, ignored profile env files, and output directories before any `sbatch` command is run.

4. Generate prompt loads per model alias.
   How: use `plan_model_sweep.py --section prompt-loads` to print model-specific `create_prompt_loads.py` commands. Each model alias writes to its own prompt-load directory, preventing output collisions with the completed Llama 3.1 70B baseline.

5. Download models in controlled batches.
   How: start with ungated Qwen models using `print_model_download_commands.py --family qwen`, verify local snapshot folders, then handle gated Meta and Google models after license acceptance and token access are confirmed.

6. Run worst-prompt smoke tests.
   How: use `plan_model_sweep.py --section smoke` to create longest-prompt diagnostic loads and submit one representative smoke job for `small_1gpu`, `medium_1gpu`, and `large_2gpu` profiles.

7. Inspect smoke outputs before full launch.
   How: check `run_summary.json`, `generation_errors.jsonl`, Slurm `.out/.err` logs, prompt-length failures, invalid answers, GPU memory behavior, and average seconds per prompt; adjust `batch_size`, `max_num_seqs`, `max_model_len`, or GPU count in `model_sweep.json` if needed.

8. Launch full experiments in waves.
   How: use `plan_model_sweep.py --section submit`, but launch by family/profile instead of all at once. Target up to 16 concurrent `small_1gpu` jobs, up to 16 concurrent `medium_1gpu` jobs after smoke validation, and up to 8 concurrent `large_2gpu` jobs.

9. Add monitoring helpers if needed.
   How: extend the existing diagnostics to summarize `run_summary.json` files by model, source collection, dataset, retriever, experiment, runtime, prompt errors, invalid answers, and missing predictions.

10. Validate and evaluate predictions.
    How: reuse `validate_predictions.py` and `evaluate_predictions.py` per completed run. Keep the same missing-prediction policy as the Llama 3.1 70B baseline: missing or invalid predictions are counted as incorrect and documented.

11. Produce model comparison tables.
    How: add result tables grouped by model family and source collection, with Llama 3.1 70B as the baseline and the same precision, recall, F1, accuracy, and missing-prediction reporting.

12. Finalize the operational recipe.
    How: update the repo docs with the tested CINECA commands, confirmed profiles, known model-specific quirks, and final concurrency limits after smoke and full-run evidence.

### Launch Profile Plan

Create or prepare model-specific launch profiles instead of using one `hpc.private.env` for everything. Smaller models should not pay the memory cost of the 70B long-context profile.

Initial profile sketch:

| Model size | GPUs | Tensor parallel | Suggested context policy | Initial batch size |
|---|---:|---:|---|---:|
| 1B-4B | 1 | 1 | Use prompt-length diagnostics; likely 12288-22528 depending on prompt load | 4-8 |
| 7B-14B | 1 | 1 | Use prompt-length diagnostics; likely 12288-22528 depending on prompt load | 2-6 |
| 27B-32B | 2 | 2 | Use prompt-length diagnostics; start conservative | 1-4 |
| 70B-72B | 4 | 4 | Known PubMed MedQA needs 22528 for long prompts | 1-2 |

Before full runs, create longest-prompt smoke loads for each model/profile combination using:

```bash
python llm_frag_evaluation/tests/diagnostics/create_long_prompt_smoke_load.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl \
  --model-path /leonardo_work/IscrC_SpecDLM/models/<MODEL_LOCAL_DIR> \
  --run-name <model>_pubmed_medqa_longest \
  --top-longest 5 \
  --include-first 2
```

Then run the diagnostic output with the intended profile before launching full jobs.

### Experiment Creation Tasks

1. Add or generate model aliases for all new models so outputs do not overwrite `Meta-Llama-3-70B-Instruct`.
2. Create prompt loads or adapt prompt-load paths per model alias if the current prompt-load structure is model-specific.
3. Build a submission matrix: model, dataset, retriever, experiment, source collection, context window, GPUs, tensor parallel size, batch size.
4. Start with small models and smoke tests to validate tokenizer/chat-template compatibility.
5. For each model family, run a small subset first, inspect outputs for valid answer format, then launch the full matrix.

## PubMed Campaign Status

| Dataset | Status | Next action |
|---|---|---|
| medqa | Complete/accepted; metrics recorded. BM25 RAG has 9 missing predictions and BM25 FRAG has 11 missing predictions. | Add/maintain table values and missing-prediction note |
| bioasq | Complete; metrics recorded for all five settings | Add/maintain table values |
| mmlu | Complete; metrics recorded. BM25 RAG and BM25 FRAG each have 4 missing predictions. | Add/maintain table values and missing-prediction note |
| pubmedqa | RAG/FRAG complete/accepted with 1 missing or invalid prediction per run; metrics recorded. Zero-shot reused from retrieval-independent baseline. | Add/maintain table values and missing-prediction note |

All PubMed prompt loads are under:

```text
llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/
```

All PubMed predictions are expected under:

```text
llm_frag_evaluation/outputs/predictions/source_collection_pubmed/
```

## PubMed Context Sizing

Do not assume the Wikipedia `12288` context setting is sufficient for PubMed-backed RAG/FRAG prompt loads. PubMed passages are longer, and context requirements vary by dataset, retriever, and experiment.

Inspect all PubMed prompt loads before full generation:

```bash
python llm_frag_evaluation/scripts/report_prompt_lengths.py \
  --prompt-load-dir llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed \
  --model-path /leonardo_work/IscrC_SpecDLM/models/Llama-3.1-70B-Instruct \
  --threshold 12288 \
  --threshold 16384 \
  --threshold 20480 \
  --threshold 22528 \
  --threshold 24576 \
  --threshold 32768
```

For a specific failed or smoke run, generate a diagnostic report with a recommended `GENERATE_MAX_MODEL_LEN`:

```bash
python llm_frag_evaluation/tests/diagnostics/diagnose_vllm_run.py \
  --summary llm_frag_evaluation/outputs/predictions/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/run_summary.json \
  --errors llm_frag_evaluation/outputs/predictions/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/generation_errors.jsonl \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl \
  --model-path /leonardo_work/IscrC_SpecDLM/models/Llama-3.1-70B-Instruct \
  --run-name pubmed_medqa_contriever_frag_12288 \
  --context-buffer-tokens 512
```

Current measured MedQA Contriever FRAG recommendation:

```bash
export GENERATE_MAX_MODEL_LEN="22528"
export GENERATE_BATCH_SIZE="1"
export GENERATE_GPU_MEMORY_UTILIZATION="0.90"
```

Use `GENERATE_BATCH_SIZE=1` for worst-prompt smoke tests. Increase batch size only after the longest-prompt smoke succeeds.

## Validate PubMed MedQA

Run these after the MedQA PubMed generation jobs finish:

```bash
source .venv_frag/bin/activate

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Expected full-run result for each validation:

```text
errors: 0
```

## Evaluate PubMed MedQA

```bash
source .venv_frag/bin/activate

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json \
  --dataset medqa --retriever bm25 --experiment zero_shot \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json \
  --dataset medqa --retriever bm25 --experiment standard_rag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json \
  --dataset medqa --retriever bm25 --experiment frag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_Contriever.json \
  --dataset medqa --retriever contriever --experiment standard_rag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_Contriever.json \
  --dataset medqa --retriever contriever --experiment frag \
  --llm Meta-Llama-3-70B-Instruct
```

## Submit Remaining PubMed Datasets

Submit one dataset at a time, keeping the same 5-job pattern: `bm25/zero_shot`, `bm25/standard_rag`, `bm25/frag`, `contriever/standard_rag`, `contriever/frag`.

### BioASQ

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/bioasq/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/bioasq/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/bioasq/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/bioasq/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/bioasq/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

### MMLU

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/mmlu/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/mmlu/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/mmlu/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/mmlu/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/mmlu/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

### PubMedQA

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/pubmedqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/pubmedqa/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/pubmedqa/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/pubmedqa/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/pubmedqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

## Previous Wikipedia Campaign

All 20 planned Wikipedia-resource generation jobs are complete and all metrics have been recorded in `llm_frag_evaluation/RESULTS_TABLE.md`.

### Completed Experiments

| Dataset | Retriever | Experiment | Notes |
|---|---|---|---|
| mmlu | bm25 | zero_shot | Completed, metrics recorded |
| medqa | bm25 | zero_shot | Completed, metrics recorded |
| pubmedqa | bm25 | zero_shot | Completed, metrics recorded with 2 missing/invalid predictions |
| bioasq | bm25 | zero_shot | Completed, metrics recorded |
| pubmedqa | bm25 | standard_rag | Completed, metrics recorded with 1 missing/invalid prediction |
| pubmedqa | bm25 | frag | Completed, metrics recorded with 1 missing/invalid prediction |
| pubmedqa | contriever | standard_rag | Completed, metrics recorded with 1 missing/invalid prediction |
| pubmedqa | contriever | frag | Completed, metrics recorded with 1 missing/invalid prediction |
| bioasq | bm25 | standard_rag | Completed, metrics recorded |
| bioasq | bm25 | frag | Completed, metrics recorded |
| bioasq | contriever | standard_rag | Completed, metrics recorded |
| bioasq | contriever | frag | Completed, metrics recorded |
| medqa | bm25 | standard_rag | Completed, metrics recorded |
| medqa | bm25 | frag | Completed, metrics recorded |
| medqa | contriever | standard_rag | Completed, metrics recorded |
| medqa | contriever | frag | Completed, metrics recorded |
| mmlu | bm25 | standard_rag | Completed, metrics recorded |
| mmlu | bm25 | frag | Completed, metrics recorded |
| mmlu | contriever | standard_rag | Completed, metrics recorded |
| mmlu | contriever | frag | Completed, metrics recorded |

### Last Wikipedia Metric Commands

```bash
source .venv_frag/bin/activate

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_medqa_scored_bm25.json \
  --dataset medqa --retriever bm25 --experiment standard_rag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_medqa_scored_bm25.json \
  --dataset medqa --retriever bm25 --experiment frag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_medqa_scored_contriever.json \
  --dataset medqa --retriever contriever --experiment standard_rag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_medqa_scored_contriever.json \
  --dataset medqa --retriever contriever --experiment frag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_mmlu_scored_bm25.json \
  --dataset mmlu --retriever bm25 --experiment standard_rag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_mmlu_scored_bm25.json \
  --dataset mmlu --retriever bm25 --experiment frag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_mmlu_scored_contriever.json \
  --dataset mmlu --retriever contriever --experiment standard_rag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_mmlu_scored_contriever.json \
  --dataset mmlu --retriever contriever --experiment frag \
  --llm Meta-Llama-3-70B-Instruct
```

## Failure Handling

Inspect summaries and per-record errors:

```bash
cat llm_frag_evaluation/outputs/predictions/<dataset>/<retriever>/<experiment>/Meta-Llama-3-70B-Instruct/run_summary.json
cat llm_frag_evaluation/outputs/predictions/<dataset>/<retriever>/<experiment>/Meta-Llama-3-70B-Instruct/generation_errors.jsonl
tail -n 100 llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.err
tail -n 100 llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.out
```

Known PubMedQA issue:

- `test_31.json` can be missing because the model treats `"Is it Crohn's disease?"` as underspecified and does not output a valid `yes/no/maybe`.
- Count this as missing/wrong; do not manually fill the gold label.
