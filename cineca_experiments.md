# CINECA Experiments

Run from the repository root:

```bash
cd /path/to/hpc/work/FRAG
source .venv_frag/bin/activate
```

Before submitting full RAG/FRAG jobs, confirm:

```bash
grep -E "GENERATE_MAX_MODEL_LEN|GENERATE_MAX_TOKENS|GENERATE_TIME_LIMIT|GENERATE_GPUS|GENERATE_TENSOR_PARALLEL_SIZE" \
  llm_frag_evaluation/slurm/hpc.private.env
```

Expected:

```bash
export GENERATE_MAX_MODEL_LEN="<set from prompt diagnostics>"
export GENERATE_MAX_TOKENS="1024"
export GENERATE_TIME_LIMIT="12:00:00"
export GENERATE_GPUS="4"
export GENERATE_TENSOR_PARALLEL_SIZE="4"
```

For the current PubMed MedQA RAG/FRAG reruns, diagnostics recommended:

```bash
export GENERATE_MAX_MODEL_LEN="22528"
export GENERATE_BATCH_SIZE="1"
export GENERATE_GPU_MEMORY_UTILIZATION="0.90"
```

The `22528` value is based on a measured max PubMed MedQA Contriever FRAG prompt of 20280 tokens, plus 1024 generation tokens, plus a 512-token buffer, rounded to the next 1024-token multiple. Do not apply this value blindly to every future campaign; use the diagnostics below to choose the smallest safe context for each prompt-load group.

## Monitoring

```bash
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %R"
```

Newest logs:

```bash
ls -lt llm_frag_evaluation/outputs/logs | head -20
```

Tail one job:

```bash
tail -f llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.out
```

Check errors:

```bash
cat llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.err
```

## Time Estimates

These are rough estimates from smoke tests with Llama 3.1 70B, TP=4.

| Experiment | Avg Prompt Tokens | Generation Time | Model Load | Notes |
|---|---:|---:|---:|---|
| zero_shot | 200-420 | ~4.8 sec/prompt | ~6-7 min/job | Short prompts |
| standard_rag | 5000-20000+ depending on collection | ~5.9 sec/prompt in earlier 12k runs | ~6-7 min/job | Set `max_model_len` from prompt diagnostics |
| frag | 5000-20000+ depending on collection | ~5.6 sec/prompt in earlier 12k runs | ~6-7 min/job | Set `max_model_len` from prompt diagnostics |

Approximate full-job wall times:

| Dataset | Count | Zero Shot | RAG / FRAG |
|---|---:|---:|---:|
| bioasq | 618 | ~55 min | ~70 min |
| medqa | 1273 | ~1h 50m | ~2h 15m |
| mmlu | 1089 | ~1h 35m | ~1h 55m |
| pubmedqa | 500 | ~45 min | ~55 min |

The full campaign now has 20 jobs: 4 zero-shot jobs, 8 standard RAG jobs, and 8 FRAG jobs. Zero-shot is not retriever-specific and should only be run once per dataset using the `bm25/zero_shot` prompt loads.

The completed campaign used Wikipedia as the retrieval resource. Its Step 2 inputs live under `llm_frag_evaluation/data/inputs/source_collection_wiki/`. For PubMed-backed retrieval runs, use the corresponding files under `llm_frag_evaluation/data/inputs/source_collection_pubmed/`.

## PubMed Resource Campaign

For the new PubMed-backed generation campaign, create a separate prompt-load tree from the PubMed collection:

```bash
python llm_frag_evaluation/scripts/create_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files
```

Validate it before submitting jobs:

```bash
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files
```

The PubMed prompt loads are written under:

```text
llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/
```

Prediction files generated from those prompt loads are written under:

```text
llm_frag_evaluation/outputs/predictions/source_collection_pubmed/
```

Submit the PubMed prompt loads using the collection-qualified path, for example:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

As before, do not submit `contriever/zero_shot`; zero-shot is not retriever-specific and is generated only under `bm25`.

All jobs request 4 GPUs. Running two jobs in parallel requests 8 GPUs.

### PubMed Context Diagnostics

PubMed prompt loads can be substantially longer than the completed Wikipedia prompt loads. Before submitting full RAG/FRAG jobs, tokenize the prompt loads and set the context window from measured lengths:

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

For a failed run, generate a diagnostic report:

```bash
python llm_frag_evaluation/tests/diagnostics/diagnose_vllm_run.py \
  --summary llm_frag_evaluation/outputs/predictions/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/run_summary.json \
  --errors llm_frag_evaluation/outputs/predictions/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/generation_errors.jsonl \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl \
  --model-path /leonardo_work/IscrC_SpecDLM/models/Llama-3.1-70B-Instruct \
  --run-name pubmed_medqa_contriever_frag_12288 \
  --context-buffer-tokens 512
```

The diagnostic reports:

```text
max prompt tokens + max generation tokens + safety buffer
```

and prints the exact `GENERATE_MAX_MODEL_LEN` line to set in `llm_frag_evaluation/slurm/hpc.private.env`.

Current measured PubMed MedQA Contriever FRAG values:

| Field | Value |
|---|---:|
| Prompt count | 1273 |
| Max prompt tokens | 20280 |
| Prompts over 12288 | 406 |
| Prompts over 16384 | 90 |
| Max generation tokens | 1024 |
| Safety buffer | 512 |
| Recommended `GENERATE_MAX_MODEL_LEN` | 22528 |

The longest-prompt smoke prompt load completed 7/7 predictions at `GENERATE_MAX_MODEL_LEN=22528`, `GENERATE_BATCH_SIZE=1`, and `GENERATE_GPU_MEMORY_UTILIZATION=0.90`.

### PubMed Job Status As Of 2026-05-28

The all-run diagnostic over `source_collection_pubmed` found 20 prompt loads:

| Group | Status | Action |
|---|---|---|
| `bioasq/bm25/zero_shot` | Complete | No rerun needed |
| `medqa/bm25/zero_shot` | Complete | No rerun needed |
| MedQA BM25/Contriever RAG/FRAG | Complete/accepted; metrics recorded. BM25 RAG has 9 missing predictions and BM25 FRAG has 11 missing predictions. | Keep missing-prediction note in tables |
| BioASQ RAG/FRAG | Complete; metrics recorded | No rerun needed |
| MMLU all prompt loads | Complete; metrics recorded. BM25 RAG and BM25 FRAG each have 4 missing predictions. | Keep missing-prediction note in tables |
| PubMedQA RAG/FRAG | Complete/accepted; metrics recorded with 1 missing or invalid prediction per run | Keep missing-prediction note in tables |

With 16 GPUs available and `GENERATE_GPUS=4`, run at most four jobs concurrently. For large-context jobs, start with `GENERATE_BATCH_SIZE=1`; increase only after a worst-prompt smoke test succeeds.

## Two-At-A-Time Schedule

If running two jobs at a time, use this cadence. The check-back time is when it is reasonable to log in again and submit the next pair.

|         Round | Jobs | Check Back After |
|--------------:|---|---:|
| 1 (Completed) | `medqa/bm25/zero_shot` + `mmlu/bm25/zero_shot` | ~2h |
| 2 (Completed) | `bioasq/bm25/zero_shot` + `pubmedqa/bm25/zero_shot` | ~1h |
| 3 (Completed) | `medqa/bm25/standard_rag` + `mmlu/bm25/standard_rag` | ~2.5h |
| 4 (Completed) | `bioasq/bm25/standard_rag` + `pubmedqa/bm25/standard_rag` | ~1.5h |
| 5 (Completed) | `medqa/contriever/standard_rag` + `mmlu/contriever/standard_rag` | ~2.5h |
| 6 (Completed) | `bioasq/contriever/standard_rag` + `pubmedqa/contriever/standard_rag` | ~1.5h |
| 7 (Completed) | `medqa/bm25/frag` + `mmlu/bm25/frag` | ~2.5h |
| 8 (Completed) | `bioasq/bm25/frag` + `pubmedqa/bm25/frag` | ~1.5h |
| 9 (Completed) | `medqa/contriever/frag` + `mmlu/contriever/frag` | ~2.5h |
| 10 (Completed) | `bioasq/contriever/frag` + `pubmedqa/contriever/frag` | ~1.5h |

Current deviation from the pair schedule for the completed Wikipedia-resource campaign: PubMedQA and BioASQ RAG/FRAG blocks were run as four-job batches. The MedQA and MMLU RAG/FRAG blocks are also complete. All 20 planned Wikipedia full-generation jobs are complete.

## Current Job Status

Completed:

| Dataset | Retriever | Experiment | Status |
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

No planned CINECA generation jobs remain for the Wikipedia-resource campaign. The PubMed-resource Llama 3.1 70B campaign is also complete and recorded; next planned work is the broader model sweep.

The exact prompt-load path appears in each job's `.out` file. For completed jobs, inspect the corresponding `frag-vllm_<JOBID>.out` log if the prompt-load path is needed later.

## Submit Zero-Shot Jobs

Recommended: start with two jobs in parallel, then submit more as jobs finish.

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/bioasq/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/pubmedqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Zero-shot is not retriever-specific. Run it only once per dataset, using the `bm25/zero_shot` prompt loads.

## Submit Standard RAG Jobs

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/bioasq/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/pubmedqa/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/bioasq/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/pubmedqa/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

## Submit FRAG Jobs

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/bioasq/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/pubmedqa/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/bioasq/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/pubmedqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

## Validate Completed Predictions

Run this for each completed prompt load:

```bash
python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/<dataset>/<retriever>/<experiment>/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Example:

```bash
python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Expected full-run result:

```text
errors: 0
```

## Compute Metrics

Example:

```bash
python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file source_collection_wiki/cache_step2_medqa_scored_bm25.json \
  --dataset medqa \
  --retriever bm25 \
  --experiment zero_shot \
  --llm Meta-Llama-3-70B-Instruct
```

Use the matching input file for each dataset/retriever. For the completed Wikipedia-resource experiments, use `source_collection_wiki`:

| Dataset | Retriever | Input file |
|---|---|---|
| medqa | bm25 | `source_collection_wiki/cache_step2_medqa_scored_bm25.json` |
| medqa | contriever | `source_collection_wiki/cache_step2_medqa_scored_contriever.json` |
| mmlu | bm25 | `source_collection_wiki/cache_step2_mmlu_scored_bm25.json` |
| mmlu | contriever | `source_collection_wiki/cache_step2_mmlu_scored_contriever.json` |
| pubmedqa | bm25 | `source_collection_wiki/cache_step2_pubmedqa_scored_bm25.json` |
| pubmedqa | contriever | `source_collection_wiki/cache_step2_pubmedqa_scored_contriever.json` |
| bioasq | bm25 | `source_collection_wiki/cache_step2_bioasq_scored_bm25 (1).json` |
| bioasq | contriever | `source_collection_wiki/cache_step2_bioasq_scored_contriever (1).json` |

For PubMed-resource experiments, use the matching files in `source_collection_pubmed`, for example `source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json` and `source_collection_pubmed/cache_step2_medqa_scored_PubMed_Contriever.json`.

## Failure Handling

The runner now continues after individual prompt errors:

- prompt too long: logged as `PromptTooLong`
- invalid generated answer: logged as `InvalidAnswer`
- per-record vLLM generation failure: logged with the exception type

Inspect:

```bash
cat llm_frag_evaluation/outputs/predictions/<dataset>/<retriever>/<experiment>/Meta-Llama-3-70B-Instruct/generation_errors.jsonl
cat llm_frag_evaluation/outputs/predictions/<dataset>/<retriever>/<experiment>/Meta-Llama-3-70B-Instruct/run_summary.json
```
