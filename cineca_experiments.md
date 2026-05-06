# CINECA Experiments

Run from the repository root:

```bash
cd /leonardo_work/IscrC_SpecDLM/FRAG
source .venv_frag/bin/activate
```

Before submitting full RAG/FRAG jobs, confirm:

```bash
grep -E "GENERATE_MAX_MODEL_LEN|GENERATE_MAX_TOKENS|GENERATE_TIME_LIMIT|GENERATE_GPUS|GENERATE_TENSOR_PARALLEL_SIZE" \
  llm_frag_evaluation/slurm/hpc.private.env
```

Expected:

```bash
export GENERATE_MAX_MODEL_LEN="12288"
export GENERATE_MAX_TOKENS="1024"
export GENERATE_TIME_LIMIT="12:00:00"
export GENERATE_GPUS="4"
export GENERATE_TENSOR_PARALLEL_SIZE="4"
```

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
| standard_rag | 5000-7000 | ~5.9 sec/prompt | ~6-7 min/job | Needs `max_model_len=12288` |
| frag | 5000-7000 | ~5.6 sec/prompt | ~6-7 min/job | Needs `max_model_len=12288` |

Approximate full-job wall times:

| Dataset | Count | Zero Shot | RAG / FRAG |
|---|---:|---:|---:|
| bioasq | 618 | ~55 min | ~70 min |
| medqa | 1273 | ~1h 50m | ~2h 15m |
| mmlu | 1089 | ~1h 35m | ~1h 55m |
| pubmedqa | 500 | ~45 min | ~55 min |

The full campaign now has 20 jobs: 4 zero-shot jobs, 8 standard RAG jobs, and 8 FRAG jobs. Zero-shot is not retriever-specific and should only be run once per dataset using the `bm25/zero_shot` prompt loads.

All jobs request 4 GPUs. Running two jobs in parallel requests 8 GPUs.

## Two-At-A-Time Schedule

If running two jobs at a time, use this cadence. The check-back time is when it is reasonable to log in again and submit the next pair.

|         Round | Jobs | Check Back After |
|--------------:|---|---:|
| 1 (Completed) | `medqa/bm25/zero_shot` + `mmlu/bm25/zero_shot` | ~2h |
| 2 (Completed) | `bioasq/bm25/zero_shot` + `pubmedqa/bm25/zero_shot` | ~1h |
|             3 | `medqa/bm25/standard_rag` + `mmlu/bm25/standard_rag` | ~2.5h |
| 4 (Partially submitted) | `bioasq/bm25/standard_rag` + `pubmedqa/bm25/standard_rag` | ~1.5h |
|             5 | `medqa/contriever/standard_rag` + `mmlu/contriever/standard_rag` | ~2.5h |
| 6 (Partially submitted) | `bioasq/contriever/standard_rag` + `pubmedqa/contriever/standard_rag` | ~1.5h |
|             7 | `medqa/bm25/frag` + `mmlu/bm25/frag` | ~2.5h |
| 8 (Partially submitted) | `bioasq/bm25/frag` + `pubmedqa/bm25/frag` | ~1.5h |
|             9 | `medqa/contriever/frag` + `mmlu/contriever/frag` | ~2.5h |
| 10 (Partially submitted) | `bioasq/contriever/frag` + `pubmedqa/contriever/frag` | ~1.5h |

Current deviation from the pair schedule: the four PubMedQA RAG/FRAG jobs were submitted together. If queue wait time is high, add that wait time on top of the estimate.

## Current Job Status

Completed:

| Dataset | Retriever | Experiment | Status |
|---|---|---|---|
| mmlu | bm25 | zero_shot | Completed, metrics recorded |
| medqa | bm25 | zero_shot | Completed, metrics recorded |
| pubmedqa | bm25 | zero_shot | Completed, metrics recorded with 2 missing/invalid predictions |
| bioasq | bm25 | zero_shot | Completed, metrics recorded |

Submitted:

| Job ID | Dataset | Retriever | Experiment | SLURM State |
|---:|---|---|---|---|
| 40995888 | pubmedqa | bm25 | standard_rag | Running at last check |
| 40995896 | pubmedqa | bm25 | frag | Running at last check |
| 40995903 | pubmedqa | contriever | standard_rag | Pending at last check |
| 40995913 | pubmedqa | contriever | frag | Pending at last check |

The exact prompt-load path appears in each job's `.out` file at completion. Before completion, use `scontrol show job <JOBID>` to inspect the exported `PROMPT_LOAD_PATH` if available.

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
  --input-file cache_step2_medqa_scored_bm25.json \
  --dataset medqa \
  --retriever bm25 \
  --experiment zero_shot \
  --llm Meta-Llama-3-70B-Instruct
```

Use the matching input file for each dataset/retriever:

| Dataset | Retriever | Input file |
|---|---|---|
| medqa | bm25 | `cache_step2_medqa_scored_bm25.json` |
| medqa | contriever | `cache_step2_medqa_scored_contriever.json` |
| mmlu | bm25 | `cache_step2_mmlu_scored_bm25.json` |
| mmlu | contriever | `cache_step2_mmlu_scored_contriever.json` |
| pubmedqa | bm25 | `cache_step2_pubmedqa_scored_bm25.json` |
| pubmedqa | contriever | `cache_step2_pubmedqa_scored_contriever.json` |
| bioasq | bm25 | `cache_step2_bioasq_scored_bm25 (1).json` |
| bioasq | contriever | `cache_step2_bioasq_scored_contriever (1).json` |

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
