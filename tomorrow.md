# Tomorrow Checklist: Continue CINECA `llm_frag_evaluation`

Current state as of May 6, 2026:

- Prompt code is aligned with previous MedRAG templates and pushed to GitHub.
- Zero-shot is not retriever-specific; run it only once per dataset using `bm25/zero_shot`.
- Prompt loads should now be 20 files after regeneration, not 24.
- RAG/FRAG require `GENERATE_MAX_MODEL_LEN=12288`.
- Prompt length report shows 0 prompts over 12288 tokens; max prompt is 11025 tokens.
- The vLLM runner logs individual prompt errors and continues instead of aborting the whole job.

## Completed Experiments

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

## Currently Submitted

BioASQ RAG/FRAG jobs were submitted together:

| Job ID | Dataset | Retriever | Experiment | Status at last check |
|---:|---|---|---|---|
| 41000410 | bioasq | bm25 | standard_rag | Running |
| 41000414 | bioasq | bm25 | frag | Running |
| 41000419 | bioasq | contriever | standard_rag | Pending |
| 41000425 | bioasq | contriever | frag | Pending |

Check status:

```bash
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %R"
```

## Remaining Experiments After BioASQ Finishes

| Dataset | Retriever | Experiment |
|---|---|---|
| mmlu | bm25 | standard_rag |
| medqa | bm25 | standard_rag |
| mmlu | bm25 | frag |
| medqa | bm25 | frag |
| mmlu | contriever | standard_rag |
| medqa | contriever | standard_rag |
| mmlu | contriever | frag |
| medqa | contriever | frag |

Recommended cadence after BioASQ: run MMLU and MedQA in pairs by experiment/retriever.

## Validate BioASQ When Jobs Finish

```bash
source .venv_frag/bin/activate

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/bioasq/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/bioasq/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/bioasq/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/bioasq/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

## Compute BioASQ Metrics

```bash
python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file "cache_step2_bioasq_scored_bm25 (1).json" \
  --dataset bioasq --retriever bm25 --experiment standard_rag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file "cache_step2_bioasq_scored_bm25 (1).json" \
  --dataset bioasq --retriever bm25 --experiment frag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file "cache_step2_bioasq_scored_contriever (1).json" \
  --dataset bioasq --retriever contriever --experiment standard_rag \
  --llm Meta-Llama-3-70B-Instruct

python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --input-file "cache_step2_bioasq_scored_contriever (1).json" \
  --dataset bioasq --retriever contriever --experiment frag \
  --llm Meta-Llama-3-70B-Instruct
```

## Submit Remaining MMLU/MedQA Jobs

BM25 RAG:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

BM25 FRAG:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Contriever RAG:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Contriever FRAG:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/mmlu/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
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
