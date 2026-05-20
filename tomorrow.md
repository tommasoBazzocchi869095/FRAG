# CINECA `llm_frag_evaluation` Next Steps

Current state as of May 13, 2026:

- Prompt code is aligned with previous MedRAG templates and pushed to GitHub.
- The completed experiments used Wikipedia as the retrieval resource, so their Step 2 inputs are the files under `llm_frag_evaluation/data/inputs/source_collection_wiki/`.
- PubMed-backed reruns should use `llm_frag_evaluation/configs/pubmed_config.json`, which selects the matching files under `llm_frag_evaluation/data/inputs/source_collection_pubmed/` and writes collection-qualified prompt loads/predictions.
- Zero-shot is not retriever-specific; run it only once per dataset using `bm25/zero_shot`.
- Prompt loads should now be 20 files after regeneration, not 24.
- RAG/FRAG require `GENERATE_MAX_MODEL_LEN=12288`.
- Prompt length report shows 0 prompts over 12288 tokens; max prompt is 11025 tokens.
- The vLLM runner logs individual prompt errors and continues instead of aborting the whole job.
- The PubMed-resource campaign has started. MedQA was the first PubMed dataset run.
- Next: validate and evaluate the PubMed MedQA outputs, then run the remaining PubMed datasets: BioASQ, MMLU, and PubMedQA.

## PubMed Campaign Status

| Dataset | Status | Next action |
|---|---|---|
| medqa | Generation submitted/run first | Validate predictions, compute metrics, record results |
| bioasq | Not run yet for PubMed resource | Submit 5 prompt-load jobs |
| mmlu | Not run yet for PubMed resource | Submit 5 prompt-load jobs |
| pubmedqa | Not run yet for PubMed resource | Submit 5 prompt-load jobs |

All PubMed prompt loads are under:

```text
llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/
```

All PubMed predictions are expected under:

```text
llm_frag_evaluation/outputs/predictions/source_collection_pubmed/
```

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
