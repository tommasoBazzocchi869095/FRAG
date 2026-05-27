# vLLM Run Diagnostics

This folder is for small, reproducible diagnostics of CINECA generation runs.

Use it after each smoke or full vLLM job to inspect:

- `run_summary.json`
- `generation_errors.jsonl`
- optional prompt-load token lengths

Example on CINECA:

```bash
python llm_frag_evaluation/tests/diagnostics/diagnose_vllm_run.py \
  --summary llm_frag_evaluation/outputs/predictions/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/run_summary.json \
  --errors llm_frag_evaluation/outputs/predictions/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/generation_errors.jsonl \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl \
  --model-path /leonardo_work/IscrC_SpecDLM/models/Llama-3.1-70B-Instruct \
  --run-name pubmed_medqa_contriever_frag_12288
```

The script writes a Markdown report and a JSON summary under `reports/`.

To test a context-length fix, create a smoke prompt-load from the longest prompts
instead of only running the first five records:

```bash
python llm_frag_evaluation/tests/diagnostics/create_long_prompt_smoke_load.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl \
  --model-path /leonardo_work/IscrC_SpecDLM/models/Llama-3.1-70B-Instruct \
  --run-name pubmed_medqa_contriever_frag_longest \
  --top-longest 5 \
  --include-first 2
```

Then submit the generated `prompts.jsonl` with the normal full submitter. Do not use
`submit_generate_prompt_load_smoke.sh` for this diagnostic file because the file is
already small.
