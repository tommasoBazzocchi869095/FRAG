# Experiment Creation To Evaluation Example

This document shows how to run one experiment for one dataset, then generalize the same workflow to all datasets.

The pattern is the same for Wikipedia and PubMed:

1. Create prompt loads.
2. Validate prompt loads.
3. Submit generation jobs.
4. Validate predictions.
5. Compute metrics.

The only things that change are:

- the collection config: `wiki_config.json` or `pubmed_config.json`
- the input file path
- the source collection prefix: `source_collection_wiki` or `source_collection_pubmed`

## Example 1: One Dataset

This example uses `MedQA`, `bm25`, and `frag`.

### 1. Create The Prompt Load

Wikipedia example:

```bash
python llm_frag_evaluation/scripts/create_prompt_loads.py \
  --config llm_frag_evaluation/configs/wiki_config.json \
  --input-file source_collection_wiki/cache_step2_medqa_scored_bm25.json \
  --experiment frag \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-alias Llama-3.1-8B-Instruct
```

PubMed example:

```bash
python llm_frag_evaluation/scripts/create_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json \
  --experiment frag \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-alias Llama-3.1-8B-Instruct
```

The output prompt load will be written under:

```text
llm_frag_evaluation/outputs/prompt_loads/<source_collection>/medqa/bm25/frag/Llama-3.1-8B-Instruct/prompts.jsonl
```

### 2. Validate The Prompt Load

```bash
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/wiki_config.json \
  --input-file source_collection_wiki/cache_step2_medqa_scored_bm25.json
```

For PubMed, swap the config and input file:

```bash
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json
```

### 3. Submit The Generation Job

Use the prompt-load path created above:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  --model-alias Llama-3.1-8B-Instruct \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki/medqa/bm25/frag/Llama-3.1-8B-Instruct/prompts.jsonl
```

For a smoke test, use the smoke wrapper instead:

```bash
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load_smoke.sh \
  --model-alias Llama-3.1-8B-Instruct \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki/medqa/bm25/frag/Llama-3.1-8B-Instruct/prompts.jsonl
```

### 4. Validate Predictions

After the job finishes:

```bash
python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki/medqa/bm25/frag/Llama-3.1-8B-Instruct/prompts.jsonl
```

### 5. Evaluate Metrics

```bash
python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --config llm_frag_evaluation/configs/wiki_config.json \
  --input-file source_collection_wiki/cache_step2_medqa_scored_bm25.json \
  --dataset medqa \
  --retriever bm25 \
  --experiment frag \
  --llm Llama-3.1-8B-Instruct
```

The metrics are printed as JSON and can be copied into the results table or LaTeX manually.

## Example 2: Generalize To All Datasets

The generalization is simple: replace the single `--input-file` with `--all-input-files`.

### Wikipedia

Create all prompt loads:

```bash
python llm_frag_evaluation/scripts/create_prompt_loads.py \
  --config llm_frag_evaluation/configs/wiki_config.json \
  --all-input-files \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-alias Llama-3.1-8B-Instruct
```

Validate all prompt loads:

```bash
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/wiki_config.json \
  --all-input-files
```

Run all generation jobs:

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_wiki \
  --section submit > /tmp/frag_submit_wiki.sh

bash /tmp/frag_submit_wiki.sh
```

Validate all predictions by looping over the prompt loads:

```bash
find llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki -name prompts.jsonl | sort > /tmp/wiki_prompt_loads.txt

while IFS= read -r prompt_load; do
  python llm_frag_evaluation/scripts/validate_predictions.py \
    --prompt-load "$prompt_load"
done < /tmp/wiki_prompt_loads.txt
```

Evaluate all results by looping over the prompt loads:

```bash
while IFS= read -r prompt_load; do
  rel="${prompt_load#llm_frag_evaluation/outputs/prompt_loads/source_collection_wiki/}"
  dataset="${rel%%/*}"
  rest="${rel#*/}"
  retriever="${rest%%/*}"
  rest="${rest#*/}"
  experiment="${rest%%/*}"
  llm_dir="${rest#*/}"
  llm="${llm_dir%/prompts.jsonl}"

  case "${dataset}:${retriever}" in
    mmlu:bm25) input_file="source_collection_wiki/cache_step2_mmlu_scored_bm25.json" ;;
    mmlu:contriever) input_file="source_collection_wiki/cache_step2_mmlu_scored_contriever.json" ;;
    medqa:bm25) input_file="source_collection_wiki/cache_step2_medqa_scored_bm25.json" ;;
    medqa:contriever) input_file="source_collection_wiki/cache_step2_medqa_scored_contriever.json" ;;
    pubmedqa:bm25) input_file="source_collection_wiki/cache_step2_pubmedqa_scored_bm25.json" ;;
    pubmedqa:contriever) input_file="source_collection_wiki/cache_step2_pubmedqa_scored_contriever.json" ;;
    bioasq:bm25) input_file="source_collection_wiki/cache_step2_bioasq_scored_bm25 (1).json" ;;
    bioasq:contriever) input_file="source_collection_wiki/cache_step2_bioasq_scored_contriever (1).json" ;;
    *) continue ;;
  esac

  python llm_frag_evaluation/scripts/evaluate_predictions.py \
    --config llm_frag_evaluation/configs/wiki_config.json \
    --input-file "$input_file" \
    --dataset "$dataset" \
    --retriever "$retriever" \
    --experiment "$experiment" \
    --llm "$llm"
done < /tmp/wiki_prompt_loads.txt
```

### PubMed

The PubMed version is the same pattern with different inputs:

```bash
python llm_frag_evaluation/scripts/create_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-alias Llama-3.1-8B-Instruct
```

```bash
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files
```

```bash
python llm_frag_evaluation/scripts/plan_model_sweep.py \
  --collection source_collection_pubmed \
  --section submit > /tmp/frag_submit_pubmed.sh

bash /tmp/frag_submit_pubmed.sh
```

```bash
find llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed -name prompts.jsonl | sort > /tmp/pubmed_prompt_loads.txt
```

```bash
while IFS= read -r prompt_load; do
  python llm_frag_evaluation/scripts/validate_predictions.py \
    --prompt-load "$prompt_load"
done < /tmp/pubmed_prompt_loads.txt
```

```bash
while IFS= read -r prompt_load; do
  rel="${prompt_load#llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/}"
  dataset="${rel%%/*}"
  rest="${rel#*/}"
  retriever="${rest%%/*}"
  rest="${rest#*/}"
  experiment="${rest%%/*}"
  llm_dir="${rest#*/}"
  llm="${llm_dir%/prompts.jsonl}"

  case "$retriever" in
    bm25) retriever_tag="bm25" ;;
    contriever) retriever_tag="Contriever" ;;
    *) continue ;;
  esac

  input_file="source_collection_pubmed/cache_step2_${dataset}_scored_PubMed_${retriever_tag}.json"

  python llm_frag_evaluation/scripts/evaluate_predictions.py \
    --config llm_frag_evaluation/configs/pubmed_config.json \
    --input-file "$input_file" \
    --dataset "$dataset" \
    --retriever "$retriever" \
    --experiment "$experiment" \
    --llm "$llm"
done < /tmp/pubmed_prompt_loads.txt
```

## Generalization Rule

The reusable pattern is:

1. Choose a collection config.
2. Choose one `--input-file` for a single dataset, or `--all-input-files` for the full set.
3. Create prompt loads.
4. Validate prompt loads.
5. Launch generation.
6. Validate predictions.
7. Evaluate metrics.

The important path rule is:

```text
source_collection_wiki/cache_step2_<dataset>_scored_<retriever>.json
source_collection_pubmed/cache_step2_<dataset>_scored_PubMed_<retriever>.json
```

For zero-shot, use only the `bm25` prompt loads.
