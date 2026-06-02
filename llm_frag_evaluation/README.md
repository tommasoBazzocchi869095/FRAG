# LLM FRAG Evaluation

This folder contains a separate line of experiments for evaluating different LLMs with:

1. `zero_shot`: question plus answer options only.
2. `standard_rag`: question, answer options, instructions, and the top 32 topically relevant passages.
3. `frag`: question, answer options, instructions, and the top 32 passages after aggregating topicality and factuality scores over the 100 retrieved passages.

`zero_shot` does not use retrieval, so it is generated only once per dataset. By default those prompt loads are stored under the configured primary retriever, `bm25`.

For the current multi-model CINECA sweep, use the dedicated runbook:

- [`CINECA_MODEL_SWEEP_RUNBOOK.md`](CINECA_MODEL_SWEEP_RUNBOOK.md)

The FRAG score is:

```text
final_score = alpha * normalized_topicality + (1 - alpha) * factuality
```

The default value is `alpha = 0.6`, meaning 60 percent topicality and 40 percent factuality.

## Folder Layout

```text
llm_frag_evaluation/
  configs/
    default_config.json
    wiki_config.json
    pubmed_config.json
    models.example.json
  data/
    inputs/
      README.md
      sample_question.json
      source_collection_wiki/
        cache_step2_<dataset>_scored_<retriever>.json
      source_collection_pubmed/
        cache_step2_<dataset>_scored_PubMed_<retriever>.json
    processed/
      .gitkeep
  outputs/
    predictions/
      .gitkeep
    logs/
      .gitkeep
  prompts/
    default_prompts.json
  scripts/
    evaluate_predictions.py
    run_experiment.py
  slurm/
    run_experiment.slurm
  src/
    config.py
    data_loader.py
    experiments.py
    prompts.py
    scoring.py
  requirements.txt
```

## Input Format

Each input file should contain either a single JSON object or a list of JSON objects. The expected object shape is:

```json
{
  "id": "medqa_test_0001",
  "dataset": "medqa",
  "question": "Question text",
  "options": {
    "A": "First option",
    "B": "Second option",
    "C": "Third option",
    "D": "Fourth option"
  },
  "answer": "A",
  "passages": [
    {
      "id": "passage_001",
      "title": "Optional title",
      "content": "Passage text",
      "score_topic": 12.3,
      "score_factuality": 0.91
    }
  ]
}
```

The `answer` field is optional for generation, but required for the local metric script. The current code accepts common aliases for passage scores, but the preferred names are `score_topic` and `score_factuality`.

Step 2 files named like `cache_step2_medqa_scored_bm25.json` are supported. If the question objects do not contain a `dataset` field, the dataset is inferred from the filename.

The input files are now split by source collection:

- Use `llm_frag_evaluation/data/inputs/source_collection_wiki/` for experiments where Wikipedia is the retrieval resource. The completed Llama 3.1 70B experiments reported in `RESULTS_TABLE.md` used this collection.
- Use `llm_frag_evaluation/data/inputs/source_collection_pubmed/` for experiments where PubMed is the retrieval resource.

When passing `--input-file`, include the path relative to `llm_frag_evaluation/data/inputs`, for example `source_collection_wiki/cache_step2_medqa_scored_bm25.json` or `source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json`.

Use the collection-specific configs for full campaigns:

- `llm_frag_evaluation/configs/wiki_config.json` selects all Wikipedia-source Step 2 files and writes outputs under `outputs/prompt_loads/source_collection_wiki` and `outputs/predictions/source_collection_wiki`.
- `llm_frag_evaluation/configs/pubmed_config.json` selects all PubMed-source Step 2 files and writes outputs under `outputs/prompt_loads/source_collection_pubmed` and `outputs/predictions/source_collection_pubmed`.

For `standard_rag`, the code sorts passages by topicality score and takes the first 32.

For `frag`, the code computes the aggregate score with `alpha = 0.6` for topicality, sorts by that score, and then takes the first 32 passages.

## Output Format

Predictions are saved with one folder per dataset, retriever, experiment, and LLM:

```text
outputs/predictions/
  medqa/
    bm25/
      zero_shot/
        llama-3-8b/
          test_0.json
          test_1.json
      standard_rag/
        llama-3-8b/
          test_0.json
      frag/
        llama-3-8b/
          test_0.json
    contriever/
      standard_rag/
        llama-3-8b/
          test_0.json
      frag/
        llama-3-8b/
          test_0.json
```

Each question file uses the evaluation-compatible list format:

```json
[
  {
    "answer_choice": "B",
    "step_by_step_thinking": "Brief model explanation.",
    "system_info": "Offline Factuality"
  }
]
```

File names start from `test_0.json` for each dataset and continue in input order.

## Experiments

Run locally:

```shell
python llm_frag_evaluation/scripts/run_experiment.py --config llm_frag_evaluation/configs/default_config.json
```

Run one Step 2 file explicitly:

```shell
python llm_frag_evaluation/scripts/run_experiment.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json \
  --experiment frag
```

Create prompt loads for CINECA inference with Llama 3 70B:

```shell
python llm_frag_evaluation/scripts/create_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files
```

Prompt loads are JSONL files saved per source collection, dataset, retriever, experiment, and model. Zero-shot prompt loads are generated only under `bm25` by default because they are not retriever-specific:

```text
outputs/prompt_loads/
  source_collection_pubmed/
    medqa/
      bm25/
        zero_shot/
          Meta-Llama-3-70B-Instruct/
            prompts.jsonl
        standard_rag/
        frag/
      contriever/
        standard_rag/
        frag/
```

Each line contains the request id, dataset, retriever, experiment, target `test_N.json` filename, chat `messages`, gold answer, and selected-passage trace. During inference on CINECA, apply the Llama tokenizer chat template to the `messages` field.

Validate generated prompt loads:

```shell
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files
```

## vLLM Inference

The CINECA path follows the same pattern used in the previous project: keep prompt creation in the base environment and run generation in a separate vLLM environment.

## CINECA Step-By-Step Runbook

This section is the intended operational path from a CINECA login node.

### 1. Clone Or Update The Repository

First clone the repository if it is not already present:

```shell
cd /path/to/hpc/work
git clone https://github.com/<your-org-or-user>/FRAG.git
cd FRAG
```

If the repository is already present, update it:

```shell
cd /path/to/hpc/work/FRAG
git pull
```

### 2. Create The Base Environment

This environment is for prompt creation, prompt validation, and metric scripts.

```shell
module purge
module load python/3.11.7
cd /path/to/hpc/work/FRAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r llm_frag_evaluation/requirements.txt
```

### 3. Create Or Reuse The vLLM Environment

If you already have a working CINECA vLLM environment from another project, reuse it by setting `VLLM_VENV_ACTIVATE` in step 5.

Otherwise create one:

```shell
module purge
module load python/3.11.7
cd /path/to/hpc/work/FRAG
python3 -m venv .venv_vllm
source .venv_vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r llm_frag_evaluation/requirements-vllm.txt
module load cuda/12.6
module load gcc/12.2.0
python -m pip install --only-binary=:all: vllm
python -c "import vllm; print(vllm.__version__)"
```

### 4. Make Sure The Model Is Available

Preferred: use a local model snapshot on CINECA shared storage.

Example login-node download:

```shell
source .venv_vllm/bin/activate
export HF_TOKEN="..."
export MODEL_ID="meta-llama/Meta-Llama-3-70B-Instruct"
export MODEL_PATH="/path/to/hpc/work/models/Meta-Llama-3-70B-Instruct"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_ID', local_dir='$MODEL_PATH', token='$HF_TOKEN', max_workers=2)"
```

If the model already exists from another project, reuse that path.

### 5. Create The Private CINECA Settings File

```shell
cd /path/to/hpc/work/FRAG
cp llm_frag_evaluation/slurm/hpc.private.env.example llm_frag_evaluation/slurm/hpc.private.env
```

Edit `llm_frag_evaluation/slurm/hpc.private.env`:

```shell
nano llm_frag_evaluation/slurm/hpc.private.env
```

Set at least:

```shell
export VLLM_VENV_ACTIVATE="/path/to/hpc/work/FRAG/.venv_vllm/bin/activate"
export HPC_ACCOUNT="your_account"
export HPC_QOS="normal"
export HPC_PARTITION="boost_usr_prod"
export HPC_MODEL_PATH="/path/to/hpc/work/models/Meta-Llama-3-70B-Instruct"
```

Use the same `VLLM_VENV_ACTIVATE` path as your previous working vLLM project if reusing that environment.

### 6. Place Or Verify Step 2 Input Files

The input files should be under the source collection that matches the retrieval resource:

```text
llm_frag_evaluation/data/inputs/source_collection_wiki/
llm_frag_evaluation/data/inputs/source_collection_pubmed/
```

Use `source_collection_wiki` for Wikipedia-backed retrieval runs. These are the files used by the completed experiments in `RESULTS_TABLE.md`:

```text
source_collection_wiki/
cache_step2_medqa_scored_bm25.json
cache_step2_medqa_scored_contriever.json
cache_step2_mmlu_scored_bm25.json
cache_step2_mmlu_scored_contriever.json
cache_step2_pubmedqa_scored_bm25.json
cache_step2_pubmedqa_scored_contriever.json
cache_step2_bioasq_scored_bm25.json
cache_step2_bioasq_scored_contriever.json
```

Use `source_collection_pubmed` for PubMed-backed retrieval runs:

```text
source_collection_pubmed/
cache_step2_medqa_scored_PubMed_bm25.json
cache_step2_medqa_scored_PubMed_Contriever.json
cache_step2_mmlu_scored_PubMed_bm25.json
cache_step2_mmlu_scored_PubMed_Contriever.json
cache_step2_pubmedqa_scored_PubMed_bm25.json
cache_step2_pubmedqa_scored_PubMed_Contriever.json
cache_step2_bioasq_scored_PubMed_bm25.json
cache_step2_bioasq_scored_PubMed_Contriever.json
```

### 7. Create Prompt Loads

Run this in the base environment:

```shell
cd /path/to/hpc/work/FRAG
source .venv/bin/activate
python llm_frag_evaluation/scripts/create_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files
```

Validate the prompt loads:

```shell
python llm_frag_evaluation/scripts/validate_prompt_loads.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --all-input-files
```

Expected result:

```text
errors: 0
```

With the default four datasets and two retrievers, this creates 20 prompt-load files: 4 zero-shot files under `bm25`, plus 16 RAG/FRAG files under `bm25` and `contriever`.

For the PubMed campaign, the prompt loads are written under:

```text
llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/
```

### PubMed Context Sizing

Do not assume that the Wikipedia `GENERATE_MAX_MODEL_LEN=12288` setting is sufficient for PubMed-backed RAG/FRAG jobs. PubMed passages are longer, and the required context window depends on the dataset, retriever, and experiment.

Before full PubMed generation, scan prompt lengths:

```shell
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

For a failed or smoke run, create a diagnostic report:

```shell
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

Current measured PubMed MedQA Contriever FRAG recommendation:

```shell
export GENERATE_MAX_MODEL_LEN="22528"
export GENERATE_BATCH_SIZE="1"
export GENERATE_GPU_MEMORY_UTILIZATION="0.90"
```

This setting was validated by a 7-record longest-prompt smoke run. Use `GENERATE_BATCH_SIZE=1` for worst-prompt smoke tests and increase only after the smoke succeeds.

### 8. Run One Smoke Job

Start with one small smoke run before submitting full jobs:

```shell
cd /path/to/hpc/work/FRAG
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load_smoke.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Monitor:

```shell
squeue -u $USER
tail -f llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.out
tail -f llm_frag_evaluation/outputs/logs/frag-vllm_<JOBID>.err
```

After it finishes, inspect:

```shell
cat llm_frag_evaluation/outputs/predictions/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/run_summary.json
cat llm_frag_evaluation/outputs/predictions/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/generation_errors.jsonl
```

Validate smoke predictions:

```shell
source .venv/bin/activate
python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

For a smoke run, this validation will report missing predictions for records beyond the smoke limit. That is expected. Inspect the first generated `test_N.json` files manually:

```shell
ls llm_frag_evaluation/outputs/predictions/source_collection_pubmed/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/test_*.json | head
cat llm_frag_evaluation/outputs/predictions/source_collection_pubmed/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/test_0.json
```

### 9. Run A Full Prompt Load

Once the smoke run is clean:

```shell
cd /path/to/hpc/work/FRAG
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Repeat for each prompt load you want to run. Prompt loads are organized as:

```text
llm_frag_evaluation/outputs/prompt_loads/<source_collection>/<dataset>/<retriever>/<experiment>/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Do not submit `contriever/zero_shot`; zero-shot is not retriever-specific and is only generated under `bm25` by default.

Examples:

```shell
bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl

bash llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh \
  llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/contriever/frag/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

### 10. Validate Full Predictions

After a full job finishes:

```shell
source .venv/bin/activate
python llm_frag_evaluation/scripts/validate_predictions.py \
  --prompt-load llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed/medqa/bm25/zero_shot/Meta-Llama-3-70B-Instruct/prompts.jsonl
```

Expected result:

```text
errors: 0
```

### 11. Compute Metrics

Run metrics for one dataset/retriever/experiment/model output:

```shell
source .venv/bin/activate
python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json \
  --dataset medqa \
  --retriever bm25 \
  --experiment zero_shot \
  --llm Meta-Llama-3-70B-Instruct
```

The prediction layout includes the retriever level, so pass `--retriever bm25` or `--retriever contriever`.

Select one experiment:

```shell
python llm_frag_evaluation/scripts/run_experiment.py --experiment frag
```

Dry run without calling a model backend:

```shell
python llm_frag_evaluation/scripts/run_experiment.py --dry-run
```

## Metrics

The local metric script uses `sklearn.metrics`:

- Accuracy: exact match percentage between predictions and ground truth.
- Precision, recall, and F1: macro average with `zero_division=0`.

Example:

```shell
python llm_frag_evaluation/scripts/evaluate_predictions.py \
  --config llm_frag_evaluation/configs/pubmed_config.json \
  --input-file source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json \
  --dataset medqa \
  --retriever bm25 \
  --experiment frag \
  --llm Meta-Llama-3-70B-Instruct
```

Install the metric dependency first if needed:

```shell
pip install -r llm_frag_evaluation/requirements.txt
```

## CINECA

The `slurm/run_experiment.slurm` file is a template. Before using it on CINECA, set the account, partition, module loading commands, virtual environment path, and input/output paths required by the allocation.

## Open Questions

These should be confirmed before implementing the final model execution layer:

- Exact input JSON schema produced by the retrieval and factuality scoring pipeline.
- Exact LLMs to run and whether they use Hugging Face, vLLM, llama.cpp, or an API-compatible server.
- CINECA environment details: partition, GPUs, module stack, storage path, and job array strategy.
