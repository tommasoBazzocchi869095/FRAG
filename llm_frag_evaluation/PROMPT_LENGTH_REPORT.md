# Prompt Length Report

Generated on CINECA with:

```bash
python llm_frag_evaluation/scripts/report_prompt_lengths.py \
  --model-path /leonardo_work/IscrC_SpecDLM/models/Llama-3.1-70B-Instruct \
  --fail-over 12288
```

Tokenizer/model:

```text
/leonardo_work/IscrC_SpecDLM/models/Llama-3.1-70B-Instruct
```

## Summary

- Prompt loads checked: 24
- Failure threshold: 12288 tokens
- Prompts over 12288 tokens: 0
- Overall maximum prompt length: 11025 tokens
- Longest prompt: `medqa_contriever_standard_rag_41` / `test_41.json`
- Longest prompt load: `medqa/contriever/standard_rag`

Conclusion: use `GENERATE_MAX_MODEL_LEN=12288` for full RAG and FRAG runs. This covers all generated prompt loads with margin. Zero-shot prompts are much shorter and also fit comfortably.

## Overall Maximum

| Field | Value |
|---|---|
| Prompt load | `llm_frag_evaluation/outputs/prompt_loads/medqa/contriever/standard_rag/Meta-Llama-3-70B-Instruct/prompts.jsonl` |
| Count | 1273 |
| Average tokens | 6625.34 |
| Max tokens | 11025 |
| Max request | `medqa_contriever_standard_rag_41` |
| Max output file | `test_41.json` |
| >4096 | 1272 |
| >8192 | 8 |
| >12288 | 0 |
| >16384 | 0 |

## Per Prompt Load

| Dataset | Retriever | Experiment | Count | Avg Tokens | Max Tokens | Max Request | >4096 | >8192 | >12288 | >16384 |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| bioasq | bm25 | frag | 618 | 6103.40 | 8962 | `bioasq_bm25_frag_490` | 617 | 1 | 0 | 0 |
| bioasq | bm25 | standard_rag | 618 | 6110.71 | 8962 | `bioasq_bm25_standard_rag_490` | 617 | 1 | 0 | 0 |
| bioasq | bm25 | zero_shot | 618 | 209.96 | 245 | `bioasq_bm25_zero_shot_494` | 0 | 0 | 0 | 0 |
| bioasq | contriever | frag | 618 | 5210.57 | 7928 | `bioasq_contriever_frag_490` | 575 | 0 | 0 | 0 |
| bioasq | contriever | standard_rag | 618 | 5245.67 | 7928 | `bioasq_contriever_standard_rag_490` | 581 | 0 | 0 | 0 |
| bioasq | contriever | zero_shot | 618 | 209.96 | 245 | `bioasq_contriever_zero_shot_494` | 0 | 0 | 0 | 0 |
| medqa | bm25 | frag | 1273 | 6821.80 | 8301 | `medqa_bm25_frag_1228` | 1273 | 2 | 0 | 0 |
| medqa | bm25 | standard_rag | 1273 | 6822.38 | 8301 | `medqa_bm25_standard_rag_1228` | 1273 | 2 | 0 | 0 |
| medqa | bm25 | zero_shot | 1273 | 415.36 | 1091 | `medqa_bm25_zero_shot_535` | 0 | 0 | 0 | 0 |
| medqa | contriever | frag | 1273 | 6619.56 | 10927 | `medqa_contriever_frag_41` | 1272 | 10 | 0 | 0 |
| medqa | contriever | standard_rag | 1273 | 6625.34 | 11025 | `medqa_contriever_standard_rag_41` | 1272 | 8 | 0 | 0 |
| medqa | contriever | zero_shot | 1273 | 415.36 | 1091 | `medqa_contriever_zero_shot_535` | 0 | 0 | 0 | 0 |
| mmlu | bm25 | frag | 1089 | 6358.93 | 8132 | `mmlu_bm25_frag_326` | 1089 | 0 | 0 | 0 |
| mmlu | bm25 | standard_rag | 1089 | 6363.11 | 8317 | `mmlu_bm25_standard_rag_236` | 1089 | 1 | 0 | 0 |
| mmlu | bm25 | zero_shot | 1089 | 305.05 | 1176 | `mmlu_bm25_zero_shot_610` | 0 | 0 | 0 | 0 |
| mmlu | contriever | frag | 1089 | 5696.52 | 9250 | `mmlu_contriever_frag_624` | 1075 | 3 | 0 | 0 |
| mmlu | contriever | standard_rag | 1089 | 5707.50 | 9207 | `mmlu_contriever_standard_rag_624` | 1075 | 3 | 0 | 0 |
| mmlu | contriever | zero_shot | 1089 | 305.05 | 1176 | `mmlu_contriever_zero_shot_610` | 0 | 0 | 0 | 0 |
| pubmedqa | bm25 | frag | 500 | 5951.11 | 7163 | `pubmedqa_bm25_frag_140` | 500 | 0 | 0 | 0 |
| pubmedqa | bm25 | standard_rag | 500 | 5952.80 | 7163 | `pubmedqa_bm25_standard_rag_140` | 500 | 0 | 0 | 0 |
| pubmedqa | bm25 | zero_shot | 500 | 217.69 | 252 | `pubmedqa_bm25_zero_shot_427` | 0 | 0 | 0 | 0 |
| pubmedqa | contriever | frag | 500 | 4974.70 | 7524 | `pubmedqa_contriever_frag_105` | 468 | 0 | 0 | 0 |
| pubmedqa | contriever | standard_rag | 500 | 5002.07 | 7524 | `pubmedqa_contriever_standard_rag_105` | 468 | 0 | 0 | 0 |
| pubmedqa | contriever | zero_shot | 500 | 217.69 | 252 | `pubmedqa_contriever_zero_shot_427` | 0 | 0 | 0 | 0 |

## Operational Notes

- `GENERATE_MAX_MODEL_LEN=4096` is not enough for any RAG/FRAG prompt load.
- `GENERATE_MAX_MODEL_LEN=8192` is not enough for some MedQA, MMLU, and BioASQ prompt loads.
- `GENERATE_MAX_MODEL_LEN=12288` is enough for all checked prompt loads.
- With `GENERATE_MAX_TOKENS=1024`, the longest observed prompt plus maximum generation budget is approximately 12049 tokens, still within 12288.
- The updated vLLM runner preflights prompt length and logs oversized prompts to `generation_errors.jsonl` instead of aborting the whole job.
