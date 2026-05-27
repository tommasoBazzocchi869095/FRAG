# Llama 3.1 70B Results Table

## Scope

The filled results below are for the completed Wikipedia-resource campaign. PubMed-resource results are not finalized yet and should not be mixed into this table until prediction validation and metric computation are complete.

Current PubMed-resource state as of May 27, 2026:

| Dataset | Retriever | Experiment | Status |
|---|---|---|---|
| BioASQ | BM25 | zero_shot | Complete; metrics pending |
| MedQA | BM25 | zero_shot | Complete; metrics pending |
| MedQA | BM25 | standard_rag | First run incomplete at `GENERATE_MAX_MODEL_LEN=12288`; rerun submitted with measured context setting |
| MedQA | BM25 | frag | First run incomplete at `GENERATE_MAX_MODEL_LEN=12288`; rerun submitted with measured context setting |
| MedQA | Contriever | standard_rag | First run incomplete at `GENERATE_MAX_MODEL_LEN=12288`; rerun submitted with measured context setting |
| MedQA | Contriever | frag | First run incomplete at `GENERATE_MAX_MODEL_LEN=12288`; rerun submitted with measured context setting |
| BioASQ | BM25/Contriever | RAG/FRAG | Submitted; metrics pending |
| MMLU | BM25/Contriever | zero_shot/RAG/FRAG | Not run or not located |
| PubMedQA | BM25/Contriever | zero_shot/RAG/FRAG | Not run or not located |

The current measured PubMed MedQA Contriever FRAG context recommendation is `GENERATE_MAX_MODEL_LEN=22528`, based on a 20280-token max prompt, 1024 generation tokens, and a 512-token buffer. Add PubMed metric rows only after full reruns are complete and validated.

Current filled result:

- MMLU / BM25 / zero-shot / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 0`
- Accuracy: `0.8714416896235078`
- Precision macro: `0.8680315537269185`
- Recall macro: `0.8715694678185778`
- F1 macro: `0.8695903739632252`
- MedQA / BM25 / zero-shot / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 0`
- Accuracy: `0.783974862529458`
- Precision macro: `0.7880424652768914`
- Recall macro: `0.7788646197680493`
- F1 macro: `0.781184669047834`
- PubMedQA / BM25 / zero-shot / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 2`
- Accuracy: `0.606`
- Precision macro: `0.3008562918838421`
- Recall macro: `0.30886609210187804`
- F1 macro: `0.2917741323463905`
- BioASQ / BM25 / zero-shot / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.8317152103559871`
- Precision macro: `0.8490629136765833`
- Recall macro: `0.7843900777657944`
- F1 macro: `0.8018278018278018`
- MMLU / BM25 / standard_rag / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 0`
- Accuracy: `0.8741965105601469`
- Precision macro: `0.8701696766593894`
- Recall macro: `0.8744992663401672`
- F1 macro: `0.8719375232525679`
- MMLU / BM25 / frag / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 0`
- Accuracy: `0.8741965105601469`
- Precision macro: `0.8703106713348145`
- Recall macro: `0.8747175257648387`
- F1 macro: `0.8721917316654304`
- MMLU / Contriever / standard_rag / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 0`
- Accuracy: `0.8741965105601469`
- Precision macro: `0.8706163153786104`
- Recall macro: `0.8746059952226287`
- F1 macro: `0.8722962614469405`
- MMLU / Contriever / frag / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 0`
- Accuracy: `0.8778696051423324`
- Precision macro: `0.8745175684481137`
- Recall macro: `0.8793535984920131`
- F1 macro: `0.8764973731048527`
- MedQA / BM25 / standard_rag / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 0`
- Accuracy: `0.7737627651217597`
- Precision macro: `0.7748720938022104`
- Recall macro: `0.7676602688662705`
- F1 macro: `0.7695761832294051`
- MedQA / BM25 / frag / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 0`
- Accuracy: `0.7706205813040062`
- Precision macro: `0.7715135999541051`
- Recall macro: `0.7656099729805018`
- F1 macro: `0.76727522609773`
- MedQA / Contriever / standard_rag / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 0`
- Accuracy: `0.7611940298507462`
- Precision macro: `0.7651032815566858`
- Recall macro: `0.7551391412122571`
- F1 macro: `0.7575253356054776`
- MedQA / Contriever / frag / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 0`
- Accuracy: `0.7635506677140613`
- Precision macro: `0.7680219785252502`
- Recall macro: `0.757768018060722`
- F1 macro: `0.7600966356551042`
- PubMedQA / BM25 / standard_rag / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 1`
- Accuracy: `0.546`
- Precision macro: `0.2880992588422929`
- Recall macro: `0.29336073625371284`
- F1 macro: `0.28362965978802757`
- PubMedQA / BM25 / frag / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 1`
- Accuracy: `0.548`
- Precision macro: `0.29413807189542485`
- Recall macro: `0.2959870118732995`
- F1 macro: `0.2858113517352656`
- PubMedQA / Contriever / standard_rag / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 1`
- Accuracy: `0.55`
- Precision macro: `0.2699662594316012`
- Recall macro: `0.29440013720950176`
- F1 macro: `0.28116553382621834`
- PubMedQA / Contriever / frag / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 1`
- Accuracy: `0.556`
- Precision macro: `0.27233110466184196`
- Recall macro: `0.2982645141926078`
- F1 macro: `0.28415697674418605`
- BioASQ / BM25 / standard_rag / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.7928802588996764`
- Precision macro: `0.7795507078109521`
- Recall macro: `0.7979451666004428`
- F1 macro: `0.7838971938106478`
- BioASQ / BM25 / frag / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.7944983818770227`
- Precision macro: `0.7805664730006836`
- Recall macro: `0.7982346597036953`
- F1 macro: `0.7851364558438252`
- BioASQ / Contriever / standard_rag / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.8058252427184466`
- Precision macro: `0.7968073336494389`
- Recall macro: `0.8197877050576148`
- F1 macro: `0.7994721813619452`
- BioASQ / Contriever / frag / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.7928802588996764`
- Precision macro: `0.7850674687897687`
- Recall macro: `0.8077084634160187`
- F1 macro: `0.7866022099447514`

Values in the LaTeX table are reported as percentages.

```latex
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccccccccccccccc}
\toprule
& \multicolumn{4}{c}{MMLU}
& \multicolumn{4}{c}{MedQA}
& \multicolumn{4}{c}{PubMedQA}
& \multicolumn{4}{c}{BioASQ}
& \multicolumn{4}{c}{Average} \\
\cmidrule(lr){2-5}
\cmidrule(lr){6-9}
\cmidrule(lr){10-13}
\cmidrule(lr){14-17}
\cmidrule(lr){18-21}
\textbf{Llama-3.1-70B-Instruct}
& P & R & F1 & Acc
& P & R & F1 & Acc
& P & R & F1 & Acc
& P & R & F1 & Acc
& P & R & F1 & Acc \\
\midrule
Zero-shot
& 86.80 & 87.16 & 86.96 & 87.14
& 78.80 & 77.89 & 78.12 & 78.40
& 30.09 & 30.89 & 29.18 & 60.60
& 84.91 & 78.44 & 80.18 & 83.17
& 70.15 & 68.59 & 68.61 & 77.33 \\

BM25 RAG
& 87.02 & 87.45 & 87.19 & 87.42
& 77.49 & 76.77 & 76.96 & 77.38
& 28.81 & 29.34 & 28.36 & 54.60
& 77.96 & 79.79 & 78.39 & 79.29
& 67.82 & 68.34 & 67.73 & 74.67 \\

BM25 FRAG
& 87.03 & 87.47 & 87.22 & 87.42
& 77.15 & 76.56 & 76.73 & 77.06
& 29.41 & 29.60 & 28.58 & 54.80
& 78.06 & 79.82 & 78.51 & 79.45
& 67.91 & 68.36 & 67.76 & 74.68 \\

Contriever RAG
& 87.06 & 87.46 & 87.23 & 87.42
& 76.51 & 75.51 & 75.75 & 76.12
& 27.00 & 29.44 & 28.12 & 55.00
& 79.68 & 81.98 & 79.95 & 80.58
& 67.56 & 68.60 & 67.76 & 74.78 \\

Contriever FRAG
& 87.45 & 87.94 & 87.65 & 87.79
& 76.80 & 75.78 & 76.01 & 76.36
& 27.23 & 29.83 & 28.42 & 55.60
& 78.51 & 80.77 & 78.66 & 79.29
& 67.50 & 68.58 & 67.68 & 74.76 \\
\bottomrule
\end{tabular}
\caption{Evaluation results for Llama-3.1-70B-Instruct on the MIRAGE medical question answering benchmarks. We report macro precision (P), macro recall (R), macro F1 (F1), and accuracy (Acc) for zero-shot prompting, standard retrieval-augmented generation (RAG), and factuality-aware retrieval-augmented generation (FRAG). RAG and FRAG are evaluated with BM25 and Contriever retrieval over Wikipedia passages. The Average columns report the arithmetic mean across MMLU, MedQA, PubMedQA, and BioASQ.}
\label{tab:llama31_70b_frag_results}
\end{table*}
```

## Metric Mapping

For each completed experiment, fill columns in this order:

```text
P   = precision_macro * 100
R   = recall_macro * 100
F1  = f1_macro * 100
Acc = accuracy * 100
```
