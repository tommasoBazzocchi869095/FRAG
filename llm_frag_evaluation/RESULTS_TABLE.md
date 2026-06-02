# Llama 3.1 70B Results Table

## Scope

The filled Wikipedia-resource results are kept separate from PubMed-resource results. PubMed rows should be added only after prediction validation and metric computation are complete for each run.

Current PubMed-resource state as of May 28, 2026:

| Dataset | Retriever | Experiment | Status |
|---|---|---|---|
| BioASQ | BM25 | zero_shot | Complete; metrics recorded |
| MedQA | BM25 | zero_shot | Complete; metrics recorded from retrieval-independent zero-shot baseline |
| MedQA | BM25 | standard_rag | Complete with 9 missing predictions; metrics recorded |
| MedQA | BM25 | frag | Complete with 11 missing predictions; metrics recorded |
| MedQA | Contriever | standard_rag | Complete; metrics recorded |
| MedQA | Contriever | frag | Complete; metrics recorded |
| BioASQ | BM25/Contriever | RAG/FRAG | Complete; metrics recorded |
| MMLU | BM25/Contriever | zero_shot/RAG/FRAG | Complete; metrics recorded; BM25 RAG and BM25 FRAG each have 4 missing predictions |
| PubMedQA | BM25/Contriever | zero_shot/RAG/FRAG | RAG/FRAG complete with 1 missing or invalid prediction each; metrics recorded; zero-shot reused from retrieval-independent baseline |

The current measured PubMed MedQA Contriever FRAG context recommendation is `GENERATE_MAX_MODEL_LEN=22528`, based on a 20280-token max prompt, 1024 generation tokens, and a 512-token buffer. MedQA BM25 still had 9 missing predictions for RAG and 11 for FRAG at this setting; these are accepted as missing/wrong. MMLU BM25 RAG and BM25 FRAG each have 4 missing predictions, counted as incorrect.

## PubMed-Resource Results

Current filled PubMed-resource result:

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
- MMLU / BM25 / standard_rag / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 4`
- Accuracy: `0.8733`
- Precision macro: `0.6982`
- Recall macro: `0.7008`
- F1 macro: `0.6988`
- MMLU / BM25 / frag / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 4`
- Accuracy: `0.8724`
- Precision macro: `0.6973`
- Recall macro: `0.6992`
- F1 macro: `0.6978`
- MMLU / Contriever / standard_rag / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 0`
- Accuracy: `0.8843`
- Precision macro: `0.8811`
- Recall macro: `0.8865`
- F1 macro: `0.8833`
- MMLU / Contriever / frag / Llama-3.1-70B-Instruct
- `n = 1089`
- `missing_predictions = 0`
- Accuracy: `0.8825`
- Precision macro: `0.8792`
- Recall macro: `0.8850`
- F1 macro: `0.8815`
- BioASQ / BM25 / zero-shot / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.8317152103559871`
- Precision macro: `0.8490629136765833`
- Recall macro: `0.7843900777657944`
- F1 macro: `0.8018278018278018`
- BioASQ / BM25 / standard_rag / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.9239482200647249`
- Precision macro: `0.9218328840970351`
- Recall macro: `0.9121927683487541`
- F1 macro: `0.9166386844031168`
- BioASQ / BM25 / frag / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.9239482200647249`
- Precision macro: `0.9227591036414566`
- Recall macro: `0.9112164386671964`
- F1 macro: `0.9164622273863312`
- BioASQ / Contriever / standard_rag / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.8996763754045307`
- Precision macro: `0.8961289643312549`
- Recall macro: `0.8844184594425839`
- F1 macro: `0.8896835268103924`
- BioASQ / Contriever / frag / Llama-3.1-70B-Instruct
- `n = 618`
- `missing_predictions = 0`
- Accuracy: `0.8980582524271845`
- Precision macro: `0.8956310679611651`
- Recall macro: `0.8811999772946586`
- F1 macro: `0.887539969324356`
- MedQA / BM25 / standard_rag / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 9`
- Accuracy: `0.769835035349568`
- Precision macro: `0.6241293398620343`
- Recall macro: `0.6117566261356446`
- F1 macro: `0.6158745670031489`
- MedQA / BM25 / frag / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 11`
- Accuracy: `0.7659073055773763`
- Precision macro: `0.6221920154452606`
- Recall macro: `0.6083169070446856`
- F1 macro: `0.613082443771767`
- MedQA / Contriever / standard_rag / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 0`
- Accuracy: `0.7674783974862529`
- Precision macro: `0.7708989887374971`
- Recall macro: `0.7618220600196907`
- F1 macro: `0.7642955317606197`
- MedQA / Contriever / frag / Llama-3.1-70B-Instruct
- `n = 1273`
- `missing_predictions = 0`
- Accuracy: `0.7729772191673213`
- Precision macro: `0.7767664271616609`
- Recall macro: `0.7664960197660415`
- F1 macro: `0.769308796952899`
- PubMedQA / BM25 / standard_rag / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 1`
- Accuracy: `0.8`
- Precision macro: `0.45155677655677656`
- Recall macro: `0.4479679740549306`
- F1 macro: `0.429724111866969`
- PubMedQA / BM25 / frag / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 1`
- Accuracy: `0.798`
- Precision macro: `0.4021291208791209`
- Recall macro: `0.44399601234885516`
- F1 macro: `0.42196493982208266`
- PubMedQA / Contriever / standard_rag / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 1`
- Accuracy: `0.796`
- Precision macro: `0.43560470387857075`
- Recall macro: `0.4467298726914112`
- F1 macro: `0.4280397622177209`
- PubMedQA / Contriever / frag / Llama-3.1-70B-Instruct
- `n = 500`
- `missing_predictions = 1`
- Accuracy: `0.798`
- Precision macro: `0.4020469886911545`
- Recall macro: `0.4445695051882343`
- F1 macro: `0.42224032566864567`

Values in the PubMed-resource table are reported as percentages.

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
\textbf{PubMed}
& P & R & F1 & Acc
& P & R & F1 & Acc
& P & R & F1 & Acc
& P & R & F1 & Acc
& P & R & F1 & Acc \\
\midrule
Zero Shot
& 86.80 & 87.16 & 86.96 & 87.14
& 78.80 & 77.89 & 78.12 & 78.40
& 30.09 & 30.89 & 29.18 & 60.60
& 84.91 & 78.44 & 80.18 & 83.17
& 70.15 & 68.59 & 68.61 & 77.33 \\ \hline

BM25 RAG
& 69.82 & 70.08 & 69.88 & 87.33
& 62.41 & 61.18 & 61.59 & 76.98
& 45.16 & 44.80 & 42.97 & 80.00
& 92.18 & 91.22 & 91.66 & 92.39
& 67.39 & 66.82 & 66.53 & 84.18 \\

BM25 FRAG
& 69.73 & 69.92 & 69.78 & 87.24
& 62.22 & 60.83 & 61.31 & 76.59
& 40.21 & 44.40 & 42.20 & 79.80
& 92.28 & 91.12 & 91.65 & 92.39
& 66.11 & 66.57 & 66.24 & 84.01 \\ \hline

Contriever RAG
& 88.11 & 88.65 & 88.33 & 88.43
& 77.09 & 76.18 & 76.43 & 76.75
& 43.56 & 44.67 & 42.80 & 79.60
& 89.61 & 88.44 & 88.97 & 89.97
& 74.59 & 74.49 & 74.13 & 83.69 \\

Contriever FRAG
& 87.92 & 88.50 & 88.15 & 88.25
& 77.68 & 76.65 & 76.93 & 77.30
& 40.20 & 44.46 & 42.22 & 79.80
& 89.56 & 88.12 & 88.75 & 89.81
& 73.84 & 74.43 & 74.01 & 83.79 \\
\bottomrule
\end{tabular}
\caption{PubMed-resource evaluation results for Llama-3.1-70B-Instruct on the MIRAGE medical question answering benchmarks. We report macro precision (P), macro recall (R), macro F1 (F1), and accuracy (Acc) for zero-shot prompting, standard retrieval-augmented generation (RAG), and factuality-aware retrieval-augmented generation (FRAG). RAG and FRAG are evaluated with BM25 and Contriever retrieval over PubMed passages. MedQA BM25 RAG and BM25 FRAG include 9 and 11 missing predictions, respectively; MMLU BM25 RAG and BM25 FRAG include 4 missing predictions each; PubMedQA RAG/FRAG runs include 1 missing or invalid prediction each. Missing predictions are counted as incorrect. The Average columns report the arithmetic mean across MMLU, MedQA, PubMedQA, and BioASQ.}
\label{tab:llama31_70b_pubmed_frag_results}
\end{table*}
```

## Wikipedia-Resource Results

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
