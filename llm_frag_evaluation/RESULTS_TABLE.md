# Llama 3.1 70B Results Table

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
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & -- \\

BM25 FRAG
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & -- \\

Contriever RAG
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & -- \\

Contriever FRAG
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & --
& -- & -- & -- & -- \\
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
