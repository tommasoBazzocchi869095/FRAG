# FRAG: Factuality-aware Retrieval-Augmented Generation

> A multidimensional RAG framework designed to mitigate medical misinformation in LLM-based Question Answering by retrieving knowledge based on both topical relevance and factual reliability.

## Overview
Large Language Models (LLMs) are increasingly used for medical question answering, but they are prone to hallucinations and can easily propagate misinformation if fed with unreliable retrieved context. 
**FRAG** addresses this critical issue by introducing a dual-filtering retrieval pipeline. Before augmenting the LLM's prompt, the retrieved medical documents are evaluated not just for semantic similarity to the user's query, but for their **scientific factuality**.

This repository contains the full source code for the FRAG pipeline, the factuality classifiers, and the evaluation scripts used for my Master's Thesis.

## Key Features
* **Multidimensional Retrieval:** Combines traditional semantic search with a dedicated factuality estimation step.
* **Dual Factuality Approaches:** Implements two distinct factuality estimation strategies (Source-Based and Claim-Based) using fine-tuned PubMedBERT models.
* **Safe Medical QA:** Acts as a safeguard, ensuring the generative model only relies on verified, scientifically accurate context.
* **MIRAGE Benchmark Evaluation:** Includes scripts to evaluate the pipeline's performance against standard medical benchmarks.

## Factuality Classifiers (Hugging Face)
As part of this project, two custom factuality classifiers were fine-tuned on the Monant Medical Misinformation Dataset. They are publicly available on Hugging Face:

1. **[Source-Based Classifier](https://huggingface.co/tommibazzo01/factuality-classifier-pubmedbert-SourceBased):** Evaluates factuality by proxy, identifying linguistic and structural patterns associated with authoritative vs. unreliable medical publishers.
2. **[Claim-Based Classifier](https://huggingface.co/tommibazzo01/factuality-classifier-pubmedbert-ClaimBased):** Evaluates factuality through fine-grained content verification, assessing the article's stance toward specific medical claims.

