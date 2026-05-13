# Input Files

Place JSON input files here during local development.

Step 2 inputs are split by retrieval resource:

- `source_collection_wiki/`: Wikipedia-backed retrieval inputs. The completed Llama 3.1 70B evaluation used this collection.
- `source_collection_pubmed/`: PubMed-backed retrieval inputs.

Each file may contain one question object or a list of question objects. Each question should include the question text, options when available, and the 100 topically retrieved passages with factuality scores.

Large datasets should not be committed unless they are intentionally small fixtures.
