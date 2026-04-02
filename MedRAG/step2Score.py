import json
import os
import torch
from tqdm import tqdm
from src.factuality import FactualityScorer

DATASET_NAMES = ["mmlu", "medqa",  "pubmedqa", "bioasq"]
MODEL_PATH = "../Model/estimationFactualityBasedOnArticlesVeracity/modello_finale_bertClaimsBased"

print("Caricamento Modello Factuality (BERT) su GPU...")
scorer = FactualityScorer(model_path=MODEL_PATH)
for dataset_name in DATASET_NAMES:
    input_file = f"cache_step1_{dataset_name}.json"
    output_file = f"cache_step2_{dataset_name}_scored.json"

    print(f"\n=== Calcolo Fattualità per dataset: {dataset_name} ===")

    if not os.path.exists(input_file):
        print(f"File {input_file} non trovato. Salto.")
        continue

    with open(input_file, "r") as f:
        data = json.load(f)

    for entry in tqdm(data):
        snippets = entry["snippets"]
        texts_to_score = [s.get("contents", s.get("content", "")) for s in snippets]
        if len(texts_to_score) > 0:
            fact_scores = scorer.predict(texts_to_score)
            for i, snip in enumerate(snippets):
                snip["score_factuality"] = float(fact_scores[i])
                snip["score_topic"] = float(entry["scores_topic"][i])
        else:
            for snip in snippets:
                snip["score_factuality"] = 0.0
                snip["score_topic"] = 0.0

    print(f"Salvataggio risultati arricchiti in {output_file}...")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
