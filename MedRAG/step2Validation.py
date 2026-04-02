import json
import os
from tqdm import tqdm
from src.factuality import FactualityScorer

MODEL_TYPE = "source"
MODEL_PATH = "../Model/estimationFactualityOne/modello_finale_bertSourceBased"
DATASET_NAMES = ["medqa", "medmcqa", "pubmedqa"]
INPUT_PREFIX = "validation_set_external"

print(f"Caricamento Modello {MODEL_TYPE} da {MODEL_PATH}...")
scorer = FactualityScorer(model_path=MODEL_PATH)

for name in DATASET_NAMES:
    input_file = f"{INPUT_PREFIX}_{name}.json"
    output_file = f"{INPUT_PREFIX}_{name}_scored_{MODEL_TYPE}.json"

    if not os.path.exists(input_file):
        continue

    print(f"Scoring {name}...")
    with open(input_file, "r") as f:
        data = json.load(f)

    for entry in tqdm(data):
        snippets = entry["snippets"]
        texts = [s.get("contents", s.get("content", "")) for s in snippets]

        if texts:
            scores = scorer.predict(texts)
            for i, s in enumerate(snippets):
                s[f"score_factuality_{MODEL_TYPE}"] = float(scores[i])
                if "score_topic" not in s:
                    s["score_topic"] = float(entry["scores_topic"][i])
        else:
            for s in snippets:
                s[f"score_factuality_{MODEL_TYPE}"] = 0.0

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

print(f"Finito scoring {MODEL_TYPE}.")