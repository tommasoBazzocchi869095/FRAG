import json
import os
import random
import torch
from tqdm import tqdm
from datasets import load_dataset
from src.medrag import MedRAG
import importlib.util


VALIDATION_SIZE = 33
RANDOM_SEED = 42
K = 100
CORPUS_NAME = "Wikipedia"
RETRIEVER_NAME = "BM25"
OUTPUT_PREFIX = "validation_set_external"

random.seed(RANDOM_SEED)
medrag = MedRAG(
    llm_name="openai/gpt-3.5-turbo", # Dummy, non serve LLM qui
    rag=True,
    retriever_name=RETRIEVER_NAME,
    corpus_name=CORPUS_NAME,
    corpus_cache=True,
    factuality=False
)

def format_medqa(example, idx):
    return {
        "id": f"medqa_val_{idx}",
        "question": example["question"],
        "options": example["options"],
        "answer": example["answer_idx"],
        "dataset": "medqa"
    }

def format_medmcqa(example, idx):
    options = {
        "A": example["opa"],
        "B": example["opb"],
        "C": example["opc"],
        "D": example["opd"]
    }
    map_ans = {0: "A", 1: "B", 2: "C", 3: "D"}
    return {
        "id": f"medmcqa_val_{idx}",
        "question": example["question"],
        "options": options,
        "answer": map_ans.get(example["cop"], "A"),
        "dataset": "medmcqa"
    }

def format_pubmedqa(example, idx):
    return {
        "id": f"pubmedqa_val_{idx}",
        "question": example["question"],
        "options": {}, # Non ha opzioni A/B/C
        "answer": example["final_decision"],
        "dataset": "pubmedqa"
    }

datasets_config = [
    {
        "name": "medqa",
        "hf_path": "GBaker/MedQA-USMLE-4-options",
        "split": "train",
        "formatter": format_medqa
    },
    {
        "name": "medmcqa",
        "hf_path": "medmcqa",
        "split": "validation",
        "formatter": format_medmcqa
    },
    {
        "name": "pubmedqa",
        "hf_path": "pubmed_qa",
        "subset": "pqa_labeled",
        "split": "train",
        "formatter": format_pubmedqa
    }

]

combined_results = []

for conf in datasets_config:
    print(f"\nScaricamento {conf['name']} (split: {conf['split']})...")
    try:
        if "subset" in conf:
            ds = load_dataset(conf["hf_path"], conf["subset"], split=conf["split"])
        else:
            ds = load_dataset(conf["hf_path"], split=conf["split"])

        ds_list = list(ds)
        if len(ds_list) > VALIDATION_SIZE:
            sampled = random.sample(ds_list, VALIDATION_SIZE)
        else:
            sampled = ds_list

        print(f"Esecuzione Retrieval su {len(sampled)} domande...")

        dataset_results = []
        for idx, item in enumerate(tqdm(sampled)):
            formatted = conf["formatter"](item, idx)
            snippets, scores = medrag.retrieval_system.retrieve(formatted["question"], k=K)
            formatted["snippets"] = snippets
            formatted["scores_topic"] = scores
            dataset_results.append(formatted)

        with open(f"{OUTPUT_PREFIX}_{conf['name']}.json", "w") as f:
            json.dump(dataset_results, f, indent=4)

        combined_results.extend(dataset_results)

    except Exception as e:
        print(f"Errore con {conf['name']}: {e}")
