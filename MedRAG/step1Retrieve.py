import json
import os
from tqdm import tqdm
from src.medrag import MedRAG
import importlib.util

DATASET_NAMES = ["mmlu", "medqa",  "pubmedqa", "bioasq"]
K = 100
CORPUS_NAME = "Wikipedia"
RETRIEVER_NAME = "BM25"

mirage_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MIRAGE/src/utils.py"))
spec = importlib.util.spec_from_file_location("mirage_utils", mirage_utils_path)
mirage_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirage_utils)
QADataset = mirage_utils.QADataset
print("Inizializzazione Sistema di Retrieval...")
medrag = MedRAG(
    llm_name="openai/gpt-3.5-turbo", # Dummy name, non usa GPU
    rag=True,
    retriever_name=RETRIEVER_NAME,
    corpus_name=CORPUS_NAME,
    corpus_cache=True,
    factuality=False
)

for dataset_name in DATASET_NAMES:
    print(f"\n=== Avvio Retrieval per dataset: {dataset_name} ===")
    dataset = QADataset(dataset_name, dir="../MIRAGE")

    output_file = f"cache_step1_{dataset_name}.json"
    results = []
    if os.path.exists(output_file):
        print(f"File {output_file} esistente. Sovrascrittura...")

    for idx, item in enumerate(tqdm(dataset)):
        question = item["question"]
        snippets, scores = medrag.retrieval_system.retrieve(question, k=K)
        entry = {
            "id": idx, # Indice numerico per allineamento
            "question": question,
            "options": item.get("options", {}),
            "answer": item.get("answer", ""), # Ground truth (opzionale ma utile)
            "snippets": snippets, # Lista di dizionari {'title', 'content', 'id'}
            "scores_topic": scores # Score di rilevanza del retriever (BM25/MedCPT)
        }
        results.append(entry)

    print(f"Salvataggio {len(results)} elementi in {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

print("\nStep 1 Completato! Tutti i documenti sono stati scaricati.")