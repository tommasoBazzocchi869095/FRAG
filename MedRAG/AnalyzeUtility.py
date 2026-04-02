import json
import os
import numpy as np
from tqdm import tqdm

DATASET_NAMES = ["mmlu", "medqa", "pubmedqa", "bioasq"]
K_TARGET = 32
ALPHA = 0.5
CACHE_DIR = "."

def analyze_dataset(dataset_name):
    input_file = os.path.join(CACHE_DIR, f"cache_step2_{dataset_name}_scored.json")

    if not os.path.exists(input_file):
        print(f" SALTO {dataset_name}: File non trovato.")
        return None  # Ritorna None se fallisce

    # Caricamento dati (usiamo un print minimale per non sporcare l'output)
    with open(input_file, "r") as f:
        data = json.load(f)

    overlaps = []

    # Disabilito output tqdm per pulizia, o puoi lasciarlo se il dataset è grande
    for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}", leave=False)):
        snippets = item["snippets"]

        if len(snippets) <= K_TARGET:
            overlaps.append(len(snippets))
            continue

        # 1. Ranking Originale (BM25)
        bm25_top_snippets = snippets[:K_TARGET]
        bm25_ids = set(s['id'] for s in bm25_top_snippets)

        # 2. Calcolo Normalizzazione
        topic_scores = [s.get("score_topic", 0) for s in snippets]
        min_s = min(topic_scores) if topic_scores else 0
        max_s = max(topic_scores) if topic_scores else 1
        div = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0

        # 3. Re-Ranking
        scored_snippets = []
        for s in snippets:
            norm_topic = (s.get("score_topic", 0) - min_s) / div
            fact_score = s.get("score_factuality", 0)
            final_score = (ALPHA * norm_topic) + ((1 - ALPHA) * fact_score)

            # Non serve copiare tutto se dobbiamo solo ordinare,
            # ma per sicurezza facciamo una shallow copy leggera
            s_wrapper = {"id": s["id"], "score_final": final_score}
            scored_snippets.append(s_wrapper)

        scored_snippets.sort(key=lambda x: x["score_final"], reverse=True)
        reranked_top_snippets = scored_snippets[:K_TARGET]
        reranked_ids = set(s['id'] for s in reranked_top_snippets)

        # 4. Calcolo Overlap
        intersection = bm25_ids.intersection(reranked_ids)
        overlaps.append(len(intersection))

    # --- CALCOLO STATISTICHE ---
    avg_overlap = np.mean(overlaps)
    avg_new_docs = K_TARGET - avg_overlap
    change_percentage = (avg_new_docs / K_TARGET) * 100

    # Ritorna un dizionario con i risultati invece di stampare
    return {
        "dataset": dataset_name,
        "avg_overlap": avg_overlap,
        "avg_new": avg_new_docs,
        "pct_change": change_percentage
    }

def print_results_table(results):
    """Stampa una tabella formattata in ASCII."""
    if not results:
        print("Nessun risultato da mostrare.")
        return

    # Definizione larghezza colonne
    col_ds = 15
    col_over = 15
    col_new = 15
    col_pct = 15

    # Intestazione
    header = f"| {'DATASET':<{col_ds}} | {'AVG OVERLAP':<{col_over}} | {'AVG NEW DOCS':<{col_new}} | {'% CHANGED':<{col_pct}} |"
    divider = "-" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    for row in results:
        print(f"| {row['dataset'].upper():<{col_ds}} | "
              f"{row['avg_overlap']:<{col_over}.2f} | "
              f"{row['avg_new']:<{col_new}.2f} | "
              f"{row['pct_change']:<{col_pct-1}.1f}% |")

    print(divider + "\n")

if __name__ == "__main__":
    print(f"=== ANALISI IMPATTO RE-RANKING (Alpha={ALPHA} | K={K_TARGET}) ===\n")

    summary_results = []

    for ds_name in DATASET_NAMES:
        # Analizza e raccogli il risultato
        res = analyze_dataset(ds_name)
        if res:
            summary_results.append(res)

    # Stampa tabella finale
    print_results_table(summary_results)