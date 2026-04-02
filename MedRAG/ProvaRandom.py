import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURAZIONE ---
DATASET_NAMES = ["mmlu", "medqa", "pubmedqa", "bioasq"]
K_TARGET = 32
ALPHA = 0  # 0 = Solo Fattualità (Massimo cambiamento teorico), 0.6 = Misto
CACHE_DIR = "."

def get_changes_list(dataset_name):
    input_file = os.path.join(CACHE_DIR, f"cache_step2_{dataset_name}_scored.json")

    if not os.path.exists(input_file):
        print(f"[ATTENZIONE] File non trovato: {input_file}")
        return None

    with open(input_file, "r") as f:
        data = json.load(f)

    per_question_results = []

    # Barra di caricamento
    for idx, item in enumerate(tqdm(data, desc=f"Analisi {dataset_name.upper()}", leave=False)):
        snippets = item["snippets"]

        # Se ci sono meno documenti del target, il cambiamento è 0
        if len(snippets) <= K_TARGET:
            per_question_results.append({
                "Query_ID": idx,
                "Overlap": len(snippets),
                "New_Docs": 0
            })
            continue

        # 1. Ranking Originale (BM25)
        bm25_ids = set(s['id'] for s in snippets[:K_TARGET])

        # 2. Calcolo Score e Re-Ranking
        topic_scores = [s.get("score_topic", 0) for s in snippets]
        min_s = min(topic_scores) if topic_scores else 0
        max_s = max(topic_scores) if topic_scores else 1
        div = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0

        scored_snippets = []
        for s in snippets:
            norm_topic = (s.get("score_topic", 0) - min_s) / div
            fact_score = s.get("score_factuality", 0)
            final_score = (ALPHA * norm_topic) + ((1 - ALPHA) * fact_score)
            scored_snippets.append((s['id'], final_score))

        scored_snippets.sort(key=lambda x: x[1], reverse=True)
        reranked_ids = set(x[0] for x in scored_snippets[:K_TARGET])

        # 3. Calcolo Differenze
        overlap_count = len(bm25_ids.intersection(reranked_ids))
        new_docs_count = K_TARGET - overlap_count

        per_question_results.append({
            "Query_ID": idx,
            "Overlap": overlap_count,
            "New_Docs": new_docs_count
        })

    return pd.DataFrame(per_question_results)

if __name__ == "__main__":
    print(f"=== ANALISI CAMBIAMENTI (SOLO CONSOLE) | Alpha={ALPHA} | K={K_TARGET} ===\n")

    for ds_name in DATASET_NAMES:
        print(f"DATASET: {ds_name.upper()}")
        print("-" * 60)

        df = get_changes_list(ds_name)

        if df is not None:
            # 1. Statistiche Generali
            total = len(df)
            changed = len(df[df['New_Docs'] > 0])
            pct = (changed / total) * 100
            avg_new = df['New_Docs'].mean()

            print(f"  Totale Domande:     {total}")
            print(f"  Domande Modificate: {changed} ({pct:.1f}%)")
            print(f"  Media Nuovi Docs:   {avg_new:.2f}")

            # 2. Distribuzione (Frequenza)
            print("\n  [Distribuzione: Quanti documenti cambiano?]")
            dist = df['New_Docs'].value_counts().sort_index()
            # Stampa formattata della distribuzione
            for k, v in dist.items():
                print(f"    - {k} docs nuovi: {v} domande")

            # 3. Top 10 Esempi (quelli che cambiano di più)
            print("\n  [Top 10 Domande con più cambiamenti]")
            top_changes = df.sort_values(by="New_Docs", ascending=False).head(10)

            # Stampiamo il dataframe pulito senza indice
            print(top_changes.to_string(index=False))
            print("\n" + "=" * 60 + "\n")

        else:
            print(f" Errore analisi {ds_name}\n")