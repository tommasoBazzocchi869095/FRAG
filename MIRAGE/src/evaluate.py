import os
import json
import argparse
import re
from utils import QADataset, locate_answer, locate_answer4pub_llama
# AGGIUNTO: Import delle metriche
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import statistics

def evaluate(dataset, save_dir, split="test", locate_fun=locate_answer, dataset_name=""):

    # 1. IMPOSTIAMO SEMPRE LE LETTERE COME RIFERIMENTO
    # Dato che il benchmark.json usa "A", "B", "C" anche per PubMedQA,
    # forziamo l'uso delle lettere come indici.
    answer_list = ["A", "B", "C", "D", "E"]
    answer2idx = {ans.lower(): i for i, ans in enumerate(answer_list)}

    # Mappa di conversione esplicita per recuperare il tuo output attuale
    # A=yes, B=no, C=maybe (come definito nel tuo benchmark.json)
    yn_map = {
        "yes": "A",
        "no": "B",
        "maybe": "C"
    }

    flag = False
    pred = []

    # --- Ciclo sulle domande ---
    for q_idx in range(len(dataset)):
        filename = split + "_" + str(q_idx) + ".json"
        fpath = os.path.join(save_dir, filename)

        if not os.path.exists(fpath):
            pred.append(-1)
            continue

        try:
            data = json.load(open(fpath))
            if not data:
                pred.append(-1)
                continue

            item = data[0] if isinstance(data, list) else data

            # Estrazione della risposta grezza (es. "yes", "A", "Option A")
            if isinstance(item, dict):
                raw_ans = str(item.get("answer_choice", "NA")).strip()
            else:
                raw_ans = str(item).strip()

            # --- MODIFICA CRUCIALE: CLEANING IBRIDO ---

            # 1. Pulizia base (rimuove punteggiatura e spazi)
            raw_clean = re.sub(r'[^\w\s]', '', raw_ans).lower()

            # 2. Controllo se è una risposta Yes/No/Maybe
            if raw_clean in yn_map:
                # Se il modello ha scritto "yes", lo convertiamo in "A"
                clean_ans = yn_map[raw_clean]
            else:
                # Altrimenti usiamo la regex standard per cercare "A", "B", ecc.
                clean_ans = locate_fun(raw_ans)

            # ------------------------------------------

        except Exception:
            pred.append(-1)
            continue

        # Mappatura Risposta -> Indice (0, 1, 2...)
        clean_ans_lower = str(clean_ans).lower()
        if clean_ans_lower in answer2idx:
            pred.append(answer2idx[clean_ans_lower])
        else:
            # Fallback finale per le lettere sporche
            if len(clean_ans) > 0 and clean_ans[0].upper() in ["A","B","C","D","E"]:
                 pred.append(answer2idx[clean_ans[0].lower()])
            else:
                 pred.append(-1)

    # --- Calcolo Truth (Etichette vere) ---
    truth = []
    for item in dataset:
        # Il benchmark.json ha già "A", "B", "C" come risposta
        label = str(item['answer']).strip().lower() # .strip() per sicurezza

        # Gestione caso raro in cui ground truth fosse "yes" (non dovrebbe succedere col tuo json)
        if label in yn_map:
            label = yn_map[label].lower()

        if label in answer2idx:
            truth.append(answer2idx[label])
        else:
            truth.append(-1)

    if len(pred) < len(truth):
        truth = truth[:len(pred)]
        flag = True

    pred_array = np.array(pred)
    truth_array = np.array(truth)

    # Calcolo Metriche
    acc = accuracy_score(truth_array, pred_array)
    std = np.sqrt(acc * (1-acc) / len(truth)) if len(truth) > 0 else 0

    if len(truth) > 0:
        prec = precision_score(truth_array, pred_array, average='macro', zero_division=0)
        rec = recall_score(truth_array, pred_array, average='macro', zero_division=0)
        f1 = f1_score(truth_array, pred_array, average='macro', zero_division=0)
    else:
        prec, rec, f1 = 0.0, 0.0, 0.0

    return acc, std, flag, prec, rec, f1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--corpus_name", type=str, default="Wikipedia")
    parser.add_argument("--retriever_name", type=str, default="BM25")
    parser.add_argument("--results_dir", type=str, default="./prediction")

    args = parser.parse_args()

    # Caricamento Dataset
    dataset_names = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']
    try:
        datasets = {key: QADataset(key, dir="../MIRAGE") for key in dataset_names}
    except Exception as e:
        datasets = {key: QADataset(key) for key in dataset_names}

    scores = []
    for dataset_name in dataset_names:
        print("[{:s}] ".format(dataset_name), end="", flush=True)
        split = "test"
        if dataset_name == "medmcqa":
            split = "dev"

        if args.rag:
            save_dir = os.path.join(args.results_dir, dataset_name, "rag_"+str(args.k), args.llm_name, args.corpus_name, args.retriever_name)
        else:
            save_dir = os.path.join(args.results_dir, dataset_name, "cot", args.llm_name)

        if os.path.exists(save_dir):
            if "pmc_llama" in args.llm_name.lower():
                loc_fun = locate_answer4pub_llama
            else:
                loc_fun = locate_answer

            # Chiamata aggiornata con le nuove variabili di ritorno
            acc, std, flag, prec, rec, f1 = evaluate(datasets[dataset_name], save_dir, split, loc_fun, dataset_name)
            scores.append(acc)

            # Stampa aggiornata con Precision, Recall e F1
            print("Acc: {:.4f}; P: {:.4f}; R: {:.4f}; F1: {:.4f}".format(acc, prec, rec, f1), end="")

            if flag:
                print(" (NOT COMPLETED - Some files missing)")
            else:
                print("")
        else:
            print(f"NOT FOUND. (Checked: {save_dir})")

    if len(scores) > 0:
        print("[Average] mean acc: {:.4f}".format(sum(scores) / len(scores)))