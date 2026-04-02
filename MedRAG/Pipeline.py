import os
import sys
import json
import re
import torch
import gc
from tqdm import tqdm
from src.medrag import MedRAG
import importlib.util

mirage_utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MIRAGE/src/utils.py"))
if not os.path.exists(mirage_utils_path):
    raise FileNotFoundError(f"utils.py non trovato in: {mirage_utils_path}")

spec = importlib.util.spec_from_file_location("mirage_utils", mirage_utils_path)
mirage_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirage_utils)
QADataset = mirage_utils.QADataset

BATCH_SIZE = 1
llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
rag = True
k = 32
corpus_name = "Wikipedia"
retriever_name = "BM25"

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MIRAGE/prediction"))
dataset_names = ["medqa", "medmcqa", "pubmedqa", "bioasq"]

medrag = MedRAG(
    llm_name=llm_name,
    rag=rag,
    retriever_name=retriever_name,
    corpus_name=corpus_name,
    corpus_cache=True,
    HNSW=True,
    factuality=True
)

def parse_answer(raw_answer, dataset_type="mcq", debug=False):
    """
    Estrae answer_choice e step_by_step_thinking.
    Se il JSON non è valido, usa Regex che si fermano alla prima virgoletta chiusa.
    """
    answer_choice = "NA"
    step_by_step = ""
    text = raw_answer.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).replace("```", "")
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return str(data.get("answer_choice", "NA")).strip().upper(), str(data.get("step_by_step_thinking", ""))
    except:
        pass

    try:
        if debug:
            print("[DEBUG] JSON fallito, avvio estrazione Regex...")

        ac_pattern = r'[\'"]answer_choice[\'"]\s*:\s*[\'"](.*?)[\'"]'
        ac_match = re.search(ac_pattern, text, re.IGNORECASE | re.DOTALL)

        if ac_match:
            answer_choice = ac_match.group(1).strip().upper()

        sb_pattern = r'[\'"]step_by_step_thinking[\'"]\s*:\s*[\'"](.*?)[\'"]'
        sb_match = re.search(sb_pattern, text, re.IGNORECASE | re.DOTALL)

        if sb_match:
            step_by_step = sb_match.group(1).strip()

        if debug:
            print(f"[DEBUG] Regex Extracted -> Answer: {answer_choice}")

    except Exception as e:
        if debug:
            print(f"[parse_answer] Errore critico nel regex parsing: {e}")

    if dataset_type == "mcq" and len(answer_choice) > 1:
        if answer_choice[0].isalpha() and answer_choice[1] in ['.', ')', ' ']:
            answer_choice = answer_choice[0]

    return answer_choice, step_by_step

for dataset_name in dataset_names:
    print(f"\n=== Processing dataset: {dataset_name} (BATCH MODE: {BATCH_SIZE}) ===")
    dataset = QADataset(dataset_name, dir="../MIRAGE")

    # Determina cartella di output
    if rag:
        save_dir = os.path.join(
            results_dir,
            dataset_name,
            f"rag_{k}",
            llm_name,
            corpus_name,
            retriever_name,
        )
    else:
        save_dir = os.path.join(results_dir, dataset_name, "cot", llm_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Salvataggio risultati in: {save_dir}")
    if dataset_name in ["mmlu", "medqa", "medmcqa"]:
        question_type = "mcq"
    elif dataset_name == "pubmedqa":
        question_type = "ynm"
    else:
        question_type = "yn"

    split = "dev" if dataset_name == "medmcqa" else "test"

    # --- BUFFER PER IL BATCHING ---
    batch_questions = []
    batch_indices = []

    # Funzione interna per processare il batch corrente
    def process_current_batch():
        if not batch_questions: return

        try:
            # 1. PULIZIA MEMORIA GPU (Cruciale per evitare OOM)
            torch.cuda.empty_cache()
            gc.collect()

            # 2. CHIAMATA AL NUOVO METODO medrag.answer_batch
            # Passiamo options_list=None perché le opzioni sono già incorporate nel testo del prompt (formatted_q)
            raw_answers_list, _, _ = medrag.answer_batch(
                questions=batch_questions,
                options_list=None,
                k=k,
                batch_size=len(batch_questions), alpha=0.5
            )

            # 3. SALVATAGGIO RISULTATI
            for i, raw_answer in enumerate(raw_answers_list):
                original_idx = batch_indices[i]

                # Parsing
                answer_choice, step_by_step = parse_answer(raw_answer, dataset_type=question_type, debug=False)

                # Output JSON
                output_json = [{
                    "answer_choice": answer_choice,
                    "step_by_step_thinking": step_by_step
                }]

                filename = f"{split}_{dataset.index[original_idx]}.json"
                save_path = os.path.join(save_dir, filename)
                with open(save_path, "w") as f:
                    json.dump(output_json, f, indent=2)

        except Exception as e:
            print(f"Errore critico nel processare il batch: {e}")
            # In caso di errore, si potrebbe implementare una logica di retry o logging,
            # qui continuiamo per non bloccare tutto.

    # --- CICLO SULLE DOMANDE ---
    for idx, item in enumerate(tqdm(dataset)):
        # Verifica se il file esiste già (Resume capability)
        filename = f"{split}_{dataset.index[idx]}.json"
        save_path = os.path.join(save_dir, filename)
        if os.path.exists(save_path):
            continue

        question = item["question"]

        # --- TUA LOGICA DI PROMPT (MANTENUTA ORIGINALE) ---
        if question_type == "mcq":
            options = item["options"]
            formatted_q = (
                "You are a medical expert. Read the following question carefully and choose the single best answer.\n\n"
                f"Question:\n{question}\n\n"
                + "\n".join([f"{k}. {v}" for k, v in options.items()])
                + "\n\nAnswer the question and then briefly explain your reasoning.\n"
                + "Your response must be in JSON format with exactly these keys:\n"
                + '{ "answer_choice": "A", "step_by_step_thinking": "your explanation here" }'
            )
        elif question_type == "ynm":
            formatted_q = (
                "You are a biomedical researcher. Read the following question carefully.\n\n"
                f"Question:\n{question}\n\n"
                "Possible answers: yes, no, or maybe.\n"
                "Respond strictly in JSON format as follows:\n"
                '{ "answer_choice": "yes", "step_by_step_thinking": "your short biomedical justification" }'
            )
        else:  # yes/no only (BioASQ)
            formatted_q = (
                "You are a biomedical researcher. Read the following question carefully.\n\n"
                f"Question:\n{question}\n\n"
                "Possible answers: yes or no.\n"
                "Respond strictly in JSON format as follows:\n"
                '{ "answer_choice": "yes", "step_by_step_thinking": "your short biomedical justification" }'
            )
        # --------------------------------------------------

        # Aggiungi al buffer
        batch_questions.append(formatted_q)
        batch_indices.append(idx)

        # Se il buffer è pieno, processa il batch
        if len(batch_questions) >= BATCH_SIZE:
            process_current_batch()
            # Reset buffer
            batch_questions = []
            batch_indices = []

    # Processa eventuali domande rimanenti nel buffer alla fine del loop
    if batch_questions:
        process_current_batch()

    print(f"Completato dataset: {dataset_name}")

print("\nTutti i dataset completati!")