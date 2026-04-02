import os
import json
import re
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from src.medrag import MedRAG

ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
MODELS = ["source", "claim"]
DATASETS = ["medqa", "medmcqa", "pubmedqa"]
INPUT_PREFIX = "validation_set_external"
LLM_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
K = 32
BATCH_SIZE = 1


SAVE_DEBUG = True
DEBUG_DIR = "debug_results"

if SAVE_DEBUG:
    os.makedirs(DEBUG_DIR, exist_ok=True)

print(f"Inizializzazione LLM: {LLM_NAME}")
medrag = MedRAG(
    llm_name=LLM_NAME,
    rag=False,
    retriever_name="BM25",
    corpus_name="Wikipedia",
    corpus_cache=True,
    HNSW=True,
    factuality=False
)

def get_parsed_pred(raw_gen, ds_name):
    """Estrae la predizione pulita (es. 'A' o 'YES')"""
    text = raw_gen.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).replace("```", "")
    pred = "NA"
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            j = json.loads(match.group(1))
            pred = str(j.get("answer_choice", "NA")).upper().strip()
    except: pass

    if pred == "NA":
        match = re.search(r"['\"]answer_choice['\"]\s*:\s*['\"](.*?)['\"]", text, re.IGNORECASE)
        if match: pred = match.group(1).upper().strip()

    pred = re.sub(r"[^A-Z]", "", pred)

    if pred == "NA":
            return "NA"

    if ds_name == "pubmedqa":
        if pred.startswith("Y"): pred = "YES"
        elif pred.startswith("N"): pred = "NO"
        elif pred.startswith("M"): pred = "MAYBE"

    return pred

def check_answer(pred, ground_truth):
    """Confronta predizione pulita e verità"""
    return 1 if pred == str(ground_truth).upper().strip() else 0

# File di log riassuntivo (CSV) - AGGIORNATA INTESTAZIONE
log_filename = "grid_search_results.csv"
if not os.path.exists(log_filename):
    with open(log_filename, "w") as f:
        # Aggiungiamo le nuove colonne
        f.write("Dataset,Model,Alpha,Accuracy,Precision,Recall,F1_Score\n")

for ds in DATASETS:
    file_source = f"{INPUT_PREFIX}_{ds}_scored_source.json"
    file_claim = f"{INPUT_PREFIX}_{ds}_scored_claim.json"

    if ds in ["medqa", "medmcqa"]:
        question_type = "mcq"
    else:
        question_type = "ynm"

    for model_type in MODELS:
        current_file = file_source if model_type == "source" else file_claim

        if not os.path.exists(current_file):
            continue

        print(f"\nCaricamento dati: {current_file}")
        data = json.load(open(current_file))
        score_key = f"score_factuality_{model_type}"

        for alpha in ALPHAS:
            print(f"> Testing {ds} | {model_type} | Alpha {alpha}")
            all_preds_clean = []
            all_truths_clean = []
            batch_prompts = []
            batch_truths = []
            batch_items = []

            debug_log = []

            for item in tqdm(data):
                snippets = item["snippets"]
                if not snippets: continue

                t_scores = [float(s.get("score_topic", 0)) for s in snippets]
                if not t_scores: continue
                min_t, max_t = min(t_scores), max(t_scores)
                denom = (max_t - min_t) if (max_t - min_t) > 1e-9 else 1.0

                reranked = []
                for s in snippets:
                    norm_topic = (float(s.get("score_topic", 0)) - min_t) / denom
                    fact = float(s.get(score_key, 0))
                    final_score = (alpha * norm_topic) + ((1-alpha) * fact)
                    s["temp_score"] = final_score
                    reranked.append(s)

                reranked.sort(key=lambda x: x["temp_score"], reverse=True)
                top_k = reranked[:K]

                # --- Prompt ---
                contexts = [f"Document [{i}] (Title: {s.get('title','')}) {s.get('content','')}" for i, s in enumerate(top_k)]
                context_str = "\n".join(contexts)

                if question_type == "mcq":
                    options = item["options"]
                    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
                    formatted_q = (
                        "You are a medical expert. Read the following question carefully and choose the single best answer.\n\n"
                        f"Question:\n{item['question']}\n\n"
                        + options_text
                        + "\n\nAnswer the question and then briefly explain your reasoning.\n"
                        + "Your response must be in JSON format with exactly these keys:\n"
                        + '{ "answer_choice": "A", "step_by_step_thinking": "your explanation here" }'
                    )
                else:
                    formatted_q = (
                        "You are a biomedical researcher. Read the following question carefully.\n\n"
                        f"Question:\n{item['question']}\n\n"
                        "Possible answers: yes, no, or maybe.\n"
                        "Respond strictly in JSON format as follows:\n"
                        '{ "answer_choice": "yes", "step_by_step_thinking": "your short biomedical justification" }'
                    )

                prompt_text_user = medrag.templates["medrag_prompt"].render(
                    context=context_str, question=formatted_q, options=""
                )

                system_text = medrag.templates["medrag_system"]
                msgs = [{"role": "system", "content": system_text}, {"role": "user", "content": prompt_text_user}]
                full_txt = medrag.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

                batch_prompts.append(full_txt)
                batch_truths.append(item["answer"])
                batch_items.append(item)

                if len(batch_prompts) >= BATCH_SIZE:
                    with torch.no_grad():
                        outs = medrag.model(batch_prompts, max_new_tokens=512, do_sample=False, temperature=0.0, pad_token_id=medrag.tokenizer.eos_token_id)

                    for i, o in enumerate(outs):
                        gen = o[0]["generated_text"][len(batch_prompts[i]):]
                        pred = get_parsed_pred(gen, ds)

                        all_preds_clean.append(pred)
                        all_truths_clean.append(str(batch_truths[i]).upper().strip())

                        is_correct = check_answer(pred, batch_truths[i])

                        if SAVE_DEBUG:
                            debug_log.append({
                                "id": batch_items[i]["id"],
                                "question": batch_items[i]["question"],
                                "ground_truth": batch_truths[i],
                                "model_raw_output": gen,
                                "parsed_prediction": pred,
                                "is_correct": bool(is_correct)
                            })

                    batch_prompts, batch_truths, batch_items = [], [], []

            # Flush finale
            if batch_prompts:
                with torch.no_grad():
                    outs = medrag.model(batch_prompts, max_new_tokens=512, do_sample=False, temperature=0.0, pad_token_id=medrag.tokenizer.eos_token_id)
                for i, o in enumerate(outs):
                    gen = o[0]["generated_text"][len(batch_prompts[i]):]
                    pred = get_parsed_pred(gen, ds)
                    all_preds_clean.append(pred)
                    all_truths_clean.append(str(batch_truths[i]).upper().strip())

                    is_correct = check_answer(pred, batch_truths[i])

                    if SAVE_DEBUG:
                        debug_log.append({
                            "id": batch_items[i]["id"],
                            "question": batch_items[i]["question"],
                            "ground_truth": batch_truths[i],
                            "model_raw_output": gen,
                            "parsed_prediction": pred,
                            "is_correct": bool(is_correct)
                        })

            if all_truths_clean:
                correct_count = sum([1 for p, t in zip(all_preds_clean, all_truths_clean) if p == t])
                acc = correct_count / len(all_truths_clean)
                prec = precision_score(all_truths_clean, all_preds_clean, average='macro', zero_division=0)
                rec = recall_score(all_truths_clean, all_preds_clean, average='macro', zero_division=0)
                f1 = f1_score(all_truths_clean, all_preds_clean, average='macro', zero_division=0)
            else:
                acc, prec, rec, f1 = 0, 0, 0, 0

            print(f"--> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

            with open(log_filename, "a") as f:
                f.write(f"{ds},{model_type},{alpha},{acc},{prec},{rec},{f1}\n")

            if SAVE_DEBUG:
                debug_filename = f"{DEBUG_DIR}/{ds}_{model_type}_alpha{alpha}.json"
                with open(debug_filename, "w") as f:
                    json.dump(debug_log, f, indent=4)
