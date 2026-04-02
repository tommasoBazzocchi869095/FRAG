import os
import json
import re
import torch
from tqdm import tqdm
from src.medrag import MedRAG

ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
MODELS = ["source", "claim"]
DATASETS = ["medqa", "medmcqa", "pubmedqa"]
INPUT_PREFIX = "validation_set_external"
LLM_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
K = 32
BATCH_SIZE = 4

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

def check_answer(raw_gen, ground_truth, ds_name):
    """
    Funzione robusta per parsare il JSON generato da Llama-3.
    """
    text = raw_gen.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).replace("```", "")
    pred = "NA"
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            j = json.loads(match.group(1))
            pred = str(j.get("answer_choice", "NA")).upper().strip()
    except:
        pass

    if pred == "NA":
        match = re.search(r"['\"]answer_choice['\"]\s*:\s*['\"](.*?)['\"]", text, re.IGNORECASE)
        if match:
            pred = match.group(1).upper().strip()

    pred = re.sub(r"[^A-Z]", "", pred)
    truth = str(ground_truth).upper().strip()
    if ds_name == "pubmedqa":
        if pred.startswith("Y"): pred = "YES"
        elif pred.startswith("N"): pred = "NO"
        elif pred.startswith("M"): pred = "MAYBE"

    return 1 if pred == truth else 0

log_filename = "grid_search_results.csv"
if not os.path.exists(log_filename):
    with open(log_filename, "w") as f:
        f.write("Dataset,Model,Alpha,Accuracy\n")

print(f"Risultati salvati in: {log_filename}")

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
            print(f"Skipping {ds} | {model_type}: file non trovato.")
            continue

        print(f"Caricamento dati: {current_file}")
        data = json.load(open(current_file))
        score_key = f"score_factuality_{model_type}"

        for alpha in ALPHAS:
            print(f"\n>>> Testing {ds} | {model_type} | Alpha {alpha}")

            correct = 0
            total = 0
            batch_prompts = []
            batch_truths = []

            for item in tqdm(data):
                snippets = item["snippets"]
                if not snippets: continue

                t_scores = [float(s.get("score_topic", 0)) for s in snippets]
                if not t_scores:
                    continue

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
                else: # PubMedQA
                    formatted_q = (
                        "You are a biomedical researcher. Read the following question carefully.\n\n"
                        f"Question:\n{item['question']}\n\n"
                        "Possible answers: yes, no, or maybe.\n"
                        "Respond strictly in JSON format as follows:\n"
                        '{ "answer_choice": "yes", "step_by_step_thinking": "your short biomedical justification" }'
                    )

                # 3. Applicazione Template Jinja
                prompt_text_user = medrag.templates["medrag_prompt"].render(
                    context=context_str,
                    question=formatted_q,
                    options=""
                )

                system_text = medrag.templates["medrag_system"]
                msgs = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt_text_user}
                ]

                full_txt = medrag.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

                batch_prompts.append(full_txt)
                batch_truths.append(item["answer"])

                # --- D. INFERENZA ---
                if len(batch_prompts) >= BATCH_SIZE:
                    with torch.no_grad():
                        outs = medrag.model(
                            batch_prompts,
                            max_new_tokens=512, # Lungo abbastanza per il thinking
                            do_sample=False,    # Determinismo per grid search
                            temperature=0.0,
                            pad_token_id=medrag.tokenizer.eos_token_id
                        )
                    for i, o in enumerate(outs):
                        # Taglia il prompt dall'output
                        gen = o[0]["generated_text"][len(batch_prompts[i]):]
                        correct += check_answer(gen, batch_truths[i], ds)
                        total += 1
                    batch_prompts, batch_truths = [], []

            if batch_prompts:
                with torch.no_grad():
                    outs = medrag.model(batch_prompts, max_new_tokens=512, do_sample=False, temperature=0.0, pad_token_id=medrag.tokenizer.eos_token_id)
                for i, o in enumerate(outs):
                    gen = o[0]["generated_text"][len(batch_prompts[i]):]
                    correct += check_answer(gen, batch_truths[i], ds)
                    total += 1

            acc = correct / total if total > 0 else 0
            print(f"--> Accuracy: {acc:.4f} ({correct}/{total})")

            with open(log_filename, "a") as f:
                f.write(f"{ds},{model_type},{alpha},{acc}\n")

print("\nGrid Search Completata! Controlla il file grid_search_results.csv")