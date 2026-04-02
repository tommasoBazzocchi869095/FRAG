import os
import sys
import json
import re
import torch
import gc
from tqdm import tqdm
from src.medrag import MedRAG

BATCH_SIZE = 1
K = 32
ALPHA = 1
llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
corpus_name = "Wikipedia"
retriever_name = "BM25"
CACHE_DIR = "."
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MIRAGE/prediction_ZERO_SHOT"))
dataset_names = ["pubmedqa", "bioasq"]

medrag = MedRAG(
    llm_name=llm_name,
    rag=False,
    retriever_name=retriever_name,
    corpus_name=corpus_name,
    corpus_cache=True,
    HNSW=True,
    factuality=False
)

def parse_answer(raw_answer, dataset_type="mcq", debug=False):
    """ Tua funzione originale di parsing """
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
        ac_pattern = r'[\'"]answer_choice[\'"]\s*:\s*[\'"](.*?)[\'"]'
        ac_match = re.search(ac_pattern, text, re.IGNORECASE | re.DOTALL)
        if ac_match:
            answer_choice = ac_match.group(1).strip().upper()
        sb_pattern = r'[\'"]step_by_step_thinking[\'"]\s*:\s*[\'"](.*?)[\'"]'
        sb_match = re.search(sb_pattern, text, re.IGNORECASE | re.DOTALL)
        if sb_match:
            step_by_step = sb_match.group(1).strip()
    except Exception as e:
        pass
    if dataset_type == "mcq" and len(answer_choice) > 1:
        if answer_choice[0].isalpha() and answer_choice[1] in ['.', ')', ' ']:
            answer_choice = answer_choice[0]
    return answer_choice, step_by_step

for dataset_name in dataset_names:
    print(f"\n=== Processing dataset: {dataset_name} (FROM CACHE STEP 2) ===")

    input_file = os.path.join(CACHE_DIR, f"cache_step2_{dataset_name}_scored.json")
    if not os.path.exists(input_file):
        print(f"File {input_file} non trovato. Salto.")
        continue

    with open(input_file, "r") as f:
        dataset_data = json.load(f)

    save_dir = os.path.join(results_dir, dataset_name, f"rag_{K}_factuality", llm_name, corpus_name, retriever_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output: {save_dir}")

    if dataset_name in ["mmlu", "medqa", "medmcqa"]:
        question_type = "mcq"
    elif dataset_name == "pubmedqa":
        question_type = "ynm"
    else:
        question_type = "yn"

    split = "dev" if dataset_name == "medmcqa" else "test"

    batch_prompts = []
    batch_indices = []

    def process_current_batch():
        if not batch_prompts: return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            gen_kwargs = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9,
                "batch_size": len(batch_prompts),
                "pad_token_id": medrag.tokenizer.pad_token_id
            }

            outputs = medrag.model(batch_prompts, **gen_kwargs)

            for i, output in enumerate(outputs):
                original_idx = batch_indices[i]
                item_data = dataset_data[original_idx]

                full_generated = output[0]["generated_text"]
                prompt_len = len(batch_prompts[i])
                raw_answer = full_generated[prompt_len:]
                raw_answer = re.sub("\s+", " ", raw_answer)

                answer_choice, step_by_step = parse_answer(raw_answer, dataset_type=question_type)

                output_json = [{
                    "answer_choice": answer_choice,
                    "step_by_step_thinking": step_by_step,
                    "system_info": "Offline Factuality"
                }]

                filename = f"{split}_{item_data['id']}.json"
                save_path = os.path.join(save_dir, filename)
                with open(save_path, "w") as f:
                    json.dump(output_json, f, indent=2)

        except Exception as e:
            print(f"Errore batch: {e}")

    for idx, item in enumerate(tqdm(dataset_data)):
        filename = f"{split}_{item['id']}.json"
        if os.path.exists(os.path.join(save_dir, filename)):
            continue

        question = item["question"]
        context_str = ""

        if question_type == "mcq":
            options = item["options"]
            options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

            formatted_q = (
                "You are a medical expert. Read the following question carefully and choose the single best answer.\n\n"
                f"Question:\n{question}\n\n"
                + options_text
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
        else:
            formatted_q = (
                "You are a biomedical researcher. Read the following question carefully.\n\n"
                f"Question:\n{question}\n\n"
                "Possible answers: yes or no.\n"
                "Respond strictly in JSON format as follows:\n"
                '{ "answer_choice": "yes", "step_by_step_thinking": "your short biomedical justification" }'
            )

        prompt_text_user = medrag.templates["medrag_prompt"].render(
            context=context_str,
            question=formatted_q,
            options=""
        )

        system_text = medrag.templates["medrag_system"]
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt_text_user}
        ]

        full_prompt = medrag.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        batch_prompts.append(full_prompt)
        batch_indices.append(idx)

        if len(batch_prompts) >= BATCH_SIZE:
            process_current_batch()
            batch_prompts = []
            batch_indices = []

    if batch_prompts:
        process_current_batch()

    print(f"Completato dataset: {dataset_name}")

print("\nTutti i dataset completati!")