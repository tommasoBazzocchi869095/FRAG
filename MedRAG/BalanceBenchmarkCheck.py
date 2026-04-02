import json
import os
from collections import Counter

dataset_config = {
    "cache_step1_mmlu.json": "MCQ",
    "cache_step1_medqa.json": "MCQ",
    "cache_step1_pubmedqa.json": "TERNARY",
    "cache_step1_bioasq.json": "BINARY"
}

def analyze_file(filename, mode):
    if not os.path.exists(filename):
        print(f"File {filename} non trovato.")
        return

    dataset_name = filename.replace('cache_step1_', '').replace('.json', '').upper()
    print(f"\nANALISI DATASET: {dataset_name} ({mode})")

    with open(filename, 'r') as f:
        data = json.load(f)

    total = len(data)
    print(f"   Totale domande: {total}")

    answers = []

    for item in data:
        raw_ans = item.get("answer", "NA")

        if mode == "MCQ":
            if str(raw_ans).isdigit():
                mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
                clean_ans = mapping.get(int(raw_ans), str(raw_ans))
            else:
                clean_ans = str(raw_ans)
            answers.append(clean_ans.upper())

        elif mode == "TERNARY" or mode == "BINARY":
            clean_ans = str(raw_ans).lower()
            if "yes" in clean_ans:
                answers.append("YES")
            elif "no" in clean_ans:
                answers.append("NO")
            elif "maybe" in clean_ans:
                answers.append("MAYBE")
            else:
                answers.append(clean_ans.upper())

    counts = Counter(answers)

    print("   Distribuzione Classi:")
    sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    for label, count in sorted_classes:
        perc = (count / total) * 100
        print(f"   {label:<10}: {count} ({perc:.2f}%)")

    top_class_perc = sorted_classes[0][1] / total
    if top_class_perc > 0.60:
        print(f"   ATTENZIONE: Dataset sbilanciato! La classe '{sorted_classes[0][0]}' domina.")

for filename, mode in dataset_config.items():
    analyze_file(filename, mode)