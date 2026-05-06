import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
sys.path.insert(0, str(REPO_ROOT))

from llm_frag_evaluation.src.config import load_json, resolve_repo_path
from llm_frag_evaluation.src.data_loader import iter_input_questions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLM FRAG predictions with macro metrics.")
    parser.add_argument("--config", default="llm_frag_evaluation/configs/default_config.json")
    parser.add_argument("--input-file", action="append", default=None, help="Input JSON filename relative to input_dir. Can be repeated.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retriever", default=None)
    parser.add_argument("--experiment", required=True, choices=["zero_shot", "standard_rag", "frag"])
    parser.add_argument("--llm", required=True)
    return parser.parse_args()


def load_prediction(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        return str(data[0].get("answer_choice", "NA")).strip()
    if isinstance(data, dict):
        return str(data.get("answer_choice", "NA")).strip()
    return "NA"


LETTER_TO_YNM = {
    "A": "yes",
    "B": "no",
    "C": "maybe",
}

LETTER_TO_YN = {
    "A": "yes",
    "B": "no",
}


def normalize_answer(answer, dataset):
    answer = str(answer).strip()
    answer_upper = answer.upper()
    answer_lower = answer.lower()

    if dataset == "pubmedqa":
        if answer_upper in LETTER_TO_YNM:
            return LETTER_TO_YNM[answer_upper]
        if answer_lower in {"yes", "no", "maybe"}:
            return answer_lower
        return "NA"

    if dataset == "bioasq":
        if answer_upper in LETTER_TO_YN:
            return LETTER_TO_YN[answer_upper]
        if answer_lower in {"yes", "no"}:
            return answer_lower
        return "NA"

    if len(answer) > 1 and answer[0].isalpha():
        return answer[0].upper()
    return answer.upper()


def main():
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for evaluation. Install it with: "
            "pip install -r llm_frag_evaluation/requirements.txt"
        ) from exc

    args = parse_args()
    config = load_json(args.config)
    input_dir = resolve_repo_path(config["input_dir"])
    if args.retriever:
        prediction_dir = resolve_repo_path(config["output_dir"]) / args.dataset / args.retriever / args.experiment / args.llm
    else:
        prediction_dir = resolve_repo_path(config["output_dir"]) / args.dataset / args.experiment / args.llm
    file_prefix = config.get("output_file_prefix", "test")

    gold_by_dataset = defaultdict(list)
    input_files = args.input_file if args.input_file else config.get("input_files")

    for question in iter_input_questions(input_dir, input_files=input_files):
        dataset = question.get("dataset") or "unknown_dataset"
        if "answer" in question:
            gold_by_dataset[dataset].append(normalize_answer(question["answer"], dataset))

    y_true = gold_by_dataset[args.dataset]
    y_pred = []
    missing = []
    for idx in range(len(y_true)):
        path = prediction_dir / f"{file_prefix}_{idx}.json"
        if not path.exists():
            missing.append(str(path))
            y_pred.append("NA")
            continue
        y_pred.append(normalize_answer(load_prediction(path), args.dataset))

    metrics = {
        "dataset": args.dataset,
        "retriever": args.retriever,
        "experiment": args.experiment,
        "llm": args.llm,
        "n": len(y_true),
        "missing_predictions": len(missing),
        "accuracy": accuracy_score(y_true, y_pred) if y_true else 0.0,
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0,
    }

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
