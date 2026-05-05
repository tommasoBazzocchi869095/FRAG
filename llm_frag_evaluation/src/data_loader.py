import json
import re
from pathlib import Path


def load_question_file(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON root in {path}: expected object or list")


def infer_dataset_from_path(path):
    name = Path(path).stem.lower()
    match = re.search(r"cache_step2_([^_]+)_scored", name)
    if match:
        return match.group(1)
    for dataset in ["mmlu", "medqa", "medmcqa", "pubmedqa", "bioasq"]:
        if dataset in name:
            return dataset
    return None


def infer_retriever_from_path(path):
    name = Path(path).stem.lower()
    for retriever in ["bm25", "contriever", "medcpt", "specter"]:
        if retriever in name:
            return retriever
    return "unknown_retriever"


def iter_input_questions(input_dir, input_files=None):
    input_path = Path(input_dir)
    if input_files:
        paths = [input_path / input_file for input_file in input_files]
    else:
        paths = sorted(input_path.glob("*.json"))

    for path in paths:
        inferred_dataset = infer_dataset_from_path(path)
        for question in load_question_file(path):
            if inferred_dataset and not question.get("dataset"):
                question["dataset"] = inferred_dataset
            question["_source_file"] = str(path)
            yield question


def get_passages(question):
    passages = question.get("passages")
    if passages is None:
        passages = question.get("snippets")
    if passages is None:
        return []
    if not isinstance(passages, list):
        raise ValueError(f"Passages must be a list for question {question.get('id', '<missing id>')}")
    return passages
