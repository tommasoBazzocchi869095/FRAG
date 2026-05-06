import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
sys.path.insert(0, str(REPO_ROOT))

from llm_frag_evaluation.src.config import load_json, resolve_repo_path
from llm_frag_evaluation.src.data_loader import infer_dataset_from_path, infer_retriever_from_path, load_question_file
from llm_frag_evaluation.src.scoring import _score_value, normalize_min_max


REQUIRED_RECORD_FIELDS = {
    "request_id",
    "dataset",
    "retriever",
    "experiment",
    "model",
    "question_number",
    "input_question_id",
    "output_file",
    "messages",
    "selected_passages",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate generated prompt loads.")
    parser.add_argument("--config", default="llm_frag_evaluation/configs/default_config.json")
    parser.add_argument("--prompt-load-dir", default="llm_frag_evaluation/outputs/prompt_loads")
    parser.add_argument("--input-file", action="append", default=None, help="Input JSON filename relative to input_dir. Can be repeated.")
    parser.add_argument("--all-input-files", action="store_true", help="Validate all cache_step2*.json inputs.")
    parser.add_argument("--strict-frag-order", action="store_true", help="Recompute FRAG order and require exact selected id order.")
    return parser.parse_args()


def selected_ids_from_question(question, experiment, top_k, alpha, normalize_topic):
    passages = question.get("passages") or question.get("snippets") or []
    if experiment == "zero_shot":
        return []
    if experiment == "standard_rag":
        ranked = sorted(
            passages,
            key=lambda p: _score_value(p, ["score_topic", "topic_score", "retrieval_score", "score_topic_orig"]),
            reverse=True,
        )
        return [p.get("id") for p in ranked[:top_k]]

    ranked = [dict(p) for p in passages]
    topic_scores = [
        _score_value(p, ["score_topic", "topic_score", "retrieval_score", "score_topic_orig"])
        for p in ranked
    ]
    factuality_scores = [
        _score_value(p, ["score_factuality", "factuality_score", "fact_score"])
        for p in ranked
    ]
    if normalize_topic:
        topic_scores = normalize_min_max(topic_scores)
    for passage, topic_score, factuality_score in zip(ranked, topic_scores, factuality_scores):
        passage["score_frag"] = alpha * topic_score + (1.0 - alpha) * factuality_score
    ranked.sort(key=lambda p: p["score_frag"], reverse=True)
    return [p.get("id") for p in ranked[:top_k]]


def get_input_files(input_dir, args, config):
    input_dir = Path(input_dir)
    if args.all_input_files:
        return [path.name for path in sorted(input_dir.glob("cache_step2*.json"))]
    if args.input_file:
        return args.input_file
    return config.get("input_files")


def load_records(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL record: {exc}") from exc
    return records


def expected_experiments_for_retriever(config, retriever):
    experiments = config.get("experiments", [])
    zero_shot_retriever = config.get("zero_shot_retriever", "bm25")
    if retriever in {zero_shot_retriever, "unknown_retriever"}:
        return experiments
    return [experiment for experiment in experiments if experiment != "zero_shot"]


def validate_messages(record, errors):
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        errors.append("messages must contain exactly system and user messages")
        return
    if messages[0].get("role") != "system":
        errors.append("first message must be system")
    if messages[1].get("role") != "user":
        errors.append("second message must be user")
    user_content = messages[1].get("content", "")
    if "answer_choice" not in user_content or "step_by_step_thinking" not in user_content:
        errors.append("user prompt is missing JSON answer schema")
    if "Here is the question:" not in user_content:
        errors.append("user prompt is missing MedRAG question header")
    if "Here are the potential choices:" not in user_content:
        errors.append("user prompt is missing MedRAG choices header")
    if "Please think step-by-step and generate your output in json:" not in user_content:
        errors.append("user prompt is missing MedRAG output instruction")
    if record["experiment"] == "zero_shot":
        if "Here are the relevant documents:" in user_content:
            errors.append("zero_shot prompt contains document context header")
        if "Document [0]" in user_content:
            errors.append("zero_shot prompt contains formatted documents")
    else:
        if "Here are the relevant documents:" not in user_content:
            errors.append(f"{record['experiment']} prompt is missing document context header")
        if "Document [0]" not in user_content:
            errors.append(f"{record['experiment']} prompt is missing formatted documents")


def validate_records(records, questions, input_file, dataset, retriever, experiment, config, args):
    errors = []
    top_k = config.get("top_k_context", 32)
    alpha = config.get("frag_alpha", 0.6)
    normalize_topic = config.get("normalize_topic_scores", True)
    file_prefix = config.get("output_file_prefix", "test")

    if len(records) != len(questions):
        errors.append(f"record count {len(records)} != input question count {len(questions)}")

    seen_numbers = set()
    for idx, record in enumerate(records):
        missing = REQUIRED_RECORD_FIELDS - set(record.keys())
        if missing:
            errors.append(f"record {idx}: missing fields {sorted(missing)}")
            continue

        prefix = f"record {idx} ({record.get('request_id')})"
        if record["dataset"] != dataset:
            errors.append(f"{prefix}: dataset {record['dataset']} != {dataset}")
        if record["retriever"] != retriever:
            errors.append(f"{prefix}: retriever {record['retriever']} != {retriever}")
        if record["experiment"] != experiment:
            errors.append(f"{prefix}: experiment {record['experiment']} != {experiment}")
        if record["question_number"] != idx:
            errors.append(f"{prefix}: question_number {record['question_number']} != {idx}")
        if record["question_number"] in seen_numbers:
            errors.append(f"{prefix}: duplicate question_number {record['question_number']}")
        seen_numbers.add(record["question_number"])
        expected_output = f"{file_prefix}_{idx}.json"
        if record["output_file"] != expected_output:
            errors.append(f"{prefix}: output_file {record['output_file']} != {expected_output}")

        if idx < len(questions):
            question = questions[idx]
            if record.get("input_question_id") != question.get("id"):
                errors.append(f"{prefix}: input_question_id does not match source question id")
            if record.get("gold_answer") != question.get("answer"):
                errors.append(f"{prefix}: gold_answer does not match source answer")

            expected_ids = selected_ids_from_question(question, experiment, top_k, alpha, normalize_topic)
            selected = record.get("selected_passages", [])
            selected_ids = [p.get("id") for p in selected]
            expected_count = 0 if experiment == "zero_shot" else min(top_k, len(question.get("passages") or question.get("snippets") or []))
            if len(selected) != expected_count:
                errors.append(f"{prefix}: selected passage count {len(selected)} != {expected_count}")
            if experiment in {"standard_rag", "frag"} and selected_ids != expected_ids:
                errors.append(f"{prefix}: selected passage ids do not match expected {experiment} order")
            if experiment == "frag":
                scores = [p.get("score_frag") for p in selected]
                if any(score is None for score in scores):
                    errors.append(f"{prefix}: FRAG selected passage missing score_frag")
                elif scores != sorted(scores, reverse=True):
                    errors.append(f"{prefix}: FRAG scores are not descending")

        validate_messages(record, errors)

    return errors


def main():
    args = parse_args()
    config = load_json(args.config)
    input_dir = resolve_repo_path(config["input_dir"])
    prompt_load_root = resolve_repo_path(args.prompt_load_dir)
    input_files = get_input_files(input_dir, args, config)

    if not input_files:
        raise SystemExit("No input files selected.")

    all_errors = []
    checked_files = 0
    checked_records = 0
    per_file_summary = []

    for input_file in input_files:
        input_path = input_dir / input_file
        questions = load_question_file(input_path)
        dataset = infer_dataset_from_path(input_path) or (questions[0].get("dataset") if questions else "unknown_dataset")
        retriever = infer_retriever_from_path(input_path)
        model_dirs = list((prompt_load_root / dataset / retriever).glob("*/*")) if (prompt_load_root / dataset / retriever).exists() else []

        expected_experiments = expected_experiments_for_retriever(config, retriever)
        for experiment in expected_experiments:
            experiment_dir = prompt_load_root / dataset / retriever / experiment
            if not experiment_dir.exists():
                all_errors.append(f"{input_file}: missing experiment directory {experiment_dir}")
                continue
            prompt_files = sorted(experiment_dir.glob("*/prompts.jsonl"))
            if not prompt_files:
                all_errors.append(f"{input_file}: missing prompts.jsonl under {experiment_dir}")
                continue
            for prompt_file in prompt_files:
                try:
                    records = load_records(prompt_file)
                except ValueError as exc:
                    all_errors.append(str(exc))
                    continue
                checked_files += 1
                checked_records += len(records)
                errors = validate_records(records, questions, input_file, dataset, retriever, experiment, config, args)
                all_errors.extend(f"{prompt_file}: {error}" for error in errors)
                per_file_summary.append({
                    "dataset": dataset,
                    "retriever": retriever,
                    "experiment": experiment,
                    "records": len(records),
                    "prompt_file": str(prompt_file),
                    "errors": len(errors),
                })

    report = {
        "checked_prompt_files": checked_files,
        "checked_records": checked_records,
        "errors": len(all_errors),
        "per_file_summary": per_file_summary,
    }
    print(json.dumps(report, indent=2))

    if all_errors:
        print("\nValidation errors:", file=sys.stderr)
        for error in all_errors[:100]:
            print(f"- {error}", file=sys.stderr)
        if len(all_errors) > 100:
            print(f"- ... {len(all_errors) - 100} more errors", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
