import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
sys.path.insert(0, str(REPO_ROOT))

from llm_frag_evaluation.src.config import load_json, resolve_repo_path
from llm_frag_evaluation.src.data_loader import infer_retriever_from_path, iter_input_questions
from llm_frag_evaluation.src.experiments import prepare_experiment_item
from llm_frag_evaluation.src.prompts import load_prompt_templates


DEFAULT_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Create prompt loads for LLM FRAG evaluation.")
    parser.add_argument("--config", default="llm_frag_evaluation/configs/default_config.json")
    parser.add_argument("--input-file", action="append", default=None, help="Input JSON filename relative to input_dir. Can be repeated.")
    parser.add_argument("--all-input-files", action="store_true", help="Use every cache_step2*.json file in input_dir.")
    parser.add_argument("--experiment", choices=["zero_shot", "standard_rag", "frag"], default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-alias", default="Meta-Llama-3-70B-Instruct")
    parser.add_argument("--output-dir", default="llm_frag_evaluation/outputs/prompt_loads")
    return parser.parse_args()


def get_input_files(input_dir, args, config):
    input_dir = Path(input_dir)
    if args.all_input_files:
        return [path.name for path in sorted(input_dir.glob("cache_step2*.json"))]
    if args.input_file:
        return args.input_file
    return config.get("input_files")


def passage_trace(passages):
    trace = []
    for rank, passage in enumerate(passages, start=1):
        trace.append({
            "rank": rank,
            "id": passage.get("id"),
            "title": passage.get("title"),
            "score_topic": passage.get("score_topic", passage.get("topic_score")),
            "score_factuality": passage.get("score_factuality", passage.get("factuality_score")),
            "score_frag": passage.get("score_frag"),
        })
    return trace


def safe_name(value):
    return str(value).replace("/", "__").replace("\\", "__").replace(" ", "_")


def main():
    args = parse_args()
    config = load_json(args.config)
    input_dir = resolve_repo_path(config["input_dir"])
    output_root = resolve_repo_path(args.output_dir)
    templates = load_prompt_templates(resolve_repo_path(config["prompt_file"]))
    experiments = [args.experiment] if args.experiment else config["experiments"]
    input_files = get_input_files(input_dir, args, config)

    if not input_files:
        raise SystemExit("No input files selected.")

    counts = defaultdict(int)
    handles = {}
    manifest = []
    file_prefix = config.get("output_file_prefix", "test")

    try:
        for input_file in input_files:
            retriever = infer_retriever_from_path(input_file)
            for question in iter_input_questions(input_dir, input_files=[input_file]):
                dataset = question.get("dataset") or "unknown_dataset"
                source_key = (input_file, dataset)
                question_number = counts[source_key]
                counts[source_key] += 1

                for experiment in experiments:
                    item = prepare_experiment_item(
                        question,
                        experiment,
                        templates,
                        top_k_context=config.get("top_k_context", 32),
                        alpha=config.get("frag_alpha", 0.6),
                        normalize_topic=config.get("normalize_topic_scores", True),
                    )
                    out_dir = output_root / dataset / retriever / experiment / safe_name(args.model_alias)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / "prompts.jsonl"
                    if out_path not in handles:
                        handles[out_path] = out_path.open("w", encoding="utf-8")
                        manifest.append({
                            "dataset": dataset,
                            "retriever": retriever,
                            "experiment": experiment,
                            "model": args.model,
                            "model_alias": args.model_alias,
                            "input_file": input_file,
                            "prompt_load": str(out_path),
                        })

                    record = {
                        "request_id": f"{dataset}_{retriever}_{experiment}_{question_number}",
                        "dataset": dataset,
                        "retriever": retriever,
                        "experiment": experiment,
                        "model": args.model,
                        "model_alias": args.model_alias,
                        "question_number": question_number,
                        "input_question_id": question.get("id"),
                        "output_file": f"{file_prefix}_{question_number}.json",
                        "gold_answer": question.get("answer"),
                        "messages": item["prompt"]["messages"],
                        "selected_passages": passage_trace(item["selected_passages"]),
                    }
                    handles[out_path].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    manifest_path = output_root / "manifest.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {sum(counts.values())} questions across {len(handles)} prompt-load files.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
