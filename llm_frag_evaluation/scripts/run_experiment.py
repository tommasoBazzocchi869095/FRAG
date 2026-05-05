import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
sys.path.insert(0, str(REPO_ROOT))

from llm_frag_evaluation.src.config import load_json, resolve_repo_path
from llm_frag_evaluation.src.data_loader import iter_input_questions
from llm_frag_evaluation.src.experiments import prepare_experiment_item
from llm_frag_evaluation.src.prompts import load_prompt_templates


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and run LLM evaluation experiments with FRAG.")
    parser.add_argument("--config", default="llm_frag_evaluation/configs/default_config.json")
    parser.add_argument("--input-file", action="append", default=None, help="Input JSON filename relative to input_dir. Can be repeated.")
    parser.add_argument("--experiment", choices=["zero_shot", "standard_rag", "frag"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write prompts and selected passages without model inference.")
    return parser.parse_args()


def dry_run_generate(item, llm):
    return {
        "answer_choice": "NA",
        "step_by_step_thinking": "",
        "backend": llm.get("backend", "dry_run"),
        "model_id": llm.get("model_id", ""),
    }


def write_prediction(output_dir, llm_name, item, prediction, question_number, file_prefix, system_info):
    dataset = item.get("dataset") or "unknown_dataset"
    experiment = item["experiment"]
    save_dir = Path(output_dir) / dataset / experiment / llm_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{file_prefix}_{question_number}.json"
    payload = [{
        "answer_choice": prediction.get("answer_choice", "NA"),
        "step_by_step_thinking": prediction.get("step_by_step_thinking", ""),
        "system_info": system_info,
    }]
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return save_path


def main():
    args = parse_args()
    config = load_json(args.config)
    experiments = [args.experiment] if args.experiment else config["experiments"]
    input_dir = resolve_repo_path(config["input_dir"])
    output_dir = resolve_repo_path(config["output_dir"])
    templates = load_prompt_templates(resolve_repo_path(config["prompt_file"]))
    dataset_counters = defaultdict(int)
    file_prefix = config.get("output_file_prefix", "test")
    system_info = config.get("system_info", "Offline Factuality")

    written = []
    input_files = args.input_file if args.input_file else config.get("input_files")

    for question in iter_input_questions(input_dir, input_files=input_files):
        dataset = question.get("dataset") or "unknown_dataset"
        question_number = dataset_counters[dataset]
        dataset_counters[dataset] += 1
        for experiment in experiments:
            item = prepare_experiment_item(
                question,
                experiment,
                templates,
                top_k_context=config.get("top_k_context", 32),
                alpha=config.get("frag_alpha", 0.6),
                normalize_topic=config.get("normalize_topic_scores", True),
            )
            for llm in config["llms"]:
                llm_name = llm["name"]
                if args.dry_run or llm.get("backend") == "dry_run":
                    prediction = dry_run_generate(item, llm)
                else:
                    raise NotImplementedError(
                        f"Backend {llm.get('backend')} is not implemented yet. "
                        "Use --dry-run while developing prompts and reranking."
                    )
                written.append(
                    write_prediction(
                        output_dir,
                        llm_name,
                        item,
                        prediction,
                        question_number,
                        file_prefix,
                        system_info,
                    )
                )

    print(f"Wrote {len(written)} files to {output_dir}")


if __name__ == "__main__":
    main()
