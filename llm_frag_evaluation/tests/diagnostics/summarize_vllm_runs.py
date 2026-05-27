import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize vLLM runs and flag jobs that should be repeated.")
    parser.add_argument(
        "--prompt-load-root",
        default="llm_frag_evaluation/outputs/prompt_loads/source_collection_pubmed",
        help="Root containing prompt-load trees.",
    )
    parser.add_argument(
        "--prediction-root",
        default="llm_frag_evaluation/outputs/predictions/source_collection_pubmed",
        help="Root containing prediction trees.",
    )
    parser.add_argument("--model-alias", default="Meta-Llama-3-70B-Instruct")
    parser.add_argument(
        "--output-csv",
        default="llm_frag_evaluation/tests/diagnostics/reports/vllm_run_summary.csv",
    )
    parser.add_argument(
        "--output-md",
        default="llm_frag_evaluation/tests/diagnostics/reports/vllm_run_summary.md",
    )
    return parser.parse_args()


def count_jsonl_records(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative_run_path(prompt_load, prompt_load_root, model_alias):
    relative = prompt_load.relative_to(prompt_load_root)
    parts = relative.parts
    model_index = parts.index(model_alias)
    return Path(*parts[:model_index])


def count_prediction_files(prediction_dir):
    if not prediction_dir.exists():
        return 0
    return sum(1 for path in prediction_dir.glob("test_*.json") if path.is_file())


def summarize_run(prompt_load, prompt_load_root, prediction_root, model_alias):
    run_path = relative_run_path(prompt_load, prompt_load_root, model_alias)
    prediction_dir = prediction_root / run_path / model_alias
    summary_path = prediction_dir / "run_summary.json"
    errors_path = prediction_dir / "generation_errors.jsonl"

    prompt_count = count_jsonl_records(prompt_load)
    prediction_file_count = count_prediction_files(prediction_dir)
    row = {
        "run": str(run_path).replace("\\", "/"),
        "prompt_count": prompt_count,
        "prediction_files": prediction_file_count,
        "summary_exists": summary_path.exists(),
        "parsed_record_count": "",
        "preflight_error_count": "",
        "error_record_count": "",
        "prompt_too_long": 0,
        "invalid_answer": 0,
        "other_errors": 0,
        "status": "",
        "action": "",
    }

    if not summary_path.exists():
        row["status"] = "missing_summary"
        row["action"] = "run_or_locate_job"
        return row

    summary = read_json(summary_path)
    error_type_counts = summary.get("error_type_counts", {})
    prompt_too_long = int(error_type_counts.get("PromptTooLong", 0))
    invalid_answer = int(error_type_counts.get("InvalidAnswer", 0))
    total_typed_errors = sum(int(value) for value in error_type_counts.values())
    other_errors = max(total_typed_errors - prompt_too_long - invalid_answer, 0)

    parsed = int(summary.get("parsed_record_count", 0))
    preflight = int(summary.get("preflight_error_count", 0))
    errors = int(summary.get("error_record_count", 0))

    row.update({
        "parsed_record_count": parsed,
        "preflight_error_count": preflight,
        "error_record_count": errors,
        "prompt_too_long": prompt_too_long,
        "invalid_answer": invalid_answer,
        "other_errors": other_errors,
    })

    if prompt_too_long:
        row["status"] = "incomplete_prompt_too_long"
        row["action"] = "repeat_after_context_fix"
    elif other_errors:
        row["status"] = "incomplete_generation_errors"
        row["action"] = "inspect_then_repeat"
    elif prediction_file_count < prompt_count:
        row["status"] = "incomplete_missing_predictions"
        row["action"] = "repeat_or_resume"
    elif invalid_answer:
        row["status"] = "complete_with_invalid_answers"
        row["action"] = "inspect_invalid_or_repeat"
    else:
        row["status"] = "complete"
        row["action"] = "none"

    if errors_path.exists() and errors == 0 and errors_path.stat().st_size > 0:
        row["status"] = "check_error_file"
        row["action"] = "inspect_errors_jsonl"

    return row


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "prompt_count",
        "prediction_files",
        "summary_exists",
        "parsed_record_count",
        "preflight_error_count",
        "error_record_count",
        "prompt_too_long",
        "invalid_answer",
        "other_errors",
        "status",
        "action",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    repeat_rows = [row for row in rows if row["action"] != "none"]
    lines = [
        "# vLLM Run Summary",
        "",
        f"- Runs checked: `{len(rows)}`",
        f"- Runs needing action: `{len(repeat_rows)}`",
        "",
        "## Runs Needing Action",
        "",
        "| Run | Prompt Count | Predictions | PromptTooLong | Invalid | Other Errors | Status | Action |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in repeat_rows:
        lines.append(
            f"| `{row['run']}` | {row['prompt_count']} | {row['prediction_files']} | "
            f"{row['prompt_too_long']} | {row['invalid_answer']} | {row['other_errors']} | "
            f"`{row['status']}` | `{row['action']}` |"
        )

    if not repeat_rows:
        lines.append("| None |  |  |  |  |  |  |  |")

    lines.extend([
        "",
        "## All Runs",
        "",
        "| Run | Prompt Count | Predictions | Status | Action |",
        "|---|---:|---:|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| `{row['run']}` | {row['prompt_count']} | {row['prediction_files']} | "
            f"`{row['status']}` | `{row['action']}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    prompt_load_root = Path(args.prompt_load_root)
    prediction_root = Path(args.prediction_root)
    prompt_loads = sorted(prompt_load_root.rglob(f"{args.model_alias}/prompts.jsonl"))

    rows = [
        summarize_run(prompt_load, prompt_load_root, prediction_root, args.model_alias)
        for prompt_load in prompt_loads
    ]

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(json.dumps({
        "runs_checked": len(rows),
        "runs_needing_action": sum(1 for row in rows if row["action"] != "none"),
        "csv": args.output_csv,
        "markdown": args.output_md,
    }, indent=2))


if __name__ == "__main__":
    main()
