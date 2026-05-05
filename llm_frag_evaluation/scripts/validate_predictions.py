import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
sys.path.insert(0, str(REPO_ROOT))

from llm_frag_evaluation.src.config import resolve_repo_path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate generated evaluation JSON files for one prompt load.")
    parser.add_argument("--prompt-load", required=True)
    parser.add_argument("--prediction-dir", default=None)
    return parser.parse_args()


def derive_prediction_dir(prompt_load):
    path = Path(prompt_load).resolve()
    parts = path.parts
    try:
        idx = parts.index("prompt_loads")
    except ValueError:
        return Path("llm_frag_evaluation") / "outputs" / "predictions" / "unknown"
    relative = Path(*parts[idx + 1 : -1])
    return Path("llm_frag_evaluation") / "outputs" / "predictions" / relative


def valid_answer(answer, dataset):
    if dataset == "pubmedqa":
        return answer in {"yes", "no", "maybe"}
    if dataset == "bioasq":
        return answer in {"yes", "no"}
    return answer in {"A", "B", "C", "D"}


def main():
    args = parse_args()
    prompt_load = resolve_repo_path(args.prompt_load)
    prediction_dir = resolve_repo_path(args.prediction_dir) if args.prediction_dir else resolve_repo_path(derive_prediction_dir(prompt_load))
    errors = []
    checked = 0

    with prompt_load.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            path = prediction_dir / record["output_file"]
            if not path.exists():
                errors.append(f"missing prediction: {path}")
                continue
            checked += 1
            try:
                data = json.load(path.open("r", encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"{path}: invalid JSON: {exc}")
                continue
            if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
                errors.append(f"{path}: expected one-item JSON list")
                continue
            item = data[0]
            for field in ["answer_choice", "step_by_step_thinking", "system_info"]:
                if field not in item:
                    errors.append(f"{path}: missing field {field}")
            answer = str(item.get("answer_choice", "")).strip()
            if not valid_answer(answer, record["dataset"]):
                errors.append(f"{path}: invalid answer_choice {answer!r} for {record['dataset']}")

    report = {
        "prompt_load": str(prompt_load),
        "prediction_dir": str(prediction_dir),
        "checked_predictions": checked,
        "errors": len(errors),
    }
    print(json.dumps(report, indent=2))
    if errors:
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} more errors", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
