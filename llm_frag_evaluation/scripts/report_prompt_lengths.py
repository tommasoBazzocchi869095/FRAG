import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Report token lengths for generated prompt-load JSONL files.")
    parser.add_argument("--prompt-load-dir", default="llm_frag_evaluation/outputs/prompt_loads")
    parser.add_argument("--model-path", required=True, help="Local model path or HF id used to load the tokenizer.")
    parser.add_argument("--model-alias", default="Meta-Llama-3-70B-Instruct")
    parser.add_argument("--threshold", action="append", type=int, default=None)
    parser.add_argument("--fail-over", type=int, default=None, help="Exit nonzero if any prompt is longer than this.")
    return parser.parse_args()


def load_tokenizer(model_path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers is required. Activate the vLLM/HF environment first.") from exc
    return AutoTokenizer.from_pretrained(model_path)


def prompt_text(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"


def iter_prompt_loads(root, model_alias):
    return sorted(Path(root).glob(f"*/*/*/{model_alias}/prompts.jsonl"))


def main():
    args = parse_args()
    thresholds = args.threshold if args.threshold else [4096, 8192, 12288, 16384]
    tokenizer = load_tokenizer(args.model_path)
    paths = iter_prompt_loads(args.prompt_load_dir, args.model_alias)

    if not paths:
        raise SystemExit(f"No prompt loads found under {args.prompt_load_dir!r} for model alias {args.model_alias!r}.")

    report = []
    overall_max = None
    fail_count = 0

    for path in paths:
        lengths = []
        max_record = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = prompt_text(tokenizer, record["messages"])
                token_count = len(tokenizer(text).input_ids)
                lengths.append(token_count)
                if max_record is None or token_count > max_record["tokens"]:
                    max_record = {
                        "tokens": token_count,
                        "request_id": record.get("request_id"),
                        "output_file": record.get("output_file"),
                    }

        over_counts = {str(threshold): sum(length > threshold for length in lengths) for threshold in thresholds}
        if args.fail_over is not None:
            fail_count += sum(length > args.fail_over for length in lengths)

        item = {
            "prompt_load": str(path),
            "count": len(lengths),
            "avg_tokens": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
            "max_record": max_record,
            "over_threshold": over_counts,
        }
        report.append(item)
        if overall_max is None or item["max_tokens"] > overall_max["max_tokens"]:
            overall_max = item

    output = {
        "model_path": args.model_path,
        "model_alias": args.model_alias,
        "prompt_load_count": len(paths),
        "thresholds": thresholds,
        "overall_max": overall_max,
        "fail_over": args.fail_over,
        "fail_count": fail_count,
        "prompt_loads": report,
    }
    print(json.dumps(output, indent=2))

    if fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
