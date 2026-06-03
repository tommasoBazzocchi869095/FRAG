import argparse
import heapq
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a smoke prompt-load from the longest prompts.")
    parser.add_argument("--prompt-load", required=True, help="Source prompts.jsonl.")
    parser.add_argument("--model-path", required=True, help="Local HF/vLLM model path for tokenization.")
    parser.add_argument("--run-name", required=True, help="Name used in the diagnostic output path.")
    parser.add_argument("--top-longest", type=int, default=5)
    parser.add_argument("--include-first", type=int, default=0)
    parser.add_argument(
        "--output-root",
        default="llm_frag_evaluation/outputs/prompt_loads/diagnostics",
        help="Root where the diagnostic prompt-load tree is written.",
    )
    parser.add_argument("--model-alias", default="model")
    return parser.parse_args()


def load_tokenizer(model_path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers or activate the vLLM environment to tokenize prompt loads.") from exc
    return AutoTokenizer.from_pretrained(model_path)


def render_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"


def record_key(record):
    return record.get("request_id") or record.get("output_file") or str(record.get("question_number"))


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.model_path)
    longest = []
    first_records = []

    with Path(args.prompt_load).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            token_count = len(tokenizer(render_prompt(tokenizer, record["messages"])).input_ids)
            item = (token_count, index, record)
            if index < args.include_first:
                first_records.append(item)
            if len(longest) < args.top_longest:
                heapq.heappush(longest, item)
            elif token_count > longest[0][0]:
                heapq.heapreplace(longest, item)

    selected = []
    seen = set()
    for item in first_records + sorted(longest, reverse=True):
        _, _, record = item
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)

    output_dir = Path(args.output_root) / args.run_name / args.model_alias
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prompts.jsonl"
    manifest_path = output_dir / "manifest.json"

    with output_path.open("w", encoding="utf-8") as handle:
        for _, _, record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source_prompt_load": str(args.prompt_load),
        "output_prompt_load": str(output_path),
        "selected_count": len(selected),
        "top_longest": args.top_longest,
        "include_first": args.include_first,
        "records": [
            {
                "source_index": index,
                "prompt_tokens": token_count,
                "request_id": record.get("request_id"),
                "output_file": record.get("output_file"),
                "dataset": record.get("dataset"),
                "retriever": record.get("retriever"),
                "experiment": record.get("experiment"),
                "question_number": record.get("question_number"),
            }
            for token_count, index, record in selected
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"prompt_load": str(output_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
