import argparse
import heapq
import json
import math
from collections import Counter
from pathlib import Path


DEFAULT_THRESHOLDS = [4096, 8192, 12288, 16384, 24576, 32768]


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose one vLLM prompt-load generation run.")
    parser.add_argument("--summary", required=True, help="Path to run_summary.json.")
    parser.add_argument("--errors", required=True, help="Path to generation_errors.jsonl.")
    parser.add_argument("--prompt-load", default=None, help="Optional prompts.jsonl to compute full prompt lengths.")
    parser.add_argument("--model-path", default=None, help="Optional local HF/vLLM model path for tokenization.")
    parser.add_argument("--run-name", default=None, help="Stable name for report files.")
    parser.add_argument("--output-dir", default="llm_frag_evaluation/tests/diagnostics/reports")
    parser.add_argument("--threshold", action="append", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--context-buffer-tokens",
        type=int,
        default=512,
        help="Safety buffer added on top of max prompt tokens plus max generation tokens.",
    )
    parser.add_argument(
        "--context-round-to",
        type=int,
        default=1024,
        help="Round the recommended context window up to this token multiple.",
    )
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {
                    "error_type": "MalformedJsonLine",
                    "error_message": str(exc),
                    "line_number": line_number,
                }


def update_top(heap, item, key, limit, tie_breaker=0):
    value = item.get(key)
    if value is None:
        return
    entry = (value, tie_breaker, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif value > heap[0][0]:
        heapq.heapreplace(heap, entry)


def summarize_errors(path, top_n):
    counters = Counter()
    prompt_token_stats = {
        "count": 0,
        "min": None,
        "max": None,
        "sum": 0,
    }
    top_prompt_errors = []
    invalid_answers = []
    examples_by_type = {}
    total = 0

    for line_number, record in iter_jsonl(path):
        total += 1
        error_type = record.get("error_type", "Unknown")
        counters[error_type] += 1
        examples_by_type.setdefault(error_type, record)

        prompt_tokens = record.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            prompt_token_stats["count"] += 1
            prompt_token_stats["sum"] += prompt_tokens
            prompt_token_stats["min"] = prompt_tokens if prompt_token_stats["min"] is None else min(prompt_token_stats["min"], prompt_tokens)
            prompt_token_stats["max"] = prompt_tokens if prompt_token_stats["max"] is None else max(prompt_token_stats["max"], prompt_tokens)
            update_top(top_prompt_errors, record, "prompt_tokens", top_n, line_number)

        if error_type == "InvalidAnswer" and len(invalid_answers) < top_n:
            invalid_answers.append({**record, "line_number": line_number})

    prompt_token_stats["average"] = (
        prompt_token_stats["sum"] / prompt_token_stats["count"]
        if prompt_token_stats["count"]
        else None
    )

    return {
        "total_error_lines": total,
        "error_type_counts": dict(counters),
        "prompt_token_stats": prompt_token_stats,
        "top_prompt_errors": [item for _, _, item in sorted(top_prompt_errors, reverse=True)],
        "invalid_answers": invalid_answers,
        "examples_by_type": examples_by_type,
    }


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


def summarize_prompt_load(path, model_path, thresholds, top_n):
    tokenizer = load_tokenizer(model_path)
    count = 0
    total = 0
    minimum = None
    maximum = None
    over_threshold = {str(threshold): 0 for threshold in thresholds}
    top_prompts = []

    for _, record in iter_jsonl(path):
        text = render_prompt(tokenizer, record["messages"])
        token_count = len(tokenizer(text).input_ids)
        count += 1
        total += token_count
        minimum = token_count if minimum is None else min(minimum, token_count)
        maximum = token_count if maximum is None else max(maximum, token_count)
        for threshold in thresholds:
            if token_count > threshold:
                over_threshold[str(threshold)] += 1
        update_top(
            top_prompts,
            {
                "request_id": record.get("request_id"),
                "dataset": record.get("dataset"),
                "retriever": record.get("retriever"),
                "experiment": record.get("experiment"),
                "question_number": record.get("question_number"),
                "output_file": record.get("output_file"),
                "prompt_tokens": token_count,
            },
            "prompt_tokens",
            top_n,
            count,
        )

    return {
        "count": count,
        "min_tokens": minimum,
        "max_tokens": maximum,
        "average_tokens": total / count if count else None,
        "over_threshold": over_threshold,
        "top_prompts": [item for _, _, item in sorted(top_prompts, reverse=True)],
    }


def pct(part, whole):
    if not whole:
        return "0.00%"
    return f"{(part / whole) * 100:.2f}%"


def recommended_context(summary, prompt_summary, buffer_tokens, round_to):
    max_tokens = summary.get("generation", {}).get("max_tokens")
    if prompt_summary and prompt_summary.get("max_tokens") and max_tokens:
        minimum = prompt_summary["max_tokens"] + max_tokens
        buffered = minimum + buffer_tokens
        rounded = int(math.ceil(buffered / round_to) * round_to)
        return {
            "max_prompt_tokens": prompt_summary["max_tokens"],
            "max_generation_tokens": max_tokens,
            "buffer_tokens": buffer_tokens,
            "round_to": round_to,
            "minimum_context_without_buffer": minimum,
            "minimum_context_with_buffer": buffered,
            "recommended_generate_max_model_len": rounded,
            "hpc_private_env_line": f'export GENERATE_MAX_MODEL_LEN="{rounded}"',
        }
    return None


def write_reports(output_dir, run_name, payload):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}.json"
    md_path = output_dir / f"{run_name}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    summary = payload["summary"]
    errors = payload["errors"]
    prompt_summary = payload.get("prompt_load")
    recommendation = payload.get("recommendation")

    lines = [
        f"# vLLM Run Diagnostic: {run_name}",
        "",
        "## Summary",
        "",
        f"- Prompt load: `{summary.get('prompt_load')}`",
        f"- Output dir: `{summary.get('output_dir')}`",
        f"- Prompt count: `{summary.get('prompt_count')}`",
        f"- Submitted prompts: `{summary.get('submitted_prompt_count')}`",
        f"- Parsed predictions: `{summary.get('parsed_record_count')}`",
        f"- Error records: `{summary.get('error_record_count')}`",
        f"- Usable prediction rate: `{pct(summary.get('parsed_record_count', 0), summary.get('prompt_count', 0))}`",
        "",
        "## Error Types",
        "",
    ]

    for error_type, count in sorted(errors["error_type_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{error_type}`: `{count}`")

    token_stats = errors["prompt_token_stats"]
    lines.extend([
        "",
        "## Error Prompt Token Stats",
        "",
        f"- Count with token data: `{token_stats['count']}`",
        f"- Min: `{token_stats['min']}`",
        f"- Max: `{token_stats['max']}`",
        f"- Average: `{token_stats['average']:.2f}`" if token_stats["average"] is not None else "- Average: `n/a`",
    ])

    if prompt_summary:
        lines.extend([
            "",
            "## Full Prompt Load Token Stats",
            "",
            f"- Count: `{prompt_summary['count']}`",
            f"- Min: `{prompt_summary['min_tokens']}`",
            f"- Max: `{prompt_summary['max_tokens']}`",
            f"- Average: `{prompt_summary['average_tokens']:.2f}`",
            "",
            "### Over Threshold",
            "",
        ])
        for threshold, count in prompt_summary["over_threshold"].items():
            lines.append(f"- `>{threshold}`: `{count}`")

    if recommendation:
        lines.extend([
            "",
            "## Context Recommendation",
            "",
            f"- Max prompt tokens: `{recommendation['max_prompt_tokens']}`",
            f"- Max generation tokens: `{recommendation['max_generation_tokens']}`",
            f"- Safety buffer tokens: `{recommendation['buffer_tokens']}`",
            f"- Minimum context without buffer: `{recommendation['minimum_context_without_buffer']}`",
            f"- Minimum context with buffer: `{recommendation['minimum_context_with_buffer']}`",
            f"- Rounded to token multiple: `{recommendation['round_to']}`",
            f"- Recommended `GENERATE_MAX_MODEL_LEN`: `{recommendation['recommended_generate_max_model_len']}`",
            f"- Add to `llm_frag_evaluation/slurm/hpc.private.env`: `{recommendation['hpc_private_env_line']}`",
        ])

    if errors["top_prompt_errors"]:
        lines.extend(["", "## Longest Error Prompts", ""])
        for record in errors["top_prompt_errors"][:10]:
            lines.append(
                f"- `{record.get('output_file')}` `{record.get('request_id')}`: "
                f"`{record.get('prompt_tokens')}` tokens, `{record.get('error_type')}`"
            )

    if errors["invalid_answers"]:
        lines.extend(["", "## Invalid Answers", ""])
        for record in errors["invalid_answers"]:
            lines.append(
                f"- `{record.get('output_file')}` `{record.get('request_id')}` parsed "
                f"`{record.get('parsed_answer_choice')}`"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main():
    args = parse_args()
    thresholds = args.threshold if args.threshold else DEFAULT_THRESHOLDS
    summary = read_json(args.summary)
    errors = summarize_errors(args.errors, args.top_n)
    prompt_summary = None
    if args.prompt_load and args.model_path:
        prompt_summary = summarize_prompt_load(args.prompt_load, args.model_path, thresholds, args.top_n)

    run_name = args.run_name
    if not run_name:
        prompt_load = Path(summary.get("prompt_load", "run")).parts
        run_name = "_".join(prompt_load[-6:-1]) if len(prompt_load) >= 6 else "vllm_run"

    payload = {
        "summary": summary,
        "errors": errors,
        "prompt_load": prompt_summary,
        "recommendation": recommended_context(
            summary,
            prompt_summary,
            args.context_buffer_tokens,
            args.context_round_to,
        ),
    }
    json_path, md_path = write_reports(args.output_dir, run_name, payload)
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
