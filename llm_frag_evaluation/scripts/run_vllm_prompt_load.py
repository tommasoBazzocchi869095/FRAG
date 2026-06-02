import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[0]
sys.path.insert(0, str(REPO_ROOT))

from llm_frag_evaluation.src.config import resolve_repo_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a prompt-load JSONL file with vLLM.")
    parser.add_argument("--prompt-load", required=True, help="Path to prompts.jsonl.")
    parser.add_argument("--model-path", required=True, help="HF model id or local model path.")
    parser.add_argument("--output-dir", default=None, help="Prediction output directory. Derived from prompt-load when omitted.")
    parser.add_argument("--raw-output-path", default=None, help="Raw generation JSONL path. Derived from output-dir when omitted.")
    parser.add_argument("--error-output-path", default=None, help="Parse error JSONL path. Derived from output-dir when omitted.")
    parser.add_argument("--summary-path", default=None, help="Run summary JSON path. Derived from output-dir when omitted.")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test prompt limit.")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_prompt_load(path, limit=None):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def derive_output_dir(prompt_load):
    path = Path(prompt_load).resolve()
    parts = path.parts
    try:
        idx = parts.index("prompt_loads")
    except ValueError:
        return Path("llm_frag_evaluation") / "outputs" / "predictions" / "unknown"
    relative = Path(*parts[idx + 1 : -1])
    return Path("llm_frag_evaluation") / "outputs" / "predictions" / relative


def render_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"


def batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_error_record(record, error_type, error_message, raw_record=None):
    payload = {
        "request_id": record.get("request_id"),
        "dataset": record.get("dataset"),
        "retriever": record.get("retriever"),
        "experiment": record.get("experiment"),
        "question_number": record.get("question_number"),
        "output_file": record.get("output_file"),
        "error_type": error_type,
        "error_message": error_message,
    }
    if raw_record:
        payload.update(raw_record)
    return payload


def write_jsonl(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def parse_answer(raw_answer, dataset):
    answer_choice = "NA"
    step_by_step = ""
    text = raw_answer.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            answer_choice = str(data.get("answer_choice", "NA")).strip()
            step_by_step = str(data.get("step_by_step_thinking", "")).strip()
            return normalize_answer(answer_choice, dataset), step_by_step
    except Exception:
        pass

    answer_match = re.search(r'[\'"]answer_choice[\'"]\s*:\s*[\'"]([^\'"]+)[\'"]', text, re.IGNORECASE)
    if answer_match:
        answer_choice = answer_match.group(1).strip()

    thinking_match = re.search(r'[\'"]step_by_step_thinking[\'"]\s*:\s*[\'"](.+?)[\'"]\s*[,}]', text, re.IGNORECASE | re.DOTALL)
    if thinking_match:
        step_by_step = thinking_match.group(1).strip()

    if answer_choice == "NA":
        if dataset == "pubmedqa":
            match = re.search(r"\b(yes|no|maybe)\b", text, re.IGNORECASE)
        elif dataset == "bioasq":
            match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
        else:
            match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
        if match:
            answer_choice = match.group(1)

    return normalize_answer(answer_choice, dataset), step_by_step


def normalize_answer(answer, dataset):
    answer = str(answer).strip()
    if dataset in {"pubmedqa", "bioasq"}:
        return answer.lower()
    if len(answer) > 1 and answer[0].isalpha():
        return answer[0].upper()
    return answer.upper()


def is_valid_answer(answer, dataset):
    if dataset == "pubmedqa":
        return answer in {"yes", "no", "maybe"}
    if dataset == "bioasq":
        return answer in {"yes", "no"}
    return answer in {"A", "B", "C", "D"}


def write_prediction(output_dir, record, answer_choice, step_by_step):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / record["output_file"]
    payload = [{
        "answer_choice": answer_choice,
        "step_by_step_thinking": step_by_step,
        "system_info": "Offline Factuality",
    }]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def generate_batch(llm, sampling_params, batch):
    prompts = [record["model_input_text"] for record in batch]
    return llm.generate(prompts, sampling_params)


def main():
    args = parse_args()
    prompt_load = resolve_repo_path(args.prompt_load)
    records = read_prompt_load(prompt_load, limit=args.limit)
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else resolve_repo_path(derive_output_dir(prompt_load))
    raw_output_path = resolve_repo_path(args.raw_output_path) if args.raw_output_path else output_dir / "model_outputs_raw.jsonl"
    error_output_path = resolve_repo_path(args.error_output_path) if args.error_output_path else output_dir / "generation_errors.jsonl"
    summary_path = resolve_repo_path(args.summary_path) if args.summary_path else output_dir / "run_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("vLLM is required. Activate the CINECA vLLM environment before running this script.") from exc

    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    to_run = []
    skipped = 0
    preflight_error_records = []
    for record in records:
        if args.skip_existing and (output_dir / record["output_file"]).exists():
            skipped += 1
            continue
        record["model_input_text"] = render_prompt(tokenizer, record["messages"])
        prompt_token_count = len(tokenizer(record["model_input_text"]).input_ids)
        record["prompt_token_count_preflight"] = prompt_token_count
        if prompt_token_count > args.max_model_len:
            preflight_error_records.append(
                build_error_record(
                    record,
                    "PromptTooLong",
                    f"Prompt has {prompt_token_count} tokens, exceeding max_model_len={args.max_model_len}.",
                    {"prompt_tokens": prompt_token_count},
                )
            )
            continue
        to_run.append(record)

    start_time = time.time()
    parsed_count = 0
    error_count = 0
    error_types = Counter()
    prompt_tokens_total = 0
    generated_tokens_total = 0

    with raw_output_path.open("a", encoding="utf-8") as raw_f, error_output_path.open("a", encoding="utf-8") as err_f:
        for error_record in preflight_error_records:
            error_count += 1
            error_types[error_record["error_type"]] += 1
            write_jsonl(err_f, error_record)

        for batch in batches(to_run, args.batch_size):
            try:
                outputs = generate_batch(llm, sampling_params, batch)
                batch_outputs = list(zip(batch, outputs))
            except Exception as exc:
                batch_outputs = []
                for record in batch:
                    try:
                        output = generate_batch(llm, sampling_params, [record])[0]
                        batch_outputs.append((record, output))
                    except Exception as record_exc:
                        error_count += 1
                        error_types[type(record_exc).__name__] += 1
                        write_jsonl(
                            err_f,
                            build_error_record(
                                record,
                                type(record_exc).__name__,
                                str(record_exc),
                                {"prompt_tokens": record.get("prompt_token_count_preflight", 0)},
                            ),
                        )

            for record, output in batch_outputs:
                generated_text = output.outputs[0].text if output.outputs else ""
                prompt_tokens = len(output.prompt_token_ids or [])
                generated_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
                prompt_tokens_total += prompt_tokens
                generated_tokens_total += generated_tokens

                raw_record = {
                    "request_id": record["request_id"],
                    "dataset": record["dataset"],
                    "retriever": record["retriever"],
                    "experiment": record["experiment"],
                    "question_number": record["question_number"],
                    "output_file": record["output_file"],
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "raw_output": generated_text,
                }
                write_jsonl(raw_f, raw_record)

                answer_choice, step_by_step = parse_answer(generated_text, record["dataset"])
                if not is_valid_answer(answer_choice, record["dataset"]):
                    error_count += 1
                    error_types["InvalidAnswer"] += 1
                    write_jsonl(
                        err_f,
                        build_error_record(
                            record,
                            "InvalidAnswer",
                            "Generated answer_choice is not valid for this dataset.",
                            {**raw_record, "parsed_answer_choice": answer_choice},
                        ),
                    )
                    continue

                write_prediction(output_dir, record, answer_choice, step_by_step)
                parsed_count += 1

    elapsed = time.time() - start_time
    summary = {
        "prompt_load": str(prompt_load),
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "prompt_count": len(records),
        "skipped_existing": skipped,
        "preflight_error_count": len(preflight_error_records),
        "submitted_prompt_count": len(to_run),
        "parsed_record_count": parsed_count,
        "error_record_count": error_count,
        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": generated_tokens_total,
        "average_prompt_tokens": prompt_tokens_total / len(to_run) if to_run else 0,
        "average_generated_tokens": generated_tokens_total / len(to_run) if to_run else 0,
        "total_elapsed_seconds": elapsed,
        "average_seconds_per_prompt": elapsed / len(to_run) if to_run else 0,
        "error_type_counts": dict(error_types),
        "generation": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "max_num_seqs": args.max_num_seqs,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "enforce_eager": args.enforce_eager,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
