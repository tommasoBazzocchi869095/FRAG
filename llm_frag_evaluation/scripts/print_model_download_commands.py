import argparse
import json
from pathlib import Path
from shlex import quote


DEFAULT_CONFIG = "llm_frag_evaluation/configs/model_sweep.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Print CINECA Hugging Face model download commands.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--family", choices=["biomedical", "llama", "qwen"], default=None)
    parser.add_argument("--alias", default=None, help="Print only one model alias.")
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    models = config["models"]
    if args.family:
        models = [model for model in models if model["family"] == args.family]
    if args.alias:
        models = [model for model in models if model["alias"] == args.alias]

    print("# Run on CINECA after setting HF_TOKEN and activating the download environment.")
    print("export HF_TOKEN=\"...\"")
    print()
    for model in models:
        model_path = f"{config['model_root'].rstrip('/')}/{model['local_dir']}"
        print(f"# {model['alias']}")
        print(f"# {model['hf_url']}")
        print(f"export MODEL_ID={quote(model['repo_id'])}")
        print(f"export MODEL_PATH={quote(str(model_path))}")
        print(
            "python -c "
            + quote(
                "import os; "
                "from huggingface_hub import snapshot_download; "
                "snapshot_download("
                "repo_id=os.environ['MODEL_ID'], "
                "local_dir=os.environ['MODEL_PATH'], "
                "token=os.environ.get('HF_TOKEN'), "
                f"max_workers={args.max_workers})"
            )
        )
        print()


if __name__ == "__main__":
    main()
