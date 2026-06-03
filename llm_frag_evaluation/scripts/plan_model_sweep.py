import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
from shlex import quote


DEFAULT_CONFIG = "llm_frag_evaluation/configs/model_sweep.json"
PROFILE_DIR = "llm_frag_evaluation/slurm/model_profiles"
SUBMIT_SCRIPT = "llm_frag_evaluation/slurm/sh/submit_generate_prompt_load.sh"
SMOKE_SCRIPT = "llm_frag_evaluation/slurm/sh/submit_generate_prompt_load_smoke.sh"


def parse_args():
    parser = argparse.ArgumentParser(description="Print CINECA model-sweep setup and submission commands.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--family", choices=["biomedical", "llama", "mistral", "qwen"], default=None)
    parser.add_argument("--alias", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--section",
        choices=["summary", "profiles", "prompt-loads", "smoke", "submit", "all"],
        default="summary",
    )
    parser.add_argument("--profile-dir", default=PROFILE_DIR)
    parser.add_argument("--write-profiles", action="store_true", help="Write model-specific env files.")
    parser.add_argument("--overwrite-profiles", action="store_true", help="Replace existing profile env files.")
    parser.add_argument("--hpc-account", default="IscrC_SpecDLM")
    parser.add_argument("--hpc-qos", default="normal")
    parser.add_argument("--hpc-partition", default="boost_usr_prod")
    parser.add_argument("--vllm-venv-activate", default="/leonardo_work/IscrC_SpecDLM/FRAG/.venv_frag_vllm/bin/activate")
    return parser.parse_args()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_models(config, args):
    models = config["models"]
    if args.family:
        models = [model for model in models if model["family"] == args.family]
    if args.alias:
        models = [model for model in models if model["alias"] == args.alias]
    return models


def select_collections(config, args):
    collections = config["collections"]
    if args.collection:
        collections = [collection for collection in collections if collection["name"] == args.collection]
        if not collections:
            available = ", ".join(collection["name"] for collection in config["collections"])
            raise SystemExit(f"Unknown collection {args.collection!r}. Available collections: {available}")
    return collections


def profile_env_path(profile_dir, alias):
    return PurePosixPath(profile_dir) / f"{alias}.env"


def local_profile_env_path(profile_dir, alias):
    return Path(profile_dir) / f"{alias}.env"


def model_path(config, model):
    return f"{config['model_root'].rstrip('/')}/{model['local_dir']}"


def prompt_load_path(collection_name, dataset, retriever, experiment, alias):
    return (
        PurePosixPath("llm_frag_evaluation/outputs/prompt_loads")
        / collection_name
        / dataset
        / retriever
        / experiment
        / alias
        / "prompts.jsonl"
    )


def iter_jobs(config, collections, model):
    zero_retriever = config.get("zero_shot_retriever", "bm25")
    for collection in collections:
        for dataset in config.get("datasets", ["mmlu", "medqa", "pubmedqa", "bioasq"]):
            for retriever in config.get("retrievers", ["bm25", "contriever"]):
                for experiment in config.get("experiments", ["zero_shot", "standard_rag", "frag"]):
                    if experiment == "zero_shot" and retriever != zero_retriever:
                        continue
                    yield collection, dataset, retriever, experiment


def print_summary(config, models, collections):
    total_gpus = config["hardware_assumption"]["available_gpus"]
    print("# Sweep Summary")
    print(f"Selected models: {len(models)}")
    print(f"Selected collections: {', '.join(collection['name'] for collection in collections)}")
    print(f"Available GPUs assumed: {total_gpus} x {config['hardware_assumption']['gpu_type']}")
    print()
    print("| Profile | Models | GPUs/job | Max concurrent on 16 GPUs | Initial batch | Max seqs |")
    print("|---|---:|---:|---:|---:|---:|")
    for profile_name, profile in config["profiles"].items():
        count = sum(1 for model in models if model["profile"] == profile_name)
        print(
            f"| {profile_name} | {count} | {profile['gpus']} | "
            f"{total_gpus // profile['gpus']} | {profile['initial_batch_size']} | {profile['max_num_seqs']} |"
        )
    print()
    print("First run order: one smoke job per family/profile, then one full short dataset, then the full matrix.")


def profile_lines(config, model, collections, args):
    max_model_len = model.get("max_model_len", max(collection["initial_max_model_len"] for collection in collections))
    profile = config["profiles"][model["profile"]]
    return [
        'export ENV_INIT_COMMAND="${ENV_INIT_COMMAND:-module purge; module load python/3.11.7}"',
        f"export VLLM_VENV_ACTIVATE={quote(args.vllm_venv_activate)}",
        f"export HPC_ACCOUNT={quote(args.hpc_account)}",
        f"export HPC_QOS={quote(args.hpc_qos)}",
        f"export HPC_PARTITION={quote(args.hpc_partition)}",
        f"export HPC_MODEL_PATH={quote(model_path(config, model))}",
        f"export GENERATE_TIME_LIMIT={quote(profile['time_limit'])}",
        f"export GENERATE_CPUS_PER_TASK=\"{profile['cpus_per_task']}\"",
        f"export GENERATE_MEM={quote(profile['mem'])}",
        f"export GENERATE_GPUS=\"{profile['gpus']}\"",
        f"export GENERATE_TENSOR_PARALLEL_SIZE=\"{profile['tensor_parallel_size']}\"",
        f"export GENERATE_MAX_MODEL_LEN=\"{max_model_len}\"",
        f"export GENERATE_DTYPE={quote(profile['dtype'])}",
        f"export GENERATE_MAX_NUM_SEQS=\"{profile['max_num_seqs']}\"",
        f"export GENERATE_GPU_MEMORY_UTILIZATION=\"{profile['gpu_memory_utilization']}\"",
        f"export GENERATE_MAX_TOKENS=\"{profile['max_tokens']}\"",
        f"export GENERATE_BATCH_SIZE=\"{profile['initial_batch_size']}\"",
        'export SMOKE_PROMPT_LIMIT="7"',
    ]


def print_profiles(config, models, collections, profile_dir, args):
    for model in models:
        env_path = profile_env_path(profile_dir, model["alias"])
        print(f"# {env_path}")
        print("\n".join(profile_lines(config, model, collections, args)))
        print()


def write_profiles(config, models, collections, profile_dir, overwrite, args):
    output_dir = Path(profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    skipped = []
    for model in models:
        env_path = local_profile_env_path(profile_dir, model["alias"])
        if env_path.exists() and not overwrite:
            skipped.append(env_path)
            continue
        text = "\n".join(profile_lines(config, model, collections, args)) + "\n"
        env_path.write_text(text, encoding="utf-8")
        written.append(env_path)
    for env_path in written:
        print(f"Wrote {env_path}")
    for env_path in skipped:
        print(f"Skipped existing {env_path}; pass --overwrite-profiles to replace it.")


def print_prompt_loads(models, collections):
    for model in models:
        for collection in collections:
            print(
                "python llm_frag_evaluation/scripts/create_prompt_loads.py "
                f"--config {quote(collection['config'])} "
                f"--model {quote(model['repo_id'])} "
                f"--model-alias {quote(model['alias'])}"
            )


def smoke_source_prompt(collection_name, alias):
    if collection_name == "source_collection_wiki":
        experiment = "standard_rag"
    else:
        experiment = "frag"
    return (
        PurePosixPath("llm_frag_evaluation/outputs/prompt_loads")
        / collection_name
        / "medqa"
        / "contriever"
        / experiment
        / alias
        / "prompts.jsonl"
    )


def print_smoke(config, models, collections):
    collection_name = collections[0]["name"] if collections else "source_collection_wiki"
    for model in models:
        run_name = f"{model['alias']}_{collection_name}_medqa_longest"
        source = smoke_source_prompt(collection_name, model["alias"])
        smoke_prompt = (
            PurePosixPath("llm_frag_evaluation/outputs/prompt_loads/diagnostics")
            / run_name
            / model["alias"]
            / "prompts.jsonl"
        )
        print(
            "python llm_frag_evaluation/tests/diagnostics/create_long_prompt_smoke_load.py "
            f"--prompt-load {quote(str(source))} "
            f"--model-path {quote(model_path(config, model))} "
            f"--run-name {quote(run_name)} "
            f"--model-alias {quote(model['alias'])} "
            "--top-longest 5 --include-first 2"
        )
        print(f"bash {SMOKE_SCRIPT} --model-alias {quote(model['alias'])} {quote(str(smoke_prompt))}")
        print()


def print_submit(config, models, collections, profile_dir):
    for model in models:
        for collection, dataset, retriever, experiment in iter_jobs(config, collections, model):
            prompt_path = prompt_load_path(collection["name"], dataset, retriever, experiment, model["alias"])
            print(f"bash {SUBMIT_SCRIPT} --model-alias {quote(model['alias'])} {quote(str(prompt_path))}")


def main():
    args = parse_args()
    config = load_config(args.config)
    models = select_models(config, args)
    collections = select_collections(config, args)
    if args.write_profiles:
        write_profiles(config, models, collections, args.profile_dir, args.overwrite_profiles, args)
        return
    sections = ["summary", "profiles", "prompt-loads", "smoke", "submit"] if args.section == "all" else [args.section]

    for index, section in enumerate(sections):
        if index:
            print()
        if section == "summary":
            print_summary(config, models, collections)
        elif section == "profiles":
            print_profiles(config, models, collections, args.profile_dir, args)
        elif section == "prompt-loads":
            print_prompt_loads(models, collections)
        elif section == "smoke":
            print_smoke(config, models, collections)
        elif section == "submit":
            print_submit(config, models, collections, args.profile_dir)


if __name__ == "__main__":
    main()
