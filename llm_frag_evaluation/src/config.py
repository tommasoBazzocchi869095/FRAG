import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_repo_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate
