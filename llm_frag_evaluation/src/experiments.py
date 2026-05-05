from .data_loader import get_passages
from .prompts import build_prompt
from .scoring import select_frag_passages, select_standard_rag_passages


EXPERIMENTS = {"zero_shot", "standard_rag", "frag"}


def prepare_experiment_item(question, experiment, templates, top_k_context=32, alpha=0.6, normalize_topic=True):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment: {experiment}")

    passages = get_passages(question)
    if experiment == "zero_shot":
        selected_passages = []
    elif experiment == "standard_rag":
        selected_passages = select_standard_rag_passages(passages, top_k_context)
    else:
        selected_passages = select_frag_passages(
            passages,
            top_k=top_k_context,
            alpha=alpha,
            normalize_topic=normalize_topic,
        )

    prompt = build_prompt(question, experiment, selected_passages, templates)
    return {
        "id": question.get("id"),
        "dataset": question.get("dataset"),
        "experiment": experiment,
        "answer": question.get("answer"),
        "selected_passages": selected_passages,
        "prompt": prompt,
    }
