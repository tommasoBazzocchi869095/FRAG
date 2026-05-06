import json
from pathlib import Path


MEDRAG_SYSTEM = (
    "You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant "
    "documents. Please first think step-by-step and then choose the answer from the provided options. Organize your "
    "output in a json formatted as Dict{\"step_by_step_thinking\": Str(explanation), \"answer_choice\": "
    "Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer."
)

COT_SYSTEM = (
    "You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think "
    "step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as "
    "Dict{\"step_by_step_thinking\": Str(explanation), \"answer_choice\": Str{A/B/C/...}}. Your responses will be "
    "used for research purposes only, so please have a definite answer."
)


def load_prompt_templates(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_options(options):
    if not options:
        return "No explicit options provided."
    if isinstance(options, dict):
        return "\n".join(f"{key}. {value}" for key, value in options.items())
    if isinstance(options, list):
        return "\n".join(str(item) for item in options)
    return str(options)


def format_context(passages):
    chunks = []
    for idx, passage in enumerate(passages):
        title = passage.get("title") or passage.get("id") or f"Passage {idx}"
        content = passage.get("content") or passage.get("contents") or ""
        chunks.append(f"Document [{idx}] (Title: {title}) {content}")
    return "\n".join(chunks)


def get_question_type(dataset):
    if dataset in {"mmlu", "medqa", "medmcqa", "sample"}:
        return "mcq"
    if dataset == "pubmedqa":
        return "ynm"
    if dataset == "bioasq":
        return "yn"
    return "mcq"


def format_dataset_question(question):
    dataset = question.get("dataset")
    question_type = get_question_type(dataset)
    question_text = question.get("question", "")

    if question_type == "mcq":
        options_text = format_options(question.get("options"))
        return (
            "You are a medical expert. Read the following question carefully and choose the single best answer.\n\n"
            f"Question:\n{question_text}\n\n"
            f"{options_text}\n\n"
            "Answer the question and then briefly explain your reasoning.\n"
            "Your response must be in JSON format with exactly these keys:\n"
            '{ "answer_choice": "A", "step_by_step_thinking": "your explanation here" }'
        )

    if question_type == "ynm":
        return (
            "You are a biomedical researcher. Read the following question carefully.\n\n"
            f"Question:\n{question_text}\n\n"
            "Possible answers: yes, no, or maybe.\n"
            "Respond strictly in JSON format as follows:\n"
            '{ "answer_choice": "yes", "step_by_step_thinking": "your short biomedical justification" }'
        )

    return (
        "You are a biomedical researcher. Read the following question carefully.\n\n"
        f"Question:\n{question_text}\n\n"
        "Possible answers: yes or no.\n"
        "Respond strictly in JSON format as follows:\n"
        '{ "answer_choice": "yes", "step_by_step_thinking": "your short biomedical justification" }'
    )


def build_prompt(question, experiment, selected_passages, templates):
    formatted_question = format_dataset_question(question)
    context = format_context(selected_passages)
    system_templates = templates.get("system", {})
    if experiment == "zero_shot":
        system = system_templates.get("zero_shot", COT_SYSTEM)
    else:
        system = system_templates.get("medical_qa", MEDRAG_SYSTEM)
    user_template = templates[experiment]["user"]
    user = user_template.format(
        question=formatted_question,
        options="",
        context=context,
    )
    return {
        "system": system,
        "user": user,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
