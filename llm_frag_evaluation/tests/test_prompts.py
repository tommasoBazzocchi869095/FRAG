import unittest

from llm_frag_evaluation.src.prompts import build_prompt, format_context, load_prompt_templates


TEMPLATES = load_prompt_templates("llm_frag_evaluation/prompts/default_prompts.json")


class PromptTests(unittest.TestCase):
    def test_zero_shot_mcq_has_no_document_context(self):
        prompt = build_prompt(
            {
                "id": "q1",
                "dataset": "medqa",
                "question": "What is the best answer?",
                "options": {
                    "A": "Alpha",
                    "B": "Beta",
                    "C": "Gamma",
                    "D": "Delta",
                },
            },
            "zero_shot",
            [],
            TEMPLATES,
        )

        self.assertEqual(prompt["messages"][0]["role"], "system")
        self.assertEqual(prompt["messages"][1]["role"], "user")
        self.assertNotIn("using the relevant documents", prompt["system"])
        self.assertNotIn("Here are the relevant documents:", prompt["user"])
        self.assertIn("Here is the question:", prompt["user"])
        self.assertIn("Question:\nWhat is the best answer?", prompt["user"])
        self.assertIn("A. Alpha\nB. Beta\nC. Gamma\nD. Delta", prompt["user"])
        self.assertIn("Here are the potential choices:\n\n\nPlease think step-by-step", prompt["user"])

    def test_standard_rag_mcq_matches_medrag_wrapper(self):
        prompt = build_prompt(
            {
                "id": "q1",
                "dataset": "mmlu",
                "question": "A lesion causes what?",
                "options": {
                    "A": "Facial paralysis.",
                    "B": "Taste loss.",
                    "C": "Lacrimation.",
                    "D": "Salivation.",
                },
            },
            "standard_rag",
            [
                {
                    "id": "p1",
                    "title": "Stylomastoid foramen",
                    "content": "The facial nerve exits through the stylomastoid foramen.",
                },
                {
                    "id": "p2",
                    "title": "Facial muscles",
                    "content": "The facial nerve supplies muscles of facial expression.",
                },
            ],
            TEMPLATES,
        )

        self.assertIn("using the relevant documents", prompt["system"])
        self.assertIn(
            "\nHere are the relevant documents:\n"
            "Document [0] (Title: Stylomastoid foramen) The facial nerve exits through the stylomastoid foramen.\n"
            "Document [1] (Title: Facial muscles) The facial nerve supplies muscles of facial expression.\n\n"
            "Here is the question:",
            prompt["user"],
        )
        self.assertIn("Please think step-by-step and generate your output in json:\n", prompt["user"])

    def test_frag_uses_same_prompt_shape_as_standard_rag(self):
        question = {
            "id": "q1",
            "dataset": "pubmedqa",
            "question": "Is the intervention effective?",
        }
        passages = [{"id": "p1", "title": "Trial", "content": "The trial reported efficacy."}]

        standard = build_prompt(question, "standard_rag", passages, TEMPLATES)
        frag = build_prompt(question, "frag", passages, TEMPLATES)

        self.assertEqual(standard["system"], frag["system"])
        self.assertEqual(standard["user"], frag["user"])
        self.assertIn("Possible answers: yes, no, or maybe.", frag["user"])
        self.assertIn('{ "answer_choice": "yes"', frag["user"])

    def test_bioasq_zero_shot_uses_yes_no_schema(self):
        prompt = build_prompt(
            {
                "id": "q1",
                "dataset": "bioasq",
                "question": "Are there microbes in human breast milk?",
            },
            "zero_shot",
            [],
            TEMPLATES,
        )

        self.assertNotIn("Here are the relevant documents:", prompt["user"])
        self.assertIn("Possible answers: yes or no.", prompt["user"])
        self.assertIn('{ "answer_choice": "yes"', prompt["user"])

    def test_context_numbering_starts_at_zero(self):
        context = format_context(
            [
                {"id": "p1", "title": "First", "content": "One."},
                {"id": "p2", "title": "Second", "content": "Two."},
            ]
        )

        self.assertEqual(
            context,
            "Document [0] (Title: First) One.\nDocument [1] (Title: Second) Two.",
        )


if __name__ == "__main__":
    unittest.main()
