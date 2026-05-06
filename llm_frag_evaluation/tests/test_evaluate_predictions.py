import unittest

from llm_frag_evaluation.scripts.evaluate_predictions import normalize_answer


class EvaluatePredictionsTests(unittest.TestCase):
    def test_pubmedqa_normalizes_letter_and_text_answers(self):
        self.assertEqual(normalize_answer("A", "pubmedqa"), "yes")
        self.assertEqual(normalize_answer("B", "pubmedqa"), "no")
        self.assertEqual(normalize_answer("C", "pubmedqa"), "maybe")
        self.assertEqual(normalize_answer("yes", "pubmedqa"), "yes")
        self.assertEqual(normalize_answer("Maybe", "pubmedqa"), "maybe")

    def test_bioasq_normalizes_letter_and_text_answers(self):
        self.assertEqual(normalize_answer("A", "bioasq"), "yes")
        self.assertEqual(normalize_answer("B", "bioasq"), "no")
        self.assertEqual(normalize_answer("yes", "bioasq"), "yes")
        self.assertEqual(normalize_answer("No", "bioasq"), "no")

    def test_mcq_keeps_letter_answers(self):
        self.assertEqual(normalize_answer("A", "medqa"), "A")
        self.assertEqual(normalize_answer("B. option text", "mmlu"), "B")


if __name__ == "__main__":
    unittest.main()
