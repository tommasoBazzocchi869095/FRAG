import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from llm_frag_evaluation.scripts.create_prompt_loads import (
    experiments_for_retriever,
    get_input_files,
    remove_stale_zero_shot_prompt_loads,
)
from llm_frag_evaluation.scripts.validate_prompt_loads import expected_experiments_for_retriever


class PromptLoadRetrieverPolicyTests(unittest.TestCase):
    def test_zero_shot_is_kept_for_primary_retriever(self):
        experiments = ["zero_shot", "standard_rag", "frag"]

        self.assertEqual(
            experiments_for_retriever(experiments, "bm25", "bm25"),
            experiments,
        )

    def test_zero_shot_is_skipped_for_non_primary_retriever(self):
        experiments = ["zero_shot", "standard_rag", "frag"]

        self.assertEqual(
            experiments_for_retriever(experiments, "contriever", "bm25"),
            ["standard_rag", "frag"],
        )

    def test_zero_shot_is_kept_for_unknown_sample_retriever(self):
        experiments = ["zero_shot", "standard_rag", "frag"]

        self.assertEqual(
            experiments_for_retriever(experiments, "unknown_retriever", "bm25"),
            experiments,
        )

    def test_validator_uses_same_policy(self):
        config = {
            "experiments": ["zero_shot", "standard_rag", "frag"],
            "zero_shot_retriever": "bm25",
        }

        self.assertEqual(
            expected_experiments_for_retriever(config, "contriever"),
            ["standard_rag", "frag"],
        )

    def test_stale_non_primary_zero_shot_dirs_are_removed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / "medqa" / "contriever" / "zero_shot"
            kept = root / "medqa" / "bm25" / "zero_shot"
            stale.mkdir(parents=True)
            kept.mkdir(parents=True)
            (stale / "prompts.jsonl").write_text("stale", encoding="utf-8")
            (kept / "prompts.jsonl").write_text("kept", encoding="utf-8")

            remove_stale_zero_shot_prompt_loads(root, "bm25")

            self.assertFalse(stale.exists())
            self.assertTrue(kept.exists())

    def test_all_input_files_uses_configured_collection(self):
        class Args:
            all_input_files = True
            input_collection = None
            input_file = None

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "source_collection_wiki"
            pubmed = root / "source_collection_pubmed"
            wiki.mkdir()
            pubmed.mkdir()
            (wiki / "cache_step2_medqa_scored_bm25.json").write_text("[]", encoding="utf-8")
            (pubmed / "cache_step2_medqa_scored_PubMed_bm25.json").write_text("[]", encoding="utf-8")

            files = get_input_files(root, Args(), {"input_collection": "source_collection_pubmed"})

            self.assertEqual(files, ["source_collection_pubmed/cache_step2_medqa_scored_PubMed_bm25.json"])


if __name__ == "__main__":
    unittest.main()
