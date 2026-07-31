import os
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATA_DIR", "/tmp/sentinel-test-data")

from app.agents.answer_generation_agent import AnswerGenerationAgent
from app.agents.ingestion_agent import IngestionAgent


class IngestionAndCitationTests(unittest.TestCase):
    def test_nested_raw_documents_are_read_and_chunked(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "data" / "raw"
            policy = raw / "hr" / "leave" / "policy.txt"
            policy.parent.mkdir(parents=True)
            policy.write_text("Leave policy. " * 100, encoding="utf-8")

            chunks = IngestionAgent(data_dir=raw).load_documents()

            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(c["source"] == "hr/leave/policy.txt" for c in chunks))
            self.assertTrue(all(c["text"] for c in chunks))

    def test_generated_citations_must_match_retrieved_files(self):
        chunks = [{
            "source": "hr/leave/policy.txt",
            "page": None,
            "text": "Employees receive approved leave.",
        }]
        citations = [
            {"source": "invented.txt", "page": None},
            {"source": "hr/leave/policy.txt", "page": None, "quote_hint": "invented"},
        ]

        grounded = AnswerGenerationAgent._ground_citations(citations, chunks)

        self.assertEqual(len(grounded), 1)
        self.assertEqual(grounded[0]["source"], "hr/leave/policy.txt")
        self.assertEqual(grounded[0]["quote_hint"], chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
