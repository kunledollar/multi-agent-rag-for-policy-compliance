import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DATA_DIR", "/tmp/sentinel-test-data")

from app.agents.answer_generation_agent import AnswerGenerationAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.retriever_agent import CorpusUnavailableError, RetrieverAgent


class IngestionAndCitationTests(unittest.TestCase):
    def setUp(self):
        RetrieverAgent._raw_chunk_cache.clear()

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

    def test_unreadable_nested_directory_is_logged_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            error = OSError(5, "Input/output error", str(raw / "broken"))

            def broken_walk(*args, **kwargs):
                kwargs["onerror"](error)
                yield str(raw), [], []

            with patch("app.agents.ingestion_agent.os.walk", broken_walk):
                with self.assertLogs("sentinel.agents.ingestion", "WARNING") as logs:
                    chunks = IngestionAgent(data_dir=raw).load_documents()

            self.assertEqual(chunks, [])
            warning = " ".join(logs.output)
            self.assertIn(str(raw / "broken"), warning)
            self.assertIn("OSError", warning)
            self.assertIn("Input/output error", warning)

    def test_readable_files_survive_unreadable_sibling_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            readable = raw / "readable"
            readable.mkdir(parents=True)
            (readable / "policy.txt").write_text("A readable policy", encoding="utf-8")
            error = OSError(5, "Input/output error", str(raw / "broken"))
            real_walk = os.walk

            def partially_broken_walk(*args, **kwargs):
                kwargs["onerror"](error)
                yield from real_walk(*args, **kwargs)

            with patch("app.agents.ingestion_agent.os.walk", partially_broken_walk):
                chunks = IngestionAgent(data_dir=raw).load_documents()

            self.assertEqual(
                [chunk["source"] for chunk in chunks], ["readable/policy.txt"]
            )

    def test_document_discovery_order_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            for relative in ("z/second.txt", "a/third.txt", "a/first.txt"):
                path = raw / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")

            sources = [
                chunk["source"] for chunk in IngestionAgent(data_dir=raw).load_documents()
            ]

            self.assertEqual(sources, ["a/first.txt", "a/third.txt", "z/second.txt"])

    def test_valid_artifacts_are_used_without_scanning_broken_raw_tree(self):
        import faiss
        import json
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            artifact_dir = data / "processed" / "faiss_index"
            artifact_dir.mkdir(parents=True)
            index = faiss.IndexFlatIP(2)
            index.add(np.array([[1.0, 0.0]], dtype="float32"))
            faiss.write_index(index, str(artifact_dir / "index.faiss"))
            (artifact_dir / "metadata.json").write_text(
                json.dumps([{
                    "id": "one",
                    "source": "saved.txt",
                    "page": None,
                    "text": "Saved policy text",
                }]),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"DATA_DIR": str(data)}, clear=False):
                with patch.object(
                    IngestionAgent, "load_documents", side_effect=AssertionError("raw scan")
                ):
                    retriever = RetrieverAgent()

            self.assertEqual(retriever.index.ntotal, 1)
            self.assertEqual(retriever.meta[0]["source"], "saved.txt")

    def test_empty_corpus_without_valid_artifacts_has_controlled_error(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "raw").mkdir()
            with patch.dict(os.environ, {"DATA_DIR": str(data)}, clear=False):
                with self.assertRaisesRegex(CorpusUnavailableError, "No readable policy"):
                    RetrieverAgent()

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
