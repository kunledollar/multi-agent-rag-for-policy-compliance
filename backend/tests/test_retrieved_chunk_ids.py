import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.retriever_agent import RetrieverAgent
from app.evaluation.dispatcher import ExecutionDispatcher
from app.evaluation.models import BenchmarkCase, ExecutionMode, ModeExecution
from app.evaluation.scoring import score
from app.evaluation.source_ids import normalize_retrieved_chunk
from app.rag.multi_agent_graph import run_sentinel_graph
from app.agents.contracts import AgentDecision


class _Index:
    ntotal = 1
    d = 3


class RetrievedChunkIdTests(unittest.TestCase):
    def test_metadata_dictionary_key_is_authoritative_vector_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({
                "stable-key": {"id": "stale-value", "source": "policy.txt", "page": 2, "text": "Policy text"}
            }))
            with patch.dict("os.environ", {"FAISS_DIR": directory}), patch(
                "app.agents.retriever_agent.faiss.read_index", return_value=_Index()
            ):
                retriever = RetrieverAgent()
            self.assertEqual(retriever.meta[0]["id"], "stable-key")
            self.assertEqual(retriever.meta[0]["chunk_id"], "stable-key")

    def test_normalizer_promotes_nested_and_top_level_identifiers(self):
        nested = normalize_retrieved_chunk({"metadata": {"id": "nested-id"}})
        top_level = normalize_retrieved_chunk({"chunk_id": "top-id"})
        self.assertEqual((nested["id"], nested["chunk_id"]), ("nested-id", "nested-id"))
        self.assertEqual((top_level["id"], top_level["chunk_id"]), ("top-id", "top-id"))

    def test_document_identifier_remains_separate_from_chunk_identifier(self):
        chunk = normalize_retrieved_chunk({
            "metadata": {"document_id": "policy/path.txt", "source": "policy/path.txt"}
        })
        self.assertEqual(chunk["source"], "policy/path.txt")
        self.assertNotIn("id", chunk)
        self.assertNotIn("chunk_id", chunk)

    def test_full_sentinel_mode_retains_five_resolvable_contexts(self):
        metadata_path = Path(__file__).resolve().parents[1] / "data/processed/faiss_index/metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = list(metadata.items())[:5]
        raw_chunks = [
            {"id": key, "chunk_id": key, "source": value["source"], "page": value.get("page"),
             "score": 1 - rank / 10, "text": value["text"]}
            for rank, (key, value) in enumerate(records)
        ]
        dispatcher = ExecutionDispatcher(full=lambda **_: {
            "answer": "fixture answer", "citations": [], "retrieved_chunks": raw_chunks,
        })
        output = dispatcher.execute("fixture question", ExecutionMode.FULL_SENTINEL)

        self.assertEqual(len(output.retrieved_chunks), 5)
        for chunk in output.retrieved_chunks:
            self.assertTrue(chunk["id"])
            self.assertEqual(chunk["id"], chunk["chunk_id"])
            self.assertIn(chunk["id"], metadata)
            self.assertEqual(chunk["text"], metadata[chunk["id"]]["text"])

    def test_graph_response_preserves_id_fields(self):
        chunks = [{"id": "stable-id", "chunk_id": "stable-id", "source": "policy.txt",
                   "page": 1, "score": .75, "text": "Policy text"}]
        class Agent:
            def __init__(self, name, decision, update): self.name=name; self.decision=decision; self.update=update
            def run(self, state):
                self.update(state)
                return AgentDecision(self.name,self.decision,.9,"fixture")
        agents={
            "retrieval":Agent("retrieval","EVIDENCE_SUFFICIENT",lambda s:setattr(s,"retrieved_chunks",chunks)),
            "verification":Agent("verification","APPROVE_EVIDENCE",lambda s:setattr(s,"verification_result",{"decision":"APPROVE_EVIDENCE"})),
            "policy":Agent("policy","APPROVE",lambda s:setattr(s,"policy_result",{"decision":"APPROVE","evidence_ids":["stable-id"]})),
            "reasoning":Agent("reasoning","SYNTHESIZE",lambda s:setattr(s,"reasoning_result",{"decision":"SYNTHESIZE"})),
            "risk":Agent("risk","PROCEED",lambda s:setattr(s,"risk_result",{"decision":"PROCEED"})),
            "answer":Agent("answer","DRAFT",lambda s:setattr(s,"draft_answer",{"answer":"answer","citations":[{"chunk_id":"stable-id"}]})),
            "critic":Agent("critic","APPROVE",lambda s:setattr(s,"critic_feedback",{"decision":"APPROVE"})),
            "final_decision":Agent("final_decision","ANSWER",lambda s:setattr(s,"final_action","ANSWER")),
        }
        result = run_sentinel_graph("question", top_k=1, agents=agents)
        self.assertEqual(result["retrieved_chunks"], chunks)

    def test_rag_only_shape_and_ids_remain_compatible(self):
        class Retriever:
            def retrieve(self, question, top_k):
                return [{"id": "existing-id", "source": "policy.txt", "text": "text", "score": .5}]

        def answer_run(_, state):
            state.draft_answer={"answer":"answer","citations":[]}
            return AgentDecision("answer","DRAFT",.8,"fixture")
        with patch("app.evaluation.dispatcher.AnswerAgent.run", answer_run):
            output = ExecutionDispatcher(retriever_factory=Retriever).execute("q", ExecutionMode.RAG_ONLY)
        self.assertEqual(output.retrieved_chunks[0]["id"], "existing-id")
        self.assertEqual(output.retrieved_chunks[0]["chunk_id"], "existing-id")

    def test_scoring_never_serializes_none_as_a_chunk_id(self):
        case = BenchmarkCase(question_id="q", question="question")
        output = ModeExecution(answer="answer", retrieved_chunks=[{"id": None, "source": "policy.txt"}])
        result = score("run", case, ExecutionMode.FULL_SENTINEL, output, 1)
        self.assertEqual(result.retrieved_chunk_ids, [])
        self.assertNotIn("None", result.retrieved_chunk_ids)


if __name__ == "__main__":
    unittest.main()
