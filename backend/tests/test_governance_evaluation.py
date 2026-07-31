import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.aggregation import aggregate, latency_summary
from app.evaluation.benchmark_loader import load_benchmark
from app.evaluation.exporter import SHEETS, sanitize_filename, write_workbook
from app.evaluation.models import BenchmarkCase, DetailedResult, ExecutionMode, GovernanceEvaluationRequest, ModeExecution
from app.evaluation.runner import EvaluationRunner
from app.evaluation.scoring import retrieval_scores, score


def case(**overrides):
    values = {"question_id":"Q1","category":"Risk","question":"Question?","reference_answer":"answer","requires_uncertainty":False,"expected_refusal":False,"expected_enforcement_action":"allow","expected_verified_facts":["fact"],"expected_escalation":False,"relevant_chunk_ids":["c1","c3"],"graded_relevance":{"c1":3,"c3":1},"required_citation_claims":["claim"],"required_trace_elements":["retrieval step"],"required_audit_fields":["run_id"],"expected_agent_handoffs":["a-b"]}
    values.update(overrides); return BenchmarkCase(**values)


def result(mode, latency=10, **values):
    base={"run_id":"run","question_id":"Q1","category":"x","question":"q","execution_mode":mode,"latency_ms":latency}
    base.update(values); return DetailedResult(**base)


class ScoringTests(unittest.TestCase):
    def test_all_case_metrics_and_retrieval_formulas(self):
        output=ModeExecution(answer="answer",policy_decision="allow",enforcement_action="allow",uncertainty_observed=False,refusal_observed=False,verified_facts=["fact"],escalation_observed=False,citations=[{"source":"c1","claim":"claim"}],retrieved_chunks=[{"id":"c1","source":"c1"},{"id":"x"},{"id":"c3"}],trace_elements=["retrieval step"],audit={"run_id":"run"},handoffs_attempted=2,handoffs_successful=1)
        row=score("run",case(),ExecutionMode.FULL_SENTINEL,output,12)
        self.assertTrue(row.uncertainty_correct); self.assertEqual(row.citations_complete,1)
        self.assertTrue(row.refusal_correct); self.assertTrue(row.enforcement_correct)
        self.assertEqual(row.verification_correct,1); self.assertEqual(row.trace_completeness,1)
        self.assertEqual(row.audit_completeness,1); self.assertTrue(row.escalation_correct)
        self.assertEqual(row.handoff_success,.5); self.assertEqual(row.precision_at_5,2/5)
        self.assertEqual(row.recall_at_5,1); self.assertEqual(row.reciprocal_rank,1)
        import math
        self.assertAlmostEqual(row.ndcg_at_5, (7 + 1/2)/(7 + 1/math.log2(3)), places=6)

    def test_no_relevant_items_is_not_applicable(self):
        self.assertEqual(retrieval_scores([], case(relevant_chunk_ids=None, relevant_document_ids=None)), (None,)*5)

    def test_llm_only_has_no_retrieval_or_citations(self):
        row=score("r",case(),ExecutionMode.LLM_ONLY,ModeExecution(answer="a",citations=[{"source":"fake"}],retrieved_chunks=None),1)
        self.assertIsNone(row.citations_returned); self.assertIsNone(row.precision_at_5); self.assertIsNone(row.handoff_success)

    def test_null_required_fields_are_not_zero(self):
        row=score("r",case(required_trace_elements=None,required_audit_fields=None,expected_verified_facts=None),ExecutionMode.RAG_ONLY,ModeExecution(),1)
        self.assertIsNone(row.trace_completeness); self.assertIsNone(row.audit_completeness); self.assertIsNone(row.verification_correct)


class AggregationTests(unittest.TestCase):
    def test_tie_and_latency_lowest_winner(self):
        rows=[result(ExecutionMode.FULL_SENTINEL,20,uncertainty_correct=True),result(ExecutionMode.RAG_ONLY,10,uncertainty_correct=True)]
        summary=aggregate(rows)
        self.assertTrue(summary[0].best_performing_mode.startswith("Tie"))
        latency=next(x for x in summary if x.primary_metric=="Governance Latency")
        self.assertEqual(latency.best_performing_mode,"rag_only")

    def test_latency_percentiles(self):
        stats=latency_summary([result(ExecutionMode.FULL_SENTINEL,x) for x in [1,2,3,4,5]])[0]
        self.assertEqual(stats["median"],3); self.assertAlmostEqual(stats["p90"],4.6)


class LoaderAndExportTests(unittest.TestCase):
    def test_json_mapping_order_and_duplicate_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"benchmark.json"; path.write_text(json.dumps([{"query_id":"Q2","query":"b","dataset_version":"2"},{"query_id":"Q1","query":"a","dataset_version":"2"}]))
            dataset=load_benchmark(path); self.assertEqual([x.question_id for x in dataset.cases],["Q1","Q2"]); self.assertEqual(dataset.version,"2")
            path.write_text(json.dumps([{"query_id":"Q1","query":"a"},{"query_id":"Q1","query":"b"}]))
            with self.assertRaisesRegex(ValueError,"Duplicate"): load_benchmark(path)

    def test_missing_columns_empty_and_missing_file_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"b.json"
            path.write_text("[]"); self.assertRaisesRegex(ValueError,"no benchmark",load_benchmark,path)
            path.write_text('[{"query_id":"Q"}]'); self.assertRaisesRegex(ValueError,"Missing required",load_benchmark,path)
            self.assertRaisesRegex(ValueError,"cannot be loaded",load_benchmark,Path(directory)/"missing.json")

    def test_xlsx_round_trip_and_required_sheets(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"input.xlsx"
            write_workbook(path,{name:([{"query_id":"Q1","query":"q","dataset_version":"1"}] if name=="Questions" else []) for name in SHEETS})
            self.assertEqual(load_benchmark(path).cases[0].question_id,"Q1")
            with zipfile.ZipFile(path) as book:
                workbook=book.read("xl/workbook.xml").decode(); self.assertTrue(all(name in workbook for name in SHEETS))

    def test_filename_sanitization(self):
        self.assertEqual(sanitize_filename("../../bad name"),"bad_name")


class FakeDispatcher:
    def __init__(self): self.calls=[]
    def execute(self, question, mode, **kwargs):
        self.calls.append((question,mode))
        if question=="bad" and mode==ExecutionMode.RAG_ONLY: raise RuntimeError("mode failed")
        return ModeExecution(answer="answer",retrieved_chunks=None if mode==ExecutionMode.LLM_ONLY else [])


class RunnerTests(unittest.TestCase):
    def test_defaults_are_all_modes_and_invalid_rejected(self):
        req=GovernanceEvaluationRequest(benchmark_cases=[case()])
        self.assertEqual(req.selected_modes,list(ExecutionMode))
        with self.assertRaises(Exception): GovernanceEvaluationRequest(benchmark_cases=[case()],selected_modes=["invalid"])

    def test_every_question_mode_runs_and_failure_isolated(self):
        dispatcher=FakeDispatcher()
        with tempfile.TemporaryDirectory() as directory:
            req=GovernanceEvaluationRequest(benchmark_cases=[case(question_id="Q1",question="bad"),case(question_id="Q2",question="good")],output_formats=["json"])
            payload=EvaluationRunner(dispatcher, directory).run(req)
        self.assertEqual(len(dispatcher.calls),6); self.assertEqual(len(payload["detailed_results"]),6)
        self.assertEqual(len(payload["errors"]),1); self.assertEqual(payload["status"],"completed_with_errors")


if __name__ == "__main__": unittest.main()
