import argparse, importlib.util, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app.evaluation.ablation import AblationDispatcher, AblationId, CONFIGURATIONS
from app.evaluation.metrics import summarize
from app.evaluation.models import ExecutionMode, ModeExecution
from app.evaluation.statistics import align_pairs, continuous, holm_adjust

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("ablation_runner",ROOT/"data/processed/run_ablation_study.py")
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class AblationTests(unittest.TestCase):
    def test_configuration_matrix_disables_only_named_full_stage(self):
        baseline=CONFIGURATIONS[AblationId.A0]
        fields=("enable_reranker","enable_verification","enable_compliance_enforcement","enable_answer_generation")
        for identifier, disabled in ((AblationId.A1,"enable_reranker"),(AblationId.A2,"enable_verification"),(AblationId.A3,"enable_compliance_enforcement"),(AblationId.A4,"enable_answer_generation")):
            config=CONFIGURATIONS[identifier]
            self.assertEqual([field for field in fields if getattr(config,field)!=getattr(baseline,field)],[disabled])

    def test_a0_delegates_to_existing_full_sentinel_behavior(self):
        expected=ModeExecution(answer="same",retrieved_chunks=[]); production=Mock(); production.execute.return_value=expected
        actual=AblationDispatcher(production=production).execute("q",CONFIGURATIONS[AblationId.A0])
        self.assertEqual(actual.answer,"same"); production.execute.assert_called_once()
        self.assertEqual(production.execute.call_args.args[1],ExecutionMode.FULL_SENTINEL)

    def test_existing_baselines_are_delegated_unchanged(self):
        production=Mock(); production.execute.side_effect=lambda q,mode,**kw: ModeExecution(answer=mode.value)
        dispatcher=AblationDispatcher(production=production)
        self.assertEqual(dispatcher.execute("q",CONFIGURATIONS[AblationId.A5]).answer,"rag_only")
        self.assertEqual(dispatcher.execute("q",CONFIGURATIONS[AblationId.A6]).answer,"llm_only")

    def test_each_disabled_agent_is_never_invoked(self):
        chunks=[{"id":"c","source":"s","text":"evidence","score":1}]
        retriever=Mock(); retriever.retrieve.return_value=chunks
        compliance=Mock(); compliance.run.return_value=SimpleNamespace(verdict="allow",status="ok",confidence=1)
        reasoning=Mock(); reasoning.run.return_value={"summary_reasoning":"upstream"}
        generation=Mock(); generation.run.return_value={"answer":"answer","citations":[]}
        verification=Mock(); reranker=Mock(side_effect=lambda x:x)
        dispatcher=AblationDispatcher(retriever_factory=lambda:retriever,compliance_factory=lambda:compliance,
            reasoning_factory=lambda:reasoning,generation_factory=lambda:generation,verification_factory=lambda:verification,reranker=reranker)
        expectations={AblationId.A1:(reranker,False),AblationId.A2:(verification.run,False),AblationId.A3:(compliance.run,False),AblationId.A4:(generation.run,False)}
        for identifier,(callable_mock,_) in expectations.items():
            reranker.reset_mock(); verification.reset_mock(); compliance.reset_mock(); generation.reset_mock()
            output=dispatcher.execute("q",CONFIGURATIONS[identifier])
            callable_mock.assert_not_called(); self.assertIn(CONFIGURATIONS[identifier].disabled_components[0],output.audit["disabled_components"])
            if identifier==AblationId.A4: self.assertEqual(output.answer,"upstream")

    def test_null_metrics_are_not_zero(self):
        result=summarize([{"verification_correct":None,"latency_ms":1}])
        self.assertIsNone(result["verification_correct"]["mean"]); self.assertEqual(result["verification_correct"]["n"],0)

    def test_pairing_aligns_by_question_id_not_row_order(self):
        rows=[{"configuration_id":"A0","question_id":"Q2","m":4},{"configuration_id":"A1","question_id":"Q1","m":1},
              {"configuration_id":"A0","question_id":"Q1","m":3},{"configuration_id":"A1","question_id":"Q2","m":2}]
        ids,left,right=align_pairs(rows,"A0","A1","m")
        self.assertEqual(ids,["Q1","Q2"]); self.assertEqual(left,[3,4]); self.assertEqual(right,[1,2])
        self.assertEqual(continuous(rows,"A0","A1","m")["mean_difference"],2)
        self.assertEqual(holm_adjust([.01,.04,.03]),[.03,.06,.06])

    def test_two_query_all_configuration_smoke_resume_and_uniqueness(self):
        benchmark=[{"query_id":f"Q{i}","query":f"question {i}","category":"test"} for i in (1,2)]
        fake=Mock(); fake.execute.side_effect=lambda question,config,**kw: ModeExecution(answer=f"{config.configuration_id}:{question}",retrieved_chunks=None if config.configuration_id==AblationId.A6 else [],audit={"reranker_invoked":config.enable_reranker,"verification_invoked":config.enable_verification,"compliance_invoked":config.enable_compliance_enforcement,"answer_generation_invoked":config.enable_answer_generation})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"benchmark.json"; path.write_text(json.dumps(benchmark)); out=Path(directory)/"out"
            args=argparse.Namespace(benchmark=str(path),full=False,limit=2,configuration="all",output_dir=str(out),resume=False)
            merged=runner.run(args,fake); self.assertEqual(len(merged["rows"]),14)
            self.assertEqual(len({(r["question_id"],r["configuration_id"]) for r in merged["rows"]}),14)
            calls=fake.execute.call_count; args.resume=True; resumed=runner.run(args,fake)
            self.assertEqual(fake.execute.call_count,calls); self.assertEqual(len(resumed["rows"]),14)

if __name__ == "__main__": unittest.main()
