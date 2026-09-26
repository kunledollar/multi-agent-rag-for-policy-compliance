import argparse, importlib.util, json, tempfile, unittest
from pathlib import Path
from unittest.mock import Mock

from app.evaluation.ablation import AblationDispatcher, AblationId, CONFIGURATIONS
from app.evaluation.metrics import summarize
from app.evaluation.models import ExecutionMode, ModeExecution
from app.evaluation.statistics import align_pairs, continuous, holm_adjust

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("ablation_runner",ROOT/"data/processed/run_ablation_study.py")
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class AblationTests(unittest.TestCase):
    def test_r6_configuration_matrix(self):
        baseline=CONFIGURATIONS[AblationId.A0]
        fields=("enable_critic","enable_verification","enable_policy","enable_revision_loop")
        expected=((AblationId.A1,"enable_critic"),(AblationId.A2,"enable_verification"),
                  (AblationId.A3,"enable_policy"),(AblationId.A4,"enable_revision_loop"))
        for identifier,disabled in expected:
            changed=[field for field in fields if getattr(CONFIGURATIONS[identifier],field)!=getattr(baseline,field)]
            self.assertEqual(changed,[disabled])

    def test_a0_and_external_baselines_delegate(self):
        production=Mock(); production.execute.side_effect=lambda q,mode,**kw: ModeExecution(answer=mode.value)
        dispatcher=AblationDispatcher(production=production)
        self.assertEqual(dispatcher.execute("q",CONFIGURATIONS[AblationId.A0]).answer,"full_sentinel")
        self.assertEqual(dispatcher.execute("q",CONFIGURATIONS[AblationId.A6]).answer,"rag_only")
        self.assertEqual(dispatcher.execute("q",CONFIGURATIONS[AblationId.A7]).answer,"llm_only")

    def test_baseline_architectures_are_explicit(self):
        self.assertEqual(CONFIGURATIONS[AblationId.A5].architecture,"r5")
        self.assertEqual(CONFIGURATIONS[AblationId.A6].architecture,"single_stage")
        self.assertEqual(CONFIGURATIONS[AblationId.A7].architecture,"llm_only")

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
        fake=Mock(); fake.execute.side_effect=lambda question,config,**kw: ModeExecution(
            answer=f"{config.configuration_id}:{question}",
            retrieved_chunks=None if config.configuration_id==AblationId.A7 else [])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"benchmark.json"; path.write_text(json.dumps(benchmark)); out=Path(directory)/"out"
            args=argparse.Namespace(benchmark=str(path),full=False,limit=2,configuration="all",output_dir=str(out),resume=False)
            merged=runner.run(args,fake); self.assertEqual(len(merged["rows"]),16)
            self.assertEqual(len({(r["question_id"],r["configuration_id"]) for r in merged["rows"]}),16)
            calls=fake.execute.call_count; args.resume=True; resumed=runner.run(args,fake)
            self.assertEqual(fake.execute.call_count,calls); self.assertEqual(len(resumed["rows"]),16)

if __name__ == "__main__": unittest.main()
