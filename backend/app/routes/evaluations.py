import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.evaluation.models import GovernanceEvaluationRequest
from app.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/evaluations", tags=["governance evaluation"])


@router.post("/governance")
def governance_evaluation(request: GovernanceEvaluationRequest):
    try: return EvaluationRunner().run(request)
    except (ValueError, ValidationError) as error: raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/governance/upload")
async def upload_governance_evaluation(
    benchmark: UploadFile = File(...),
    selected_modes: str = Form('["full_sentinel","rag_only","llm_only"]'),
    run_name: str = Form("governance_evaluation"),
):
    suffix = Path(benchmark.filename or "").suffix.lower()
    if suffix not in {".xlsx", ".json"}: raise HTTPException(status_code=422, detail="Benchmark must be .xlsx or .json")
    try:
        with tempfile.TemporaryDirectory(prefix="sentinel-benchmark-") as directory:
            path = Path(directory) / (Path(benchmark.filename or f"benchmark{suffix}").name)
            path.write_bytes(await benchmark.read())
            request = GovernanceEvaluationRequest(benchmark_source=str(path), selected_modes=json.loads(selected_modes), run_name=run_name)
            return EvaluationRunner().run(request)
    except (ValueError, ValidationError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
