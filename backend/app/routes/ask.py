from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import traceback
import logging

from app.rag.graph import run_sentinel_graph

router = APIRouter()
logger = logging.getLogger("sentinel.api.ask")


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
def ask(req: AskRequest):
    try:
        result = run_sentinel_graph(question=req.question)

        # -------------------------------
        # User-safe response (unchanged)
        # -------------------------------
        response = {
            "answer": result.get("answer"),
            "action_items": result.get("action_items", []),
            "citations": result.get("citations", []),
            "confidence": result.get("confidence"),
            "trace_id": result.get("trace_id"),
        }

        # -------------------------------------------------
        # Admin / Auditor introspection (non-breaking)
        # -------------------------------------------------
        if "retrieved_chunks" in result:
            response["chunks"] = [
                {
                    "text": c.get("text"),
                    "source": c.get("source"),
                    "page": c.get("page"),
                    "score": round(c.get("score", 0.0), 3),
                }
                for c in result["retrieved_chunks"]
            ]

        if "agent_trace" in result:
            response["agent_trace"] = result["agent_trace"]

        # -------------------------------------------------
        # RAGAS metrics (ADD ONLY – NO SIDE EFFECTS)
        # -------------------------------------------------
        if "ragas_metrics" in result:
            response["ragas_metrics"] = result["ragas_metrics"]

        return response

    except Exception:
        logger.error("Sentinel /v1/ask failed")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail="Sentinel failed to process the request."
        )
