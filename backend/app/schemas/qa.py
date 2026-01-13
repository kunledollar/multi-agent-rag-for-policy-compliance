from pydantic import BaseModel, Field
from typing import List, Optional

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)

class EvidenceChunk(BaseModel):
    source: str
    chunk_id: str
    text: str

class AskResponse(BaseModel):
    answer: str
    compliance: str
    reasoning: str
    evidence: List[EvidenceChunk] = []
    trace_id: Optional[str] = None
