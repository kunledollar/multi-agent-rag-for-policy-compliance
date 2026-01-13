import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from app.telemetry.metrics import (
    agent_execution_total,
    agent_execution_duration_seconds,
)

load_dotenv()


class RetrieverAgent:
    """
    Sentinel Retriever Agent
    Performs semantic search over persisted FAISS index.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        # -------------------------
        # LOCKED container root
        # -------------------------
        self.root = Path("/app")

        faiss_dir = self.root / "artifacts" / "faiss_index"
        self.index_path = faiss_dir / "index.faiss"
        self.meta_path = faiss_dir / "metadata.json"

        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL")
        if not self.embedding_model:
            raise RuntimeError("EMBEDDING_MODEL not set in .env")

        self.top_k = int(top_k or os.getenv("RETRIEVER_TOP_K", "5"))

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")

        self.client = OpenAI(api_key=api_key)

        self._load_artifacts()

    # -------------------------
    # Load FAISS + metadata
    # -------------------------
    def _load_artifacts(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError("FAISS index missing — run ingestion first")

        if not self.meta_path.exists():
            raise FileNotFoundError("Metadata missing — run ingestion first")

        self.index = faiss.read_index(str(self.index_path))
        self.meta = list(json.loads(self.meta_path.read_text()).values())

        if not self.meta:
            raise RuntimeError("Metadata is empty")

    # -------------------------
    # Embedding
    # -------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query.strip()],
        )
        vec = np.array(res.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec

    # -------------------------
    # Public API (Instrumented)
    # -------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        start_time = time.time()

        try:
            k = int(top_k or self.top_k)

            qvec = self._embed_query(query)
            scores, idxs = self.index.search(qvec, k)

            results = []
            for pos, score in zip(idxs[0], scores[0]):
                if pos < 0 or pos >= len(self.meta):
                    continue

                m = self.meta[pos]
                results.append(
                    {
                        "id": m.get("id"),
                        "source": m.get("source"),
                        "page": m.get("page"),
                        "text": m.get("text"),
                        "score": float(score),
                    }
                )

            return results

        finally:
            duration = time.time() - start_time

            agent_execution_total.labels(
                agent_name="retriever"
            ).inc()

            agent_execution_duration_seconds.labels(
                agent_name="retriever"
            ).observe(duration)


if __name__ == "__main__":
    print("🚀 Sentinel Retriever Agent starting")

    agent = RetrieverAgent()
    query = os.getenv("RETRIEVER_TEST_QUERY", "What is the policy on remote work?")
    hits = agent.retrieve(query)

    print(f"🔎 Query: {query}")
    print(f"✅ Retrieved {len(hits)} chunks")
    for i, h in enumerate(hits, 1):
        print(
            f"{i}. score={h['score']:.4f} "
            f"source={h['source']} page={h['page']} "
            f"text={h['text'][:160]}..."
        )
