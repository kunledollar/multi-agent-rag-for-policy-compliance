import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Paths (ALIGNED WITH RETRIEVER)
# -------------------------------------------------
ROOT = Path("/app")

DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

FAISS_DIR = PROCESSED_DIR / "faiss_index"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

FAISS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "metadata.json"

# -------------------------------------------------
# Config
# -------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

SUPPORTED_EXT = {".pdf", ".docx", ".txt"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class IngestionAgent:
    """
    Sentinel Ingestion Agent

    - Loads raw documents
    - Chunks using LangChain RecursiveCharacterTextSplitter
    - Embeds chunks
    - Persists FAISS index + metadata
    """

    def __init__(self):
        if not EMBEDDING_MODEL:
            raise RuntimeError("EMBEDDING_MODEL not set in .env")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",   # section / clause boundaries
                "\n",
                ". ",
                "; ",
                ", ",
                " "
            ],
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

    @staticmethod
    def _read_pdf(path: Path) -> List[str]:
        reader = PdfReader(str(path))
        return [(p.extract_text() or "").strip() for p in reader.pages]

    @staticmethod
    def _read_docx(path: Path) -> str:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    @staticmethod
    def _read_txt(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def load_documents(self) -> List[Dict]:
        records: List[Dict] = []

        for file in DATA_DIR.iterdir():
            if file.suffix.lower() not in SUPPORTED_EXT:
                continue

            if file.suffix.lower() == ".pdf":
                for page, text in enumerate(self._read_pdf(file), start=1):
                    for i, chunk in enumerate(self._chunk(text)):
                        records.append({
                            "id": self._hash(f"{file}-{page}-{i}-{chunk}"),
                            "source": file.name,
                            "page": page,
                            "text": chunk,
                        })
            else:
                text = (
                    self._read_docx(file)
                    if file.suffix.lower() == ".docx"
                    else self._read_txt(file)
                )
                for i, chunk in enumerate(self._chunk(text)):
                    records.append({
                        "id": self._hash(f"{file}-{i}-{chunk}"),
                        "source": file.name,
                        "page": None,
                        "text": chunk,
                    })

        return records

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        res = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        vectors = np.array([r.embedding for r in res.data], dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors

    def run(self):
        print("🚀 Sentinel Ingestion Agent starting")

        docs = self.load_documents()
        if not docs:
            print("⚠️ No documents found in data/raw/")
            return

        existing: Dict[str, Dict] = {}
        if META_PATH.exists():
            existing = json.loads(META_PATH.read_text())

        new_docs = [d for d in docs if d["id"] not in existing]
        if not new_docs:
            print("✅ No new documents to index")
            return

        print(f"🧩 New chunks to embed: {len(new_docs)}")

        vectors = self.embed_texts([d["text"] for d in new_docs])
        dim = vectors.shape[1]

        index = (
            faiss.read_index(str(INDEX_PATH))
            if INDEX_PATH.exists()
            else faiss.IndexFlatIP(dim)
        )

        index.add(vectors)

        for d in new_docs:
            existing[d["id"]] = d

        faiss.write_index(index, str(INDEX_PATH))
        META_PATH.write_text(json.dumps(existing, indent=2))

        print(f"✅ Indexed {len(new_docs)} new chunks")
        print("🎉 Ingestion complete")


if __name__ == "__main__":
    IngestionAgent().run()

