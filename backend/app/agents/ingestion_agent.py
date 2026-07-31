import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict

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
DATA_ROOT = Path(os.getenv("DATA_DIR", "/app/data"))

DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"

FAISS_DIR = Path(
    os.getenv("FAISS_DIR", str(PROCESSED_DIR / "faiss_index"))
)
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
logger = logging.getLogger("sentinel.agents.ingestion")


def safe_iter_files(root: Path, supported_extensions=SUPPORTED_EXT):
    """Yield supported files without allowing one bad directory to abort a scan."""
    root = Path(root)

    def warn(error: OSError) -> None:
        skipped_path = getattr(error, "filename", None) or root
        logger.warning(
            "Skipping unreadable path %s (%s: %s)",
            skipped_path,
            type(error).__name__,
            error,
        )

    try:
        walker = os.walk(root, topdown=True, onerror=warn, followlinks=False)
        for directory, dirnames, filenames in walker:
            # Mutating dirnames controls traversal order and is portable across
            # Linux containers and Windows bind mounts.
            dirnames.sort()
            filenames.sort()
            for filename in filenames:
                path = Path(directory) / filename
                if path.suffix.lower() not in supported_extensions:
                    continue
                try:
                    if path.is_file():
                        yield path
                except OSError as error:
                    warn(error)
    except OSError as error:
        # Some platform implementations can raise before invoking ``onerror``.
        warn(error)


class IngestionAgent:
    """
    Sentinel Ingestion Agent

    - Loads raw documents
    - Chunks using LangChain RecursiveCharacterTextSplitter
    - Embeds chunks
    - Persists FAISS index + metadata
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
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

        # Raw documents are grouped into source/domain subdirectories.  Walk the
        # whole tree so adding that organization does not make documents
        # invisible to ingestion. Sorting keeps chunk/embedding order stable.
        files = list(safe_iter_files(self.data_dir))

        for file in files:
            # Preserve the path below data/raw. Basenames are not unique across
            # nested sources, and the relative path is needed for traceability.
            source = file.relative_to(self.data_dir).as_posix()

            try:
                if file.suffix.lower() == ".pdf":
                    for page, text in enumerate(self._read_pdf(file), start=1):
                        for i, chunk in enumerate(self._chunk(text)):
                            records.append({
                                "id": self._hash(f"{file}-{page}-{i}-{chunk}"),
                                "source": source,
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
                            "source": source,
                            "page": None,
                            "text": chunk,
                        })
            except Exception as error:
                logger.warning(
                    "Skipping unreadable policy file %s (%s: %s)",
                    file,
                    type(error).__name__,
                    error,
                )

        return records

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        import faiss
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to build embeddings")
        if not EMBEDDING_MODEL:
            raise RuntimeError("EMBEDDING_MODEL is required to build embeddings")
        client = OpenAI(api_key=api_key)
        res = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        vectors = np.array([r.embedding for r in res.data], dtype="float32")
        faiss.normalize_L2(vectors)
        return vectors

    def run(self):
        import faiss
        print("🚀 Sentinel Ingestion Agent starting")

        docs = self.load_documents()
        if not docs:
            print("⚠️ No documents found in data/raw/")
            return

        existing: Dict[str, Dict] = {}
        if META_PATH.exists():
            existing = json.loads(META_PATH.read_text(encoding="utf-8"))

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
        META_PATH.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"✅ Indexed {len(new_docs)} new chunks")
        print("🎉 Ingestion complete")


if __name__ == "__main__":
    IngestionAgent().run()
