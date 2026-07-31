import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np
from docx import Document
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
logger = logging.getLogger("sentinel.agents.ingestion")

DATA_ROOT = Path(os.getenv("DATA_DIR", "/app/data"))
DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
FAISS_DIR = Path(os.getenv("FAISS_DIR", str(PROCESSED_DIR / "faiss_index")))
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"

FAISS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "metadata.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def safe_iter_files(
    root: Path,
    supported_extensions: Sequence[str] = tuple(SUPPORTED_EXTENSIONS),
) -> Iterator[Path]:
    """Yield supported files while skipping unreadable paths safely."""
    root = Path(root)
    extensions = {extension.lower() for extension in supported_extensions}

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
            dirnames.sort()
            filenames.sort()

            for filename in filenames:
                path = Path(directory) / filename
                if path.suffix.lower() not in extensions:
                    continue

                try:
                    if path.is_file():
                        yield path
                except OSError as error:
                    warn(error)
    except OSError as error:
        warn(error)


class IngestionAgent:
    """Load, chunk, embed, and index Sentinel policy documents."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)

        if CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero")
        if CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative")
        if CHUNK_OVERLAP >= CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if EMBED_BATCH_SIZE <= 0:
            raise ValueError("EMBED_BATCH_SIZE must be greater than zero")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        )

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return [
            chunk.strip()
            for chunk in self.splitter.split_text(text)
            if chunk and chunk.strip()
        ]

    @staticmethod
    def _read_pdf(path: Path) -> List[str]:
        reader = PdfReader(str(path))
        return [(page.extract_text() or "").strip() for page in reader.pages]

    @staticmethod
    def _read_docx(path: Path) -> str:
        document = Document(str(path))
        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

    @staticmethod
    def _read_txt(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")

    def load_documents(self) -> List[Dict]:
        records: List[Dict] = []

        for file in safe_iter_files(self.data_dir):
            try:
                source = file.relative_to(self.data_dir).as_posix()
            except ValueError:
                source = file.as_posix()

            try:
                suffix = file.suffix.lower()

                if suffix == ".pdf":
                    for page_number, page_text in enumerate(
                        self._read_pdf(file), start=1
                    ):
                        for chunk_number, chunk in enumerate(self._chunk(page_text)):
                            records.append(
                                {
                                    "id": self._hash(
                                        f"{source}|{page_number}|{chunk_number}|{chunk}"
                                    ),
                                    "source": source,
                                    "page": page_number,
                                    "text": chunk,
                                }
                            )
                else:
                    text = (
                        self._read_docx(file)
                        if suffix == ".docx"
                        else self._read_txt(file)
                    )
                    for chunk_number, chunk in enumerate(self._chunk(text)):
                        records.append(
                            {
                                "id": self._hash(
                                    f"{source}|{chunk_number}|{chunk}"
                                ),
                                "source": source,
                                "page": None,
                                "text": chunk,
                            }
                        )
            except Exception as error:
                logger.warning(
                    "Skipping unreadable policy file %s (%s: %s)",
                    file,
                    type(error).__name__,
                    error,
                )

        return records

    @staticmethod
    def _create_openai_client() -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to build embeddings")
        if not EMBEDDING_MODEL:
            raise RuntimeError("EMBEDDING_MODEL is required to build embeddings")
        return OpenAI(api_key=api_key)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts in bounded batches while preserving input order."""
        import faiss

        if not texts:
            return np.empty((0, 0), dtype="float32")

        client = self._create_openai_client()
        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

        for batch_number, start in enumerate(
            range(0, len(texts), EMBED_BATCH_SIZE), start=1
        ):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            print(
                f"Embedding batch {batch_number}/{total_batches} "
                f"({len(batch)} chunks)"
            )

            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            ordered_results = sorted(response.data, key=lambda item: item.index)

            if len(ordered_results) != len(batch):
                raise RuntimeError(
                    "Embedding response count mismatch for batch "
                    f"{batch_number}: expected {len(batch)}, "
                    f"received {len(ordered_results)}"
                )

            all_embeddings.extend(item.embedding for item in ordered_results)

        if len(all_embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(texts)}, "
                f"received {len(all_embeddings)}"
            )

        vectors = np.asarray(all_embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise RuntimeError(f"Unexpected embedding matrix shape: {vectors.shape}")

        faiss.normalize_L2(vectors)
        return vectors

    @staticmethod
    def _load_existing_metadata() -> Dict[str, Dict]:
        if not META_PATH.exists():
            return {}

        try:
            loaded = json.loads(META_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Unable to read metadata file {META_PATH}: {error}"
            ) from error

        if not isinstance(loaded, dict):
            raise RuntimeError(
                f"Metadata file must contain a JSON object: {META_PATH}"
            )
        return loaded

    @staticmethod
    def _write_metadata_atomic(metadata: Dict[str, Dict]) -> None:
        temporary_path = META_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, META_PATH)

    @staticmethod
    def _write_faiss_atomic(index) -> None:
        import faiss

        temporary_path = INDEX_PATH.with_suffix(".faiss.tmp")
        faiss.write_index(index, str(temporary_path))
        os.replace(temporary_path, INDEX_PATH)

    def run(self) -> None:
        import faiss

        print("Sentinel Ingestion Agent starting")

        documents = self.load_documents()
        if not documents:
            print("No documents found in data/raw/")
            return

        existing = self._load_existing_metadata()
        new_documents = [
            document for document in documents if document["id"] not in existing
        ]

        if not new_documents:
            print("No new documents to index")
            return

        print(f"New chunks to embed: {len(new_documents)}")
        vectors = self.embed_texts(
            [document["text"] for document in new_documents]
        )

        if vectors.size == 0:
            print("No non-empty chunks were available for embedding")
            return

        dimension = int(vectors.shape[1])

        if INDEX_PATH.exists():
            index = faiss.read_index(str(INDEX_PATH))
            if index.d != dimension:
                raise RuntimeError(
                    "Embedding dimension does not match the existing FAISS "
                    f"index: index={index.d}, embeddings={dimension}"
                )
            if index.ntotal != len(existing):
                raise RuntimeError(
                    "FAISS index and metadata are out of sync before append: "
                    f"index vectors={index.ntotal}, "
                    f"metadata records={len(existing)}"
                )
        else:
            if existing:
                raise RuntimeError(
                    "Metadata exists but the FAISS index is missing. "
                    "Restore both files together or rebuild the index."
                )
            index = faiss.IndexFlatIP(dimension)

        index.add(vectors)

        for document in new_documents:
            existing[document["id"]] = document

        if index.ntotal != len(existing):
            raise RuntimeError(
                "FAISS index and metadata are out of sync after append: "
                f"index vectors={index.ntotal}, "
                f"metadata records={len(existing)}"
            )

        self._write_faiss_atomic(index)
        self._write_metadata_atomic(existing)

        print(f"Indexed {len(new_documents)} new chunks")
        print("Ingestion complete")


if __name__ == "__main__":
    IngestionAgent().run()
