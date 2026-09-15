"""Safe full-build lifecycle for a local Chroma index."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from corpus import TARGET_COUNT

DEFAULT_BATCH_SIZE = 64
VALIDATION_SAMPLE_SIZE = 5
CHUNK_FIELDS = frozenset(
    {
        "chunk_id",
        "paper_id",
        "title",
        "year",
        "venue",
        "url",
        "page_number",
        "chunk_index",
        "pdf_sha256",
        "chunk_sha256",
        "abstract",
        "text",
    }
)


class IndexBuildError(RuntimeError):
    """Raised when a staging index cannot be built and validated safely."""


@dataclass(frozen=True, slots=True)
class IndexChunk:
    chunk_id: str
    paper_id: str
    title: str
    year: int
    venue: str
    url: str
    page_number: int
    chunk_index: int
    pdf_sha256: str
    chunk_sha256: str
    text: str

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "page": self.page_number,
            "chunk_index": self.chunk_index,
            "pdf_sha256": self.pdf_sha256,
            "chunk_sha256": self.chunk_sha256,
        }


@dataclass(frozen=True, slots=True)
class IndexManifest:
    collection_name: str
    index_identity: str
    embedding_model: str
    embedding_dimension: int
    corpus_manifest_sha256: str
    chunks_jsonl_sha256: str
    paper_count: int
    chunk_count: int
    build_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IndexBuildSummary:
    manifest: IndexManifest
    chroma_path: Path
    index_manifest_path: Path
    active_pointer_path: Path
    duration_seconds: float
    validated: bool


class SentenceTransformerEmbedder:
    """Small adapter that exposes only the behavior needed by the index builder."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_index_identity(
    *, embedding_model: str, corpus_manifest_sha256: str, chunks_jsonl_sha256: str
) -> str:
    """Identify logical index inputs independently from a particular build attempt."""
    payload = json.dumps(
        {
            "embedding_model": embedding_model,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "chunks_jsonl_sha256": chunks_jsonl_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_index_chunks(
    path: Path, *, expected_paper_count: int = TARGET_COUNT
) -> tuple[IndexChunk, ...]:
    """Validate the complete JSONL input before creating external resources."""
    chunks = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or set(item) != CHUNK_FIELDS:
                    raise IndexBuildError(f"Invalid chunk fields at {path}:{line_number}.")
                chunk = IndexChunk(
                    chunk_id=str(item["chunk_id"]),
                    paper_id=str(item["paper_id"]),
                    title=str(item["title"]),
                    year=int(item["year"]),
                    venue=str(item["venue"]),
                    url=str(item["url"]),
                    page_number=int(item["page_number"]),
                    chunk_index=int(item["chunk_index"]),
                    pdf_sha256=str(item["pdf_sha256"]),
                    chunk_sha256=str(item["chunk_sha256"]),
                    text=str(item["text"]),
                )
                chunks.append(chunk)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise IndexBuildError(f"Cannot load chunks JSONL {path}: {error}") from error

    if not chunks:
        raise IndexBuildError(f"Chunks JSONL is empty: {path}")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise IndexBuildError("Chunks JSONL contains duplicate chunk IDs.")
    paper_ids = {chunk.paper_id for chunk in chunks}
    if len(paper_ids) != expected_paper_count:
        raise IndexBuildError(
            f"Chunks JSONL represents {len(paper_ids)} papers; expected {expected_paper_count}."
        )
    return tuple(chunks)


def _atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(content, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _batched(chunks: tuple[IndexChunk, ...], size: int) -> Iterator[tuple[IndexChunk, ...]]:
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size]


def _validate_vectors(vectors: Any, *, expected_count: int, dimension: int | None) -> int:
    try:
        materialized = [list(vector) for vector in vectors]
    except TypeError as error:
        raise IndexBuildError("Embedder returned non-iterable vectors.") from error
    if len(materialized) != expected_count:
        raise IndexBuildError(
            f"Embedder returned {len(materialized)} vectors for {expected_count} chunks."
        )
    dimensions = {len(vector) for vector in materialized}
    if len(dimensions) != 1 or not dimensions or 0 in dimensions:
        raise IndexBuildError("Embedding vectors have empty or inconsistent dimensions.")
    current_dimension = dimensions.pop()
    if dimension is not None and current_dimension != dimension:
        raise IndexBuildError(
            f"Embedding dimension changed from {dimension} to {current_dimension}."
        )
    return current_dimension


def _validate_collection(
    collection: Any,
    *,
    chunks: tuple[IndexChunk, ...],
    embedding_dimension: int,
    expected_paper_count: int,
) -> None:
    if collection.count() != len(chunks):
        raise IndexBuildError(f"Collection count is {collection.count()}; expected {len(chunks)}.")

    expected_ids = [chunk.chunk_id for chunk in chunks[:VALIDATION_SAMPLE_SIZE]]
    result = collection.get(ids=expected_ids, include=["documents", "metadatas", "embeddings"])
    returned_ids = list(result.get("ids") or [])
    documents = list(result.get("documents") or [])
    metadatas = list(result.get("metadatas") or [])
    raw_embeddings = result.get("embeddings")
    embeddings = [] if raw_embeddings is None else list(raw_embeddings)
    if set(returned_ids) != set(expected_ids) or len(documents) != len(expected_ids):
        raise IndexBuildError("Sample records could not be read back from the collection.")
    if len(metadatas) != len(expected_ids) or len(embeddings) != len(expected_ids):
        raise IndexBuildError("Sample metadata or embeddings are missing from the collection.")
    if any(len(vector) != embedding_dimension for vector in embeddings):
        raise IndexBuildError("Read-back embedding dimension does not match the build.")
    expected_by_id = {chunk.chunk_id: chunk for chunk in chunks[:VALIDATION_SAMPLE_SIZE]}
    for index, returned_id in enumerate(returned_ids):
        expected = expected_by_id[returned_id]
        if documents[index] != expected.text or metadatas[index] != expected.metadata:
            raise IndexBuildError("Sample document or metadata does not match the JSONL input.")
    all_records = collection.get(include=["metadatas"])
    all_ids = list(all_records.get("ids") or [])
    if len(all_ids) != len(chunks) or len(set(all_ids)) != len(chunks):
        raise IndexBuildError("Collection IDs are missing or not unique after the build.")
    all_metadatas = all_records.get("metadatas") or []
    paper_ids = {metadata["paper_id"] for metadata in all_metadatas}
    if len(paper_ids) != expected_paper_count:
        raise IndexBuildError(
            f"Collection represents {len(paper_ids)} papers; expected {expected_paper_count}."
        )


def _default_client_factory(path: Path) -> Any:
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def build_index(
    *,
    chunks_path: Path,
    corpus_manifest_path: Path,
    chroma_path: Path,
    collection_prefix: str,
    embedding_model: str,
    index_manifest_path: Path,
    active_pointer_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    expected_paper_count: int = TARGET_COUNT,
    embedder_factory: Callable[[str], Any] = SentenceTransformerEmbedder,
    client_factory: Callable[[Path], Any] = _default_client_factory,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> IndexBuildSummary:
    """Build and validate a staging collection before atomically activating it."""
    started = time.perf_counter()
    if batch_size <= 0:
        raise IndexBuildError("Embedding batch size must be positive.")
    chunks = load_index_chunks(chunks_path, expected_paper_count=expected_paper_count)
    corpus_hash = sha256_file(corpus_manifest_path)
    chunks_hash = sha256_file(chunks_path)
    identity = compute_index_identity(
        embedding_model=embedding_model,
        corpus_manifest_sha256=corpus_hash,
        chunks_jsonl_sha256=chunks_hash,
    )

    build_time = now().astimezone(UTC)
    timestamp_suffix = build_time.strftime("%Y%m%dT%H%M%S%fZ")
    collection_name = f"{collection_prefix}_{identity[:12]}_{timestamp_suffix}"

    try:
        embedder = embedder_factory(embedding_model)
    except Exception as error:
        raise IndexBuildError(f"Could not initialize embedding model: {error}") from error
    if getattr(embedder, "model_name", None) != embedding_model:
        raise IndexBuildError(
            "Embedder model/config mismatch: "
            f"configured {embedding_model!r}, got {getattr(embedder, 'model_name', None)!r}."
        )

    chroma_path.mkdir(parents=True, exist_ok=True)
    try:
        client = client_factory(chroma_path)
        collection = client.create_collection(
            name=collection_name,
            metadata={"embedding_model": embedding_model, "index_identity": identity},
        )
    except Exception as error:
        raise IndexBuildError(f"Could not create staging collection: {error}") from error
    embedding_dimension: int | None = None
    for batch in _batched(chunks, batch_size):
        try:
            vectors = embedder.encode([chunk.text for chunk in batch])
        except Exception as error:
            raise IndexBuildError(f"Embedding batch failed: {error}") from error
        embedding_dimension = _validate_vectors(
            vectors, expected_count=len(batch), dimension=embedding_dimension
        )
        try:
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=vectors,
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
            )
        except Exception as error:
            raise IndexBuildError(f"Writing embedding batch failed: {error}") from error
    if embedding_dimension is None:
        raise IndexBuildError("No embeddings were produced.")

    try:
        _validate_collection(
            collection,
            chunks=chunks,
            embedding_dimension=embedding_dimension,
            expected_paper_count=expected_paper_count,
        )
    except IndexBuildError:
        raise
    except Exception as error:
        raise IndexBuildError(f"Collection validation failed: {error}") from error
    manifest = IndexManifest(
        collection_name=collection_name,
        index_identity=identity,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        corpus_manifest_sha256=corpus_hash,
        chunks_jsonl_sha256=chunks_hash,
        paper_count=expected_paper_count,
        chunk_count=len(chunks),
        build_timestamp=build_time.isoformat(),
    )
    try:
        _atomic_write_json(index_manifest_path, manifest.to_dict())
        _atomic_write_json(active_pointer_path, {"collection_name": collection_name})
    except OSError as error:
        raise IndexBuildError(f"Could not publish index runtime manifests: {error}") from error
    return IndexBuildSummary(
        manifest=manifest,
        chroma_path=chroma_path,
        index_manifest_path=index_manifest_path,
        active_pointer_path=active_pointer_path,
        duration_seconds=time.perf_counter() - started,
        validated=True,
    )
