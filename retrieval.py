"""Validated retrieval from the currently active local Chroma collection."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config import AppConfig
from indexing import SentenceTransformerEmbedder

ACTIVE_POINTER_FILENAME = "active_index.json"
INDEX_MANIFEST_FILENAME = "index_manifest.json"
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "collection_name",
        "index_identity",
        "embedding_model",
        "embedding_dimension",
        "corpus_manifest_sha256",
        "chunks_jsonl_sha256",
        "paper_count",
        "chunk_count",
        "build_timestamp",
    }
)
REQUIRED_RESULT_METADATA = frozenset({"paper_id", "title", "year", "venue", "page", "url"})


class RetrievalError(RuntimeError):
    """Raised when the active index cannot be validated or queried."""


@dataclass(frozen=True, slots=True)
class ActiveIndex:
    collection_name: str
    index_identity: str
    embedding_model: str
    embedding_dimension: int
    corpus_manifest_sha256: str
    chunks_jsonl_sha256: str
    paper_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    rank: int
    source_id: str
    distance: float
    chunk_id: str
    paper_id: str
    title: str
    year: int
    venue: str
    page: int
    url: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RetrievalError(f"{label} is missing: {path}")
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(content, dict):
        raise RetrievalError(f"{label} must contain a JSON object: {path}")
    return content


def load_active_index(config: AppConfig) -> ActiveIndex:
    """Validate the active pointer and its index manifest before opening Chroma."""
    pointer_path = config.chroma_path / ACTIVE_POINTER_FILENAME
    manifest_path = config.chroma_path / INDEX_MANIFEST_FILENAME
    pointer = _read_json_object(pointer_path, label="Active-index pointer")
    collection_name = pointer.get("collection_name")
    if not isinstance(collection_name, str) or not collection_name:
        raise RetrievalError(f"Active-index pointer has no valid collection_name: {pointer_path}")

    manifest = _read_json_object(manifest_path, label="Index manifest")
    missing = REQUIRED_MANIFEST_FIELDS - manifest.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise RetrievalError(f"Index manifest is missing required fields: {fields}")
    if manifest["collection_name"] != collection_name:
        raise RetrievalError(
            "Active-index pointer and index manifest identify different collections."
        )
    if manifest["embedding_model"] != config.embedding_model:
        raise RetrievalError(
            "Embedding model mismatch: "
            f"index uses {manifest['embedding_model']!r}, configured {config.embedding_model!r}."
        )

    integer_fields = ("embedding_dimension", "paper_count", "chunk_count")
    if any(
        not isinstance(manifest[field], int)
        or isinstance(manifest[field], bool)
        or manifest[field] <= 0
        for field in integer_fields
    ):
        raise RetrievalError("Index manifest has invalid dimension or count fields.")
    for field in ("index_identity", "corpus_manifest_sha256", "chunks_jsonl_sha256"):
        value = manifest[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RetrievalError(f"Index manifest has an invalid {field}.")
    if not isinstance(manifest["build_timestamp"], str) or not manifest["build_timestamp"]:
        raise RetrievalError("Index manifest has an invalid build_timestamp.")

    return ActiveIndex(
        collection_name=collection_name,
        index_identity=manifest["index_identity"],
        embedding_model=manifest["embedding_model"],
        embedding_dimension=manifest["embedding_dimension"],
        corpus_manifest_sha256=manifest["corpus_manifest_sha256"],
        chunks_jsonl_sha256=manifest["chunks_jsonl_sha256"],
        paper_count=manifest["paper_count"],
        chunk_count=manifest["chunk_count"],
    )


def _default_client_factory(path: Path) -> Any:
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


class RetrievalService:
    """Open and query the validated active collection without creating one."""

    def __init__(
        self,
        config: AppConfig,
        *,
        embedder_factory: Callable[[str], Any] = SentenceTransformerEmbedder,
        client_factory: Callable[[Path], Any] = _default_client_factory,
    ) -> None:
        self.config = config
        self.embedder_factory = embedder_factory
        self.client_factory = client_factory
        self.active_index: ActiveIndex | None = None
        self.embedder: Any | None = None
        self.client: Any | None = None
        self.collection: Any | None = None

    def initialize(self) -> None:
        if self.collection is not None:
            return
        active_index = load_active_index(self.config)
        try:
            client = self.client_factory(self.config.chroma_path)
            collection = client.get_collection(name=active_index.collection_name)
        except Exception as error:
            raise RetrievalError(
                f"Active Chroma collection is missing or cannot be opened: "
                f"{active_index.collection_name}: {error}"
            ) from error

        count = collection.count()
        if count <= 0:
            raise RetrievalError(
                f"Active Chroma collection is empty: {active_index.collection_name}"
            )
        if count != active_index.chunk_count:
            raise RetrievalError(
                f"Active collection contains {count} chunks; "
                f"index manifest declares {active_index.chunk_count}."
            )
        collection_metadata = collection.metadata or {}
        if collection_metadata.get("embedding_model") != active_index.embedding_model:
            raise RetrievalError("Collection and index manifest embedding models do not match.")
        if collection_metadata.get("index_identity") != active_index.index_identity:
            raise RetrievalError("Collection and index manifest identities do not match.")

        try:
            embedder = self.embedder_factory(active_index.embedding_model)
        except Exception as error:
            raise RetrievalError(f"Could not initialize query embedding model: {error}") from error
        if getattr(embedder, "model_name", None) != active_index.embedding_model:
            raise RetrievalError("Query embedder and index manifest models do not match.")

        self.active_index = active_index
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def query_raw(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        """Return the raw ordered Chroma result for legacy generation code."""
        if not query.strip():
            raise RetrievalError("Query must not be empty.")
        if top_k <= 0:
            raise RetrievalError("top-k must be a positive integer.")
        self.initialize()
        try:
            vectors = self.embedder.encode([query])
            query_vector = list(vectors[0])
        except Exception as error:
            raise RetrievalError(f"Could not embed retrieval query: {error}") from error
        if len(query_vector) != self.active_index.embedding_dimension:
            raise RetrievalError(
                f"Query embedding dimension is {len(query_vector)}; "
                f"index requires {self.active_index.embedding_dimension}."
            )
        try:
            return self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self.active_index.chunk_count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            raise RetrievalError(f"Chroma query failed: {error}") from error

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Return ordered passages with stable response-level source IDs."""
        raw = self.query_raw(query, top_k=top_k)
        ids = _first_result_list(raw, "ids")
        documents = _first_result_list(raw, "documents")
        metadatas = _first_result_list(raw, "metadatas")
        distances = _first_result_list(raw, "distances")
        if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
            raise RetrievalError("Chroma returned result fields with inconsistent lengths.")

        results = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        ):
            if not isinstance(metadata, dict) or not metadata.keys() >= REQUIRED_RESULT_METADATA:
                raise RetrievalError(f"Retrieval result {rank} has incomplete provenance metadata.")
            results.append(
                RetrievalResult(
                    rank=rank,
                    source_id=f"[S{rank}]",
                    distance=float(distance),
                    chunk_id=str(chunk_id),
                    paper_id=str(metadata["paper_id"]),
                    title=str(metadata["title"]),
                    year=int(metadata["year"]),
                    venue=str(metadata["venue"]),
                    page=int(metadata["page"]),
                    url=str(metadata["url"]),
                    text=str(text),
                )
            )
        return results


def _first_result_list(raw: dict[str, Any], field: str) -> list[Any]:
    value = raw.get(field)
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        raise RetrievalError(f"Chroma returned an invalid {field} result.")
    return value[0]
