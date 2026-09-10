from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from indexing import IndexBuildError, build_index, compute_index_identity


class FakeEmbedder:
    def __init__(self, model_name: str, *, dimension: int = 3) -> None:
        self.model_name = model_name
        self.dimension = dimension

    def encode(self, texts):
        return [
            [float(len(text)), float(index), 1.0][: self.dimension]
            for index, text in enumerate(texts)
        ]


class FailingEmbedder(FakeEmbedder):
    def encode(self, texts):
        raise RuntimeError("controlled embedding failure")


class FakeCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name = name
        self.collection_metadata = metadata
        self.rows = {}

    def add(self, *, ids, embeddings, documents, metadatas):
        for item_id, embedding, document, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            if item_id in self.rows:
                raise ValueError(f"duplicate ID: {item_id}")
            self.rows[item_id] = {
                "embedding": list(embedding),
                "document": document,
                "metadata": metadata,
            }

    def count(self):
        return len(self.rows)

    def get(self, ids=None, include=None):
        selected_ids = (
            list(self.rows) if ids is None else [item_id for item_id in ids if item_id in self.rows]
        )
        return {
            "ids": selected_ids,
            "documents": [self.rows[item_id]["document"] for item_id in selected_ids],
            "metadatas": [self.rows[item_id]["metadata"] for item_id in selected_ids],
            "embeddings": [self.rows[item_id]["embedding"] for item_id in selected_ids],
        }


class FakeClient:
    def __init__(self) -> None:
        self.collections = {}

    def create_collection(self, *, name: str, metadata: dict):
        if name in self.collections:
            raise ValueError(f"collection already exists: {name}")
        collection = FakeCollection(name, metadata)
        self.collections[name] = collection
        return collection


def write_inputs(tmp_path, *, paper_count: int = 2):
    corpus_manifest = tmp_path / "corpus/manifest.json"
    corpus_manifest.parent.mkdir(parents=True)
    corpus_manifest.write_text('[{"fixed":true}]\n')
    chunks_path = tmp_path / "data/processed/chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunks = []
    for paper_number in range(paper_count):
        for chunk_number in range(2):
            chunks.append(
                {
                    "chunk_id": f"chunk-{paper_number}-{chunk_number}",
                    "paper_id": f"paper-{paper_number}",
                    "title": f"Paper {paper_number}",
                    "year": 2025,
                    "venue": "ACL",
                    "url": f"https://example.invalid/paper-{paper_number}.pdf",
                    "page_number": chunk_number + 1,
                    "chunk_index": 0,
                    "pdf_sha256": f"{paper_number + 1:064x}",
                    "chunk_sha256": f"{chunk_number + 10:064x}",
                    "abstract": "not indexed as Chroma metadata",
                    "text": f"paper {paper_number} chunk {chunk_number}",
                }
            )
    chunks_path.write_text("".join(json.dumps(chunk) + "\n" for chunk in chunks))
    return corpus_manifest, chunks_path


def build_with_fakes(tmp_path, *, embedder_factory=FakeEmbedder):
    corpus_manifest, chunks_path = write_inputs(tmp_path)
    client = FakeClient()
    summary = build_index(
        chunks_path=chunks_path,
        corpus_manifest_path=corpus_manifest,
        chroma_path=tmp_path / "chroma",
        collection_prefix="test_collection",
        embedding_model="test-model",
        index_manifest_path=tmp_path / "chroma/index_manifest.json",
        active_pointer_path=tmp_path / "chroma/active_index.json",
        batch_size=2,
        expected_paper_count=2,
        embedder_factory=embedder_factory,
        client_factory=lambda path: client,
        now=lambda: datetime(2026, 9, 10, 12, 0, tzinfo=UTC),
    )
    return summary, client


def test_index_identity_and_manifest_fields_are_deterministic(tmp_path):
    identity = compute_index_identity(
        embedding_model="model",
        corpus_manifest_sha256="a" * 64,
        chunks_jsonl_sha256="b" * 64,
    )
    assert identity == compute_index_identity(
        embedding_model="model",
        corpus_manifest_sha256="a" * 64,
        chunks_jsonl_sha256="b" * 64,
    )

    summary, _ = build_with_fakes(tmp_path)
    manifest = json.loads(summary.index_manifest_path.read_text())

    assert manifest == summary.manifest.to_dict()
    assert set(manifest) == {
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
    assert manifest["index_identity"][:12] in manifest["collection_name"]


def test_successful_build_validates_and_atomically_updates_pointer(tmp_path):
    summary, client = build_with_fakes(tmp_path)
    pointer = json.loads(summary.active_pointer_path.read_text())
    collection = client.collections[summary.manifest.collection_name]

    assert summary.validated is True
    assert summary.manifest.paper_count == 2
    assert summary.manifest.chunk_count == 4
    assert summary.manifest.embedding_dimension == 3
    assert collection.count() == 4
    assert pointer == {"collection_name": summary.manifest.collection_name}
    assert all(
        set(row["metadata"])
        == {
            "paper_id",
            "title",
            "year",
            "venue",
            "url",
            "page",
            "chunk_index",
            "pdf_sha256",
            "chunk_sha256",
        }
        for row in collection.rows.values()
    )


def test_failed_build_leaves_previous_active_pointer_unchanged(tmp_path):
    corpus_manifest, chunks_path = write_inputs(tmp_path)
    pointer_path = tmp_path / "chroma/active_index.json"
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_text('{"collection_name":"previous_valid"}\n')
    previous = pointer_path.read_bytes()
    manifest_path = tmp_path / "chroma/index_manifest.json"
    manifest_path.write_text('{"collection_name":"previous_valid"}\n')
    previous_manifest = manifest_path.read_bytes()

    with pytest.raises(IndexBuildError, match="controlled embedding failure"):
        build_index(
            chunks_path=chunks_path,
            corpus_manifest_path=corpus_manifest,
            chroma_path=tmp_path / "chroma",
            collection_prefix="test_collection",
            embedding_model="test-model",
            index_manifest_path=manifest_path,
            active_pointer_path=pointer_path,
            expected_paper_count=2,
            embedder_factory=FailingEmbedder,
            client_factory=lambda path: FakeClient(),
        )

    assert pointer_path.read_bytes() == previous
    assert manifest_path.read_bytes() == previous_manifest


def test_embedder_model_config_mismatch_is_detected_before_chroma_creation(tmp_path):
    corpus_manifest, chunks_path = write_inputs(tmp_path)
    client_called = False

    def wrong_embedder(model_name):
        return FakeEmbedder("different-model")

    def client_factory(path):
        nonlocal client_called
        client_called = True
        return FakeClient()

    with pytest.raises(IndexBuildError, match="model/config mismatch"):
        build_index(
            chunks_path=chunks_path,
            corpus_manifest_path=corpus_manifest,
            chroma_path=tmp_path / "chroma",
            collection_prefix="test_collection",
            embedding_model="configured-model",
            index_manifest_path=tmp_path / "chroma/index_manifest.json",
            active_pointer_path=tmp_path / "chroma/active_index.json",
            expected_paper_count=2,
            embedder_factory=wrong_embedder,
            client_factory=client_factory,
        )

    assert client_called is False


def test_index_build_does_not_require_llm_configuration(tmp_path):
    summary, _ = build_with_fakes(tmp_path)

    assert summary.validated is True
    assert summary.manifest.embedding_model == "test-model"
