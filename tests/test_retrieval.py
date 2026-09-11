from __future__ import annotations

import json
from dataclasses import replace

import pytest

import rag_query
from config import AppConfig
from retrieval import RetrievalError, RetrievalResult, RetrievalService, load_active_index


class FakeEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeCollection:
    def __init__(self, *, count: int = 2, model: str = "test-model") -> None:
        self._count = count
        self.metadata = {"embedding_model": model, "index_identity": "a" * 64}

    def count(self):
        return self._count

    def query(self, *, query_embeddings, n_results, include):
        assert query_embeddings == [[0.1, 0.2, 0.3]]
        assert n_results == 2
        assert include == ["documents", "metadatas", "distances"]
        return {
            "ids": [["chunk-b", "chunk-a"]],
            "documents": [["second-ranked text", "first-paper text"]],
            "metadatas": [
                [
                    {
                        "paper_id": "paper-b",
                        "title": "Second Result",
                        "year": 2026,
                        "venue": "Venue B",
                        "page": 7,
                        "url": "https://example.invalid/b",
                    },
                    {
                        "paper_id": "paper-a",
                        "title": "First Paper",
                        "year": 2025,
                        "venue": "Venue A",
                        "page": 3,
                        "url": "https://example.invalid/a",
                    },
                ]
            ],
            "distances": [[0.12, 0.35]],
        }


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.requested_name = None

    def get_collection(self, *, name):
        self.requested_name = name
        return self.collection


def base_config(tmp_path, *, model: str = "test-model") -> AppConfig:
    return replace(
        AppConfig.load({}, load_env_file=False),
        chroma_path=tmp_path / "chroma",
        embedding_model=model,
        llm_base_url=None,
        llm_api_key=None,
        llm_model=None,
    )


def write_runtime_manifests(config: AppConfig, *, collection_name: str = "active-test"):
    config.chroma_path.mkdir(parents=True)
    (config.chroma_path / "active_index.json").write_text(
        json.dumps({"collection_name": collection_name})
    )
    (config.chroma_path / "index_manifest.json").write_text(
        json.dumps(
            {
                "collection_name": collection_name,
                "index_identity": "a" * 64,
                "embedding_model": "test-model",
                "embedding_dimension": 3,
                "corpus_manifest_sha256": "b" * 64,
                "chunks_jsonl_sha256": "c" * 64,
                "paper_count": 2,
                "chunk_count": 2,
                "build_timestamp": "2026-09-10T00:00:00+00:00",
            }
        )
    )


def make_service(config: AppConfig, collection: FakeCollection | None = None):
    collection = collection or FakeCollection()
    client = FakeClient(collection)
    service = RetrievalService(
        config,
        embedder_factory=FakeEmbedder,
        client_factory=lambda path: client,
    )
    return service, client


def test_active_pointer_and_manifest_are_validated(tmp_path):
    config = base_config(tmp_path)
    write_runtime_manifests(config)

    active = load_active_index(config)
    assert active.collection_name == "active-test"
    assert active.embedding_model == "test-model"
    assert active.embedding_dimension == 3
    assert active.corpus_manifest_sha256 == "b" * 64

    (config.chroma_path / "active_index.json").write_text(
        json.dumps({"collection_name": "different"})
    )
    with pytest.raises(RetrievalError, match="different collections"):
        load_active_index(config)


def test_retrieval_preserves_order_and_assigns_stable_source_ids(tmp_path):
    config = base_config(tmp_path)
    write_runtime_manifests(config)
    service, client = make_service(config)

    results = service.retrieve("emotion recognition", top_k=2)

    assert client.requested_name == "active-test"
    assert [result.chunk_id for result in results] == ["chunk-b", "chunk-a"]
    assert [result.rank for result in results] == [1, 2]
    assert [result.source_id for result in results] == ["[S1]", "[S2]"]
    assert [result.distance for result in results] == [0.12, 0.35]


def test_supporting_chunks_are_filtered_and_follow_selected_paper_order(tmp_path):
    config = base_config(tmp_path)
    write_runtime_manifests(config)

    class FilteredCollection(FakeCollection):
        def query(self, *, query_embeddings, n_results, include, where=None):
            assert query_embeddings == [[0.1, 0.2, 0.3]]
            assert n_results == 1
            paper_id = where["paper_id"]
            selected = {
                "paper-b": ("chunk-b", "support b", 0.2, 7),
                "paper-a": ("chunk-a", "support a", 0.3, 3),
            }[paper_id]
            chunk_id, text, distance, page = selected
            return {
                "ids": [[chunk_id]],
                "documents": [[text]],
                "metadatas": [
                    [
                        {
                            "paper_id": paper_id,
                            "title": f"Title {paper_id}",
                            "year": 2026,
                            "venue": "ACL",
                            "page": page,
                            "url": f"https://example.invalid/{paper_id}",
                        }
                    ]
                ],
                "distances": [[distance]],
            }

    service, _ = make_service(config, FilteredCollection())

    results = service.supporting_chunks([0.1, 0.2, 0.3], ("paper-b", "paper-a"))

    assert [item.paper_id for item in results] == ["paper-b", "paper-a"]
    assert [item.chunk_id for item in results] == ["chunk-b", "chunk-a"]
    assert [item.source_id for item in results] == ["[S1]", "[S2]"]


def test_missing_empty_index_and_model_mismatch_fail_clearly(tmp_path):
    config = base_config(tmp_path)
    with pytest.raises(RetrievalError, match="pointer.*missing"):
        load_active_index(config)

    write_runtime_manifests(config)
    service, _ = make_service(config, FakeCollection(count=0))
    with pytest.raises(RetrievalError, match="collection is empty"):
        service.retrieve("test")

    class MissingCollectionClient:
        def get_collection(self, *, name):
            raise KeyError(name)

    missing_collection = RetrievalService(
        config,
        embedder_factory=FakeEmbedder,
        client_factory=lambda path: MissingCollectionClient(),
    )
    with pytest.raises(RetrievalError, match="collection is missing"):
        missing_collection.retrieve("test")

    mismatched = replace(config, embedding_model="different-model")
    client_called = False

    def client_factory(path):
        nonlocal client_called
        client_called = True
        return FakeClient(FakeCollection())

    service = RetrievalService(
        mismatched,
        embedder_factory=FakeEmbedder,
        client_factory=client_factory,
    )
    with pytest.raises(RetrievalError, match="Embedding model mismatch"):
        service.retrieve("test")
    assert client_called is False


def test_retrieval_does_not_require_llm_configuration(tmp_path):
    config = base_config(tmp_path)
    write_runtime_manifests(config)
    service, _ = make_service(config)

    assert service.retrieve("test", top_k=2)[0].source_id == "[S1]"


def test_json_cli_output_does_not_initialize_llm(monkeypatch, tmp_path, capsys):
    config = base_config(tmp_path)
    result = RetrievalResult(
        rank=1,
        source_id="[S1]",
        distance=0.25,
        chunk_id="chunk-1",
        paper_id="paper-1",
        title="Paper One",
        year=2026,
        venue="ACL",
        page=4,
        url="https://example.invalid/1",
        text="retrieved text",
    )

    class FakeService:
        def __init__(self, received_config):
            assert received_config is config

        def retrieve(self, query, *, top_k):
            assert query == "test query"
            assert top_k == 5
            return [result]

    monkeypatch.setattr(rag_query.AppConfig, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(rag_query, "RetrievalService", FakeService)
    monkeypatch.setattr(
        rag_query.RAGQuerySystem,
        "_initialize_generation",
        lambda *args, **kwargs: pytest.fail("LLM initialization must not run"),
    )

    exit_code = rag_query.main(
        ["--query", "test query", "--top-k", "5", "--retrieval-only", "--json"]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output == [result.to_dict()]
