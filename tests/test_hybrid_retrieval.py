from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import rag_query
from config import AppConfig
from hybrid_retrieval import (
    DENSE_DEDUP,
    HYBRID,
    HYBRID_RERANK,
    BM25Index,
    PaperRecord,
    PaperRetrievalService,
    ScoredPaper,
    deduplicate_papers,
    load_paper_records,
    reciprocal_rank_fusion,
)
from indexing import sha256_file
from retrieval import RetrievalError, RetrievalResult


def paper(paper_id: str, *, title: str = "Emotion Recognition", abstract: str = ""):
    return PaperRecord(paper_id, title, abstract, 2026, "ACL", f"https://example/{paper_id}")


def result(paper_id: str, rank: int) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        source_id=f"[S{rank}]",
        distance=rank / 10,
        chunk_id=f"chunk-{paper_id}-{rank}",
        paper_id=paper_id,
        title=f"Title {paper_id}",
        year=2026,
        venue="ACL",
        page=rank,
        url=f"https://example/{paper_id}",
        text=f"support for {paper_id}",
    )


class FakeDenseService:
    def __init__(self, dense_results):
        self.dense_results = dense_results
        self.active_index = SimpleNamespace(
            chunks_jsonl_sha256="a" * 64,
            paper_count=len({item.paper_id for item in dense_results}),
        )
        self.support_requests = []

    def initialize(self):
        return None

    def embed_query(self, query):
        return [0.1, 0.2]

    def retrieve_by_vector(self, query_vector, *, top_k, paper_id=None):
        assert paper_id is None
        return self.dense_results[:top_k]

    def supporting_chunks(self, query_vector, paper_ids):
        self.support_requests.append(tuple(paper_ids))
        return [
            replace(result(paper_id, rank), source_id=f"[S{rank}]")
            for rank, paper_id in enumerate(paper_ids, 1)
        ]


class FakeBM25:
    def __init__(self, ranking):
        self.ranking = ranking

    def rank(self, query, *, top_k):
        return tuple(self.ranking[:top_k])


def configured_service(tmp_path, dense_results, bm25_ranking=(), reranker_factory=None):
    config = replace(
        AppConfig.load({}, load_env_file=False),
        processed_dir=tmp_path,
        reranker_model="fake-reranker",
    )
    dense = FakeDenseService(dense_results)
    options = {"dense_service": dense}
    if reranker_factory is not None:
        options["reranker_factory"] = reranker_factory
    service = PaperRetrievalService(config, **options)
    ids = {item.paper_id for item in dense_results} | {item.paper_id for item in bm25_ranking}
    service.papers = {paper_id: paper(paper_id, title=f"Title {paper_id}") for paper_id in ids}
    service.bm25 = FakeBM25(bm25_ranking)
    return service, dense


def test_bm25_keeps_only_positive_scores_and_rrf_is_deterministic():
    index = BM25Index([paper("b", title="unrelated"), paper("a", title="emotion recognition")])

    assert [item.paper_id for item in index.rank("emotion")] == ["a"]
    assert index.rank("不存在的詞") == ()
    fused = reciprocal_rank_fusion(["b", "a", "b"], ["a", "c"])
    assert [item.paper_id for item in fused] == ["a", "b", "c"]
    assert deduplicate_papers([result("b", 1), result("a", 2), result("b", 3)]) == (
        "b",
        "a",
    )


def test_paper_metadata_is_loaded_from_hashed_chunks(tmp_path):
    path = tmp_path / "chunks.jsonl"
    base = {
        "paper_id": "paper-a",
        "title": "Paper A",
        "abstract": "Abstract A",
        "year": 2026,
        "venue": "ACL",
        "url": "https://example/a",
    }
    path.write_text(json.dumps({**base, "text": "chunk"}) + "\n", encoding="utf-8")

    records = load_paper_records(path, expected_sha256=sha256_file(path), expected_paper_count=1)
    assert records["paper-a"].rerank_document() == "Paper A\n\nAbstract A"
    with pytest.raises(RetrievalError, match="do not match"):
        load_paper_records(path, expected_sha256="0" * 64, expected_paper_count=1)


def test_dense_dedup_and_bm25_empty_fallback_restrict_supporting_papers(tmp_path):
    dense_results = [result("a", 1), result("a", 2), result("b", 3), result("c", 4)]
    service, dense = configured_service(tmp_path, dense_results)

    deduped = service.retrieve("query", configuration=DENSE_DEDUP, top_k=3)
    hybrid = service.retrieve("無詞彙重疊", configuration=HYBRID, top_k=3)

    assert [item.paper_id for item in deduped] == ["a", "b", "c"]
    assert [item.paper_id for item in hybrid] == ["a", "b", "c"]
    assert dense.support_requests == [("a", "b", "c"), ("a", "b", "c")]
    assert [item.source_id for item in hybrid] == ["[S1]", "[S2]", "[S3]"]


def test_hybrid_reranks_unique_papers_using_title_and_abstract_lazily(tmp_path):
    dense_results = [result("a", 1), result("a", 2), result("b", 3), result("c", 4)]
    bm25 = [ScoredPaper("c", 3.0), ScoredPaper("b", 2.0)]
    calls = []

    class FakeReranker:
        def __init__(self, model_name):
            calls.append(("construct", model_name))

        def initialize(self):
            calls.append(("initialize",))

        def score_pairs(self, query, documents):
            calls.append((query, tuple(documents)))
            assert len(documents) == len(set(documents)) == 3
            return [0.1, 0.9, 0.2]

    service, dense = configured_service(
        tmp_path, dense_results, bm25, reranker_factory=FakeReranker
    )
    assert calls == []

    service.prepare(HYBRID_RERANK)
    assert calls == [("construct", "fake-reranker"), ("initialize",)]

    retrieved = service.retrieve("query", configuration=HYBRID_RERANK, top_k=3)

    assert all(document.startswith("Title ") for document in calls[2][1])
    assert [item.paper_id for item in retrieved] == ["b", "a", "c"]
    assert dense.support_requests == [("b", "a", "c")]


def test_rerank_document_truncation_is_deterministic():
    record = paper("a", title="Title", abstract="x" * 20_000)
    first = record.rerank_document()

    assert first == record.rerank_document()
    assert first.startswith("Title\n\n")
    assert len(first) == 8_000


def test_hybrid_json_cli_remains_retrieval_only(monkeypatch, tmp_path, capsys):
    config = replace(AppConfig.load({}, load_env_file=False), chroma_path=tmp_path)

    class FakePaperService:
        def __init__(self, received_config):
            assert received_config is config

        def retrieve(self, query, *, configuration, top_k):
            assert (query, configuration, top_k) == ("test", HYBRID_RERANK, 1)
            return [result("paper-a", 1)]

    monkeypatch.setattr(rag_query.AppConfig, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(rag_query, "PaperRetrievalService", FakePaperService)
    monkeypatch.setattr(
        rag_query.RAGQuerySystem,
        "_initialize_generation",
        lambda *args, **kwargs: pytest.fail("LLM initialization must not run"),
    )

    exit_code = rag_query.main(
        [
            "--query",
            "test",
            "--top-k",
            "1",
            "--retrieval-only",
            "--retrieval-config",
            HYBRID_RERANK,
            "--json",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)[0]["source_id"] == "[S1]"
