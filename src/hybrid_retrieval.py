"""Paper-level BM25, reciprocal-rank fusion, and optional BGE reranking."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import AppConfig
from corpus import normalize_text
from retrieval import RetrievalError, RetrievalResult, RetrievalService

DENSE_CANDIDATE_COUNT = 20
BM25_CANDIDATE_COUNT = 20
RRF_CANDIDATE_COUNT = 20
DEFAULT_RESULT_COUNT = 5
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
RERANK_DOCUMENT_MAX_CHARACTERS = 8_000
RERANK_TOKEN_MAX_LENGTH = 1_024
RERANK_BATCH_SIZE = 8

DENSE = "dense"
DENSE_DEDUP = "dense-dedup"
HYBRID = "hybrid"
HYBRID_RERANK = "hybrid-rerank"
RETRIEVAL_CONFIGURATIONS = (DENSE, DENSE_DEDUP, HYBRID, HYBRID_RERANK)


@dataclass(frozen=True, slots=True)
class PaperRecord:
    paper_id: str
    title: str
    abstract: str
    year: int
    venue: str
    url: str

    def rerank_document(self) -> str:
        """Return deterministic title-plus-abstract input for a cross-encoder."""
        return f"{self.title}\n\n{self.abstract}"[:RERANK_DOCUMENT_MAX_CHARACTERS]


@dataclass(frozen=True, slots=True)
class ScoredPaper:
    paper_id: str
    score: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_paper_records(
    chunks_path: Path, *, expected_sha256: str, expected_paper_count: int
) -> dict[str, PaperRecord]:
    """Aggregate immutable paper metadata from the active index's exact JSONL input."""
    if not chunks_path.is_file():
        raise RetrievalError(f"Prepared chunks are required for paper retrieval: {chunks_path}.")
    if _sha256_file(chunks_path) != expected_sha256:
        raise RetrievalError("Prepared chunks do not match the active index manifest.")

    papers: dict[str, PaperRecord] = {}
    try:
        with chunks_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                paper = PaperRecord(
                    paper_id=str(item["paper_id"]),
                    title=str(item["title"]),
                    abstract=str(item["abstract"]),
                    year=int(item["year"]),
                    venue=str(item["venue"]),
                    url=str(item["url"]),
                )
                previous = papers.setdefault(paper.paper_id, paper)
                if previous != paper:
                    raise RetrievalError(
                        f"Inconsistent paper metadata in chunks JSONL at line {line_number}."
                    )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RetrievalError(f"Cannot load paper metadata from {chunks_path}: {error}") from error

    if len(papers) != expected_paper_count:
        raise RetrievalError(
            f"Prepared chunks represent {len(papers)} papers; "
            f"active index declares {expected_paper_count}."
        )
    return papers


class BM25Index:
    """A deterministic in-memory BM25 index over paper title and abstract."""

    def __init__(
        self,
        papers: Sequence[PaperRecord],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if not papers:
            raise RetrievalError("BM25 requires at least one paper.")
        self.papers = tuple(sorted(papers, key=lambda paper: paper.paper_id))
        self.k1 = k1
        self.b = b
        self.term_frequencies = tuple(
            Counter(_tokenize(f"{paper.title} {paper.abstract}")) for paper in self.papers
        )
        self.document_lengths = tuple(sum(counts.values()) for counts in self.term_frequencies)
        self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)
        document_frequency: Counter[str] = Counter()
        for counts in self.term_frequencies:
            document_frequency.update(counts.keys())
        self.inverse_document_frequency = {
            term: math.log(1 + (len(self.papers) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def rank(self, query: str, *, top_k: int = BM25_CANDIDATE_COUNT) -> tuple[ScoredPaper, ...]:
        if top_k <= 0:
            raise RetrievalError("BM25 top-k must be positive.")
        query_terms = _tokenize(query)
        scored = []
        for paper, frequencies, length in zip(
            self.papers, self.term_frequencies, self.document_lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_document_length
                )
                score += self.inverse_document_frequency[term] * (
                    frequency * (self.k1 + 1) / denominator
                )
            if score > 0:
                scored.append(ScoredPaper(paper.paper_id, score))
        scored.sort(key=lambda item: (-item.score, item.paper_id))
        return tuple(scored[:top_k])


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(normalize_text(text).split())


def deduplicate_papers(results: Sequence[RetrievalResult]) -> tuple[str, ...]:
    """Keep each paper's first dense occurrence without filling from later queries."""
    seen = set()
    paper_ids = []
    for result in results:
        if result.paper_id not in seen:
            seen.add(result.paper_id)
            paper_ids.append(result.paper_id)
    return tuple(paper_ids)


def reciprocal_rank_fusion(
    dense_paper_ids: Sequence[str], bm25_paper_ids: Sequence[str], *, k: int = RRF_K
) -> tuple[ScoredPaper, ...]:
    """Fuse paper rankings with deterministic ties and no duplicate contributions."""
    if k <= 0:
        raise RetrievalError("RRF k must be positive.")
    scores: dict[str, float] = {}
    for ranking in (dense_paper_ids, bm25_paper_ids):
        for rank, paper_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (k + rank)
    fused = [ScoredPaper(paper_id, score) for paper_id, score in scores.items()]
    fused.sort(key=lambda item: (-item.score, item.paper_id))
    return tuple(fused)


class BGEReranker:
    """Lazily load BGE and score deterministic query/document pairs."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None

    def initialize(self) -> None:
        """Load tokenizer and model without running a query pair."""
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.torch = torch
        except Exception as error:
            raise RetrievalError(
                f"Could not initialize reranker {self.model_name}: {error}"
            ) from error

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        self.initialize()
        scores = []
        for start in range(0, len(documents), RERANK_BATCH_SIZE):
            batch = list(documents[start : start + RERANK_BATCH_SIZE])
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=RERANK_TOKEN_MAX_LENGTH,
                return_tensors="pt",
            )
            try:
                with self.torch.inference_mode():
                    logits = self.model(**inputs, return_dict=True).logits
                scores.extend(logits.view(-1).float().cpu().tolist())
            except Exception as error:
                raise RetrievalError(f"Reranker inference failed: {error}") from error
        return [float(score) for score in scores]


class PaperRetrievalService:
    """Compose dense, BM25, RRF, reranking, and paper-restricted chunk support."""

    def __init__(
        self,
        config: AppConfig,
        *,
        dense_service: RetrievalService | None = None,
        reranker_factory: Callable[[str], Any] = BGEReranker,
    ) -> None:
        self.config = config
        self.dense_service = dense_service or RetrievalService(config)
        self.reranker_factory = reranker_factory
        self.papers: dict[str, PaperRecord] | None = None
        self.bm25: BM25Index | None = None
        self.reranker: Any | None = None

    def _initialize_papers(self) -> None:
        if self.papers is not None:
            return
        self.dense_service.initialize()
        active = self.dense_service.active_index
        if active is None:
            raise RetrievalError("Active index identity is unavailable for paper retrieval.")
        self.papers = load_paper_records(
            self.config.processed_dir / "chunks.jsonl",
            expected_sha256=active.chunks_jsonl_sha256,
            expected_paper_count=active.paper_count,
        )
        self.bm25 = BM25Index(tuple(self.papers.values()))

    def prepare(self, configuration: str) -> None:
        """Initialize configuration-specific resources before query timing begins."""
        if configuration not in RETRIEVAL_CONFIGURATIONS:
            raise RetrievalError(f"Unknown retrieval configuration: {configuration}.")
        self.dense_service.initialize()
        if configuration in {HYBRID, HYBRID_RERANK}:
            self._initialize_papers()
        if configuration == HYBRID_RERANK:
            if self.reranker is None:
                self.reranker = self.reranker_factory(self.config.reranker_model)
            self.reranker.initialize()

    def retrieve(
        self,
        query: str,
        *,
        configuration: str = HYBRID_RERANK,
        top_k: int = DEFAULT_RESULT_COUNT,
    ) -> list[RetrievalResult]:
        if configuration not in RETRIEVAL_CONFIGURATIONS:
            raise RetrievalError(f"Unknown retrieval configuration: {configuration}.")
        if top_k <= 0 or top_k > RRF_CANDIDATE_COUNT:
            raise RetrievalError(f"top-k must be between 1 and {RRF_CANDIDATE_COUNT}.")
        if configuration == DENSE:
            return self.dense_service.retrieve(query, top_k=top_k)

        query_vector = self.dense_service.embed_query(query)
        dense_chunks = self.dense_service.retrieve_by_vector(
            query_vector, top_k=DENSE_CANDIDATE_COUNT
        )
        dense_papers = deduplicate_papers(dense_chunks)

        if configuration == DENSE_DEDUP:
            selected_ids = dense_papers[:top_k]
        else:
            self._initialize_papers()
            bm25_ranking = self.bm25.rank(query, top_k=BM25_CANDIDATE_COUNT)
            if bm25_ranking:
                fused = reciprocal_rank_fusion(
                    dense_papers, [candidate.paper_id for candidate in bm25_ranking]
                )
                candidate_ids = tuple(
                    candidate.paper_id for candidate in fused[:RRF_CANDIDATE_COUNT]
                )
            else:
                candidate_ids = dense_papers[:RRF_CANDIDATE_COUNT]

            if configuration == HYBRID_RERANK:
                candidate_ids = self._rerank(query, candidate_ids)
            selected_ids = candidate_ids[:top_k]

        if len(selected_ids) < top_k:
            raise RetrievalError(
                f"Only {len(selected_ids)} unique paper candidates were available; "
                f"{top_k} requested."
            )
        return self.dense_service.supporting_chunks(query_vector, selected_ids)

    def _rerank(self, query: str, candidate_ids: Sequence[str]) -> tuple[str, ...]:
        if self.papers is None:
            raise RetrievalError("Paper metadata is unavailable for reranking.")
        if self.reranker is None:
            self.reranker = self.reranker_factory(self.config.reranker_model)
        documents = [self.papers[paper_id].rerank_document() for paper_id in candidate_ids]
        scores = self.reranker.score_pairs(query, documents)
        if len(scores) != len(candidate_ids) or any(not math.isfinite(score) for score in scores):
            raise RetrievalError("Reranker returned invalid scores.")
        ranked = [
            (paper_id, float(score), original_rank)
            for original_rank, (paper_id, score) in enumerate(
                zip(candidate_ids, scores, strict=True), start=1
            )
        ]
        ranked.sort(key=lambda item: (-item[1], item[2], item[0]))
        return tuple(paper_id for paper_id, _, _ in ranked)
