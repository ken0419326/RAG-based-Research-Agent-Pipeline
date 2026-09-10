"""Run the curated paper-level retrieval evaluation against the active index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import AppConfig, ConfigurationError
from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from reporting import ReportError, atomic_write_json
from retrieval import RetrievalError, RetrievalService

EVALUATION_TOP_K = 5
CANDIDATE_CHUNK_COUNT = 20
JUDGMENT_BASIS = "manually_curated_from_titles_and_abstracts"
QUERY_FIELDS = frozenset(
    {
        "query_id",
        "query",
        "language",
        "category",
        "relevant_paper_ids",
        "judgment_basis",
        "judgment_note",
    }
)
REQUIRED_CATEGORIES = frozenset(
    {"cross_language", "english_topic", "exact_lookup", "multiple_relevant", "unanswerable"}
)


class EvaluationError(RuntimeError):
    """Raised when evaluation inputs or runtime identity are invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    language: str
    category: str
    relevant_paper_ids: tuple[str, ...]
    judgment_basis: str
    judgment_note: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluation_queries(
    path: Path, *, manifest_paper_ids: set[str]
) -> tuple[EvaluationQuery, ...]:
    queries = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict) or set(item) != QUERY_FIELDS:
                    raise EvaluationError(f"Invalid evaluation fields at {path}:{line_number}.")
                relevant = item["relevant_paper_ids"]
                if not isinstance(relevant, list) or not all(
                    isinstance(value, str) for value in relevant
                ):
                    raise EvaluationError(f"Invalid relevant_paper_ids at {path}:{line_number}.")
                query = EvaluationQuery(
                    query_id=str(item["query_id"]),
                    query=str(item["query"]),
                    language=str(item["language"]),
                    category=str(item["category"]),
                    relevant_paper_ids=tuple(relevant),
                    judgment_basis=str(item["judgment_basis"]),
                    judgment_note=str(item["judgment_note"]),
                )
                queries.append(query)
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Cannot load evaluation set {path}: {error}") from error

    if not 10 <= len(queries) <= 15:
        raise EvaluationError("Evaluation set must contain approximately 12 queries (10–15).")
    query_ids = [query.query_id for query in queries]
    if len(set(query_ids)) != len(query_ids):
        raise EvaluationError("Evaluation query IDs must be unique.")
    if any(not query.query_id or not query.query.strip() for query in queries):
        raise EvaluationError("Evaluation query IDs and text must not be empty.")
    if any(query.language not in {"en", "zh-TW"} for query in queries):
        raise EvaluationError("Evaluation language must be en or zh-TW.")
    if not {query.category for query in queries} >= REQUIRED_CATEGORIES:
        raise EvaluationError("Evaluation set does not cover all required categories.")
    for query in queries:
        relevant = query.relevant_paper_ids
        if len(set(relevant)) != len(relevant):
            raise EvaluationError(f"Evaluation query {query.query_id} has duplicate relevant IDs.")
        unknown = set(relevant) - manifest_paper_ids
        if unknown:
            raise EvaluationError(
                f"Evaluation query {query.query_id} references unknown paper IDs: "
                f"{', '.join(sorted(unknown))}."
            )
        if query.judgment_basis != JUDGMENT_BASIS:
            raise EvaluationError(f"Evaluation query {query.query_id} has unknown judgment basis.")
        if query.category == "unanswerable" and relevant:
            raise EvaluationError("Unanswerable queries must have an empty relevant set.")
        if query.category != "unanswerable" and not relevant:
            raise EvaluationError(f"Evaluation query {query.query_id} needs relevant paper IDs.")
        if query.category == "multiple_relevant" and len(relevant) < 2:
            raise EvaluationError("Multiple-relevant queries require at least two relevant papers.")
    return tuple(queries)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Cannot read required JSON {path}: {error}") from error


def _prepared_statistics(config: AppConfig) -> dict[str, Any]:
    chunks_path = config.processed_dir / "chunks.jsonl"
    paper_ids = set()
    pages = set()
    chunk_count = 0
    try:
        with chunks_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                paper_id = item["paper_id"]
                paper_ids.add(paper_id)
                pages.add((paper_id, int(item["page_number"])))
                chunk_count += 1
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"Cannot verify prepared chunks {chunks_path}: {error}") from error
    return {
        "parsed_papers": len(paper_ids),
        "pages": len(pages),
        "chunks": chunk_count,
        "chunks_jsonl_sha256": _sha256_file(chunks_path),
    }


def _git_runtime(project_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        worktree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        tracked_dirty = None
        worktree_dirty = None
    return {
        "code_revision": revision,
        "tracked_worktree_dirty": tracked_dirty,
        "worktree_dirty_including_untracked": worktree_dirty,
    }


def _mean(values: list[float | None]) -> float:
    scored = [value for value in values if value is not None]
    return statistics.fmean(scored)


def run_evaluation(config: AppConfig, *, queries_path: Path, output_path: Path) -> dict[str, Any]:
    manifest = _load_json(config.corpus_manifest_path)
    if not isinstance(manifest, list):
        raise EvaluationError("Corpus manifest must contain a JSON array.")
    manifest_ids = {item["paper_id"] for item in manifest}
    queries = load_evaluation_queries(queries_path, manifest_paper_ids=manifest_ids)

    service = RetrievalService(config)
    service.initialize()
    active = service.active_index
    if active is None:
        raise EvaluationError("Active index identity is unavailable.")
    index_manifest = _load_json(config.chroma_path / "index_manifest.json")
    corpus_hash = _sha256_file(config.corpus_manifest_path)
    prepared = _prepared_statistics(config)
    if index_manifest["corpus_manifest_sha256"] != corpus_hash:
        raise EvaluationError(
            "Active index corpus manifest hash does not match the fixed manifest."
        )
    if index_manifest["chunks_jsonl_sha256"] != prepared["chunks_jsonl_sha256"]:
        raise EvaluationError("Active index chunks hash does not match prepared chunks.")

    per_query = []
    evaluation_started = time.perf_counter()
    for query in queries:
        started = time.perf_counter()
        chunk_results = service.retrieve(query.query, top_k=CANDIDATE_CHUNK_COUNT)
        latency_ms = (time.perf_counter() - started) * 1000
        paper_rankings = []
        seen_papers = set()
        for chunk in chunk_results:
            if chunk.paper_id in seen_papers:
                continue
            seen_papers.add(chunk.paper_id)
            paper_rankings.append(
                {
                    "rank": len(paper_rankings) + 1,
                    "paper_id": chunk.paper_id,
                    "title": chunk.title,
                    "best_chunk_id": chunk.chunk_id,
                    "page": chunk.page,
                    "distance": chunk.distance,
                    "chunk_rank": chunk.rank,
                }
            )
            if len(paper_rankings) == EVALUATION_TOP_K:
                break
        ranking_ids = [item["paper_id"] for item in paper_rankings]
        relevant = set(query.relevant_paper_ids)
        per_query.append(
            {
                **asdict(query),
                "rankings": paper_rankings,
                "metrics": {
                    "recall_at_5": recall_at_k(ranking_ids, relevant, k=EVALUATION_TOP_K),
                    "mrr_at_5": reciprocal_rank_at_k(ranking_ids, relevant, k=EVALUATION_TOP_K),
                    "ndcg_at_5": ndcg_at_k(ranking_ids, relevant, k=EVALUATION_TOP_K),
                },
                "retrieval_latency_ms": latency_ms,
            }
        )
    total_duration = time.perf_counter() - evaluation_started
    scored_query_count = sum(bool(query.relevant_paper_ids) for query in queries)
    result = {
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "query_count": len(queries),
        "scored_query_count": scored_query_count,
        "judgment_method": (
            "Paper-level relevance was manually curated from manifest titles and sidecar "
            "abstracts; papers were not independently read in full for relevance judging."
        ),
        "ranking_method": (
            "Retrieve 20 chunks, preserve Chroma order, deduplicate by first paper occurrence, "
            "and score the first 5 unique papers."
        ),
        "aggregate_metrics": {
            "recall_at_5": _mean([item["metrics"]["recall_at_5"] for item in per_query]),
            "mrr_at_5": _mean([item["metrics"]["mrr_at_5"] for item in per_query]),
            "ndcg_at_5": _mean([item["metrics"]["ndcg_at_5"] for item in per_query]),
        },
        "corpus_manifest_sha256": corpus_hash,
        "index_identity": active.index_identity,
        "embedding_model": active.embedding_model,
        "embedding_dimension": active.embedding_dimension,
        "paper_count": active.paper_count,
        "chunk_count": active.chunk_count,
        "verified_ingestion_index_statistics": {
            "selected_papers": len(manifest),
            "parsed_papers": prepared["parsed_papers"],
            "pages": prepared["pages"],
            "chunks": prepared["chunks"],
            "embedding_dimension": active.embedding_dimension,
            "index_build_duration_seconds": None,
            "index_build_duration_note": (
                "Not persisted in the active index manifest; no value is claimed."
            ),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "candidate_chunk_count": CANDIDATE_CHUNK_COUNT,
            "paper_ranking_cutoff": EVALUATION_TOP_K,
            "evaluation_duration_seconds": total_duration,
            **_git_runtime(config.project_root),
        },
        "queries": per_query,
    }
    try:
        atomic_write_json(output_path, result)
    except ReportError as error:
        raise EvaluationError(str(error)) from error
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=Path, default=Path(__file__).resolve().parent / "queries.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = AppConfig.load()
        queries_path = (
            args.queries if args.queries.is_absolute() else config.project_root / args.queries
        )
        output_path = (
            args.output if args.output.is_absolute() else config.project_root / args.output
        )
        result = run_evaluation(
            config,
            queries_path=queries_path,
            output_path=output_path,
        )
        print(
            f"Evaluated {result['query_count']} queries: "
            f"Recall@5={result['aggregate_metrics']['recall_at_5']:.6f}, "
            f"MRR@5={result['aggregate_metrics']['mrr_at_5']:.6f}, "
            f"nDCG@5={result['aggregate_metrics']['ndcg_at_5']:.6f}"
        )
    except (ConfigurationError, RetrievalError, EvaluationError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
