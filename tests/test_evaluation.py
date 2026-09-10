from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from eval.run_retrieval import EvaluationError, load_evaluation_queries


def test_retrieval_metrics_use_unique_paper_rankings():
    ranking = ["paper-a", "paper-a", "paper-x", "paper-b"]
    relevant = {"paper-a", "paper-b"}

    assert recall_at_k(ranking, relevant, k=3) == 1.0
    assert reciprocal_rank_at_k(ranking, relevant, k=3) == 1.0
    expected_dcg = 1.0 + 1.0 / math.log2(4)
    expected_ideal = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(ranking, relevant, k=3) == pytest.approx(expected_dcg / expected_ideal)


def test_metrics_return_none_for_unanswerable_query():
    assert recall_at_k(["paper-a"], set(), k=5) is None
    assert reciprocal_rank_at_k(["paper-a"], set(), k=5) is None
    assert ndcg_at_k(["paper-a"], set(), k=5) is None


def test_evaluation_file_rejects_unknown_manifest_paper(tmp_path):
    path = tmp_path / "queries.jsonl"
    base = {
        "query": "query",
        "language": "en",
        "judgment_basis": "manually_curated_from_titles_and_abstracts",
        "judgment_note": "manual fixture",
    }
    categories = [
        "cross_language",
        "english_topic",
        "exact_lookup",
        "multiple_relevant",
        "unanswerable",
    ]
    rows = []
    for index in range(10):
        category = categories[index % len(categories)]
        relevant = [] if category == "unanswerable" else ["known-paper"]
        if category == "multiple_relevant":
            relevant = ["known-paper", "unknown-paper"]
        rows.append(
            {
                **base,
                "query_id": f"q{index}",
                "category": category,
                "relevant_paper_ids": relevant,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(EvaluationError, match="unknown paper IDs"):
        load_evaluation_queries(path, manifest_paper_ids={"known-paper"})


def test_release_evaluation_file_is_valid():
    manifest = json.loads(Path("corpus/manifest.json").read_text(encoding="utf-8"))
    manifest_ids = {item["paper_id"] for item in manifest}

    queries = load_evaluation_queries(
        Path("eval/queries.jsonl"),
        manifest_paper_ids=manifest_ids,
    )

    assert len(queries) == 12
    assert all(set(query.relevant_paper_ids) <= manifest_ids for query in queries)
