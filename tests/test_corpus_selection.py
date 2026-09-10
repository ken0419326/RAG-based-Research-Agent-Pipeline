from __future__ import annotations

import random

import pytest

from corpus import (
    InsufficientMatchesError,
    ManifestEntry,
    PaperRecord,
    rank_matching_papers,
    score_paper,
    select_fixed_corpus,
    selected_from_manifest,
)


def paper(
    paper_id: str,
    *,
    year: int = 2025,
    title: str = "Empathy in language models",
    abstract: str = "",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        year=year,
        venue="test-venue",
        pdf_url=f"https://aclanthology.org/{paper_id}.pdf",
        abstract=abstract,
    )


def test_filters_to_2025_2026_and_specific_keywords():
    records = [
        paper("2024.test.1", year=2024),
        paper("2025.test.1", year=2025, title="EMOTION-RECOGNITION benchmark"),
        paper("2026.test.1", year=2026, title="No keyword", abstract="Human values matter"),
        paper("2027.test.1", year=2027),
        paper("2025.test.2", title="The value of dialogue", abstract="Sentiment analysis"),
        paper("2025.test.3", title="Alignment methods", abstract="No relevant phrase"),
    ]

    ranked = rank_matching_papers(records)

    assert {candidate.paper.paper_id for candidate in ranked} == {
        "2025.test.1",
        "2026.test.1",
    }


def test_score_and_order_are_deterministic():
    records = [
        paper(
            "2025.test.b",
            title="Empathy and emotional support",
            abstract="Empathy supports emotion recognition.",
        ),
        paper("2026.test.c", title="Empathy", abstract="Emotion recognition"),
        paper("2026.test.a", title="Empathy", abstract="Emotion recognition"),
    ]
    first_score, first_matches = score_paper(records[0])

    assert first_score == 6
    assert first_matches == ("empathy", "emotional support", "emotion recognition")
    assert [item.paper.paper_id for item in rank_matching_papers(records)] == [
        "2025.test.b",
        "2026.test.a",
        "2026.test.c",
    ]

    shuffled = records.copy()
    random.Random(7).shuffle(shuffled)
    assert rank_matching_papers(shuffled) == rank_matching_papers(records)


def test_selects_exactly_50_unique_paper_ids():
    records = [paper(f"2025.test.{number:03}") for number in range(55)]
    records.extend(
        [
            paper("2025.test.001", abstract="emotion recognition"),
            paper("2025.test.002", abstract="emotion understanding"),
        ]
    )

    result = select_fixed_corpus(records)

    assert len(result.matched) == 55
    assert len(result.selected) == 50
    assert len({item.paper.paper_id for item in result.selected}) == 50
    assert [item.selection_rank for item in result.selected] == list(range(1, 51))


def test_fewer_than_50_matches_stops_selection():
    records = [paper(f"2025.test.{number:03}") for number in range(49)]

    with pytest.raises(InsufficientMatchesError) as error:
        select_fixed_corpus(records)

    assert error.value.matched_count == 49
    assert error.value.required_count == 50


def test_existing_manifest_keeps_its_fixed_selection():
    records = [paper(f"2025.test.{number:03}") for number in range(50)]
    selected = select_fixed_corpus(records).selected
    manifest = [
        ManifestEntry(
            paper_id=item.paper.paper_id,
            title=item.paper.title,
            year=item.paper.year,
            venue=item.paper.venue,
            pdf_url=item.paper.pdf_url,
            matched_keywords=item.matched_keywords,
            selection_score=item.selection_score,
            selection_rank=item.selection_rank,
            download_status="valid",
            local_pdf_path=f"data/raw/{item.paper.paper_id}.pdf",
            sha256="a" * 64,
        ).to_dict()
        for item in selected
    ]
    changed_catalog = records + [
        paper(
            "2026.test.new",
            year=2026,
            title="Empathy emotional support emotion recognition",
        )
    ]

    restored = selected_from_manifest(manifest, changed_catalog)

    assert [item.paper.paper_id for item in restored] == [item.paper.paper_id for item in selected]
