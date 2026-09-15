"""Deterministic ACL corpus selection and manifest models."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from typing import Any

MIN_YEAR = 2025
MAX_YEAR = 2026
TARGET_COUNT = 50

KEYWORDS = (
    "empathy",
    "empathetic",
    "empathic",
    "emotional support",
    "emotion recognition",
    "emotion understanding",
    "emotion classification",
    "affect recognition",
    "multimodal emotion",
    "value alignment",
    "human values",
    "AI alignment",
    "LLM alignment",
    "language model alignment",
)


class InsufficientMatchesError(ValueError):
    """Raised when the fixed-size corpus cannot be selected without broadening scope."""

    def __init__(self, matched_count: int, required_count: int = TARGET_COUNT) -> None:
        self.matched_count = matched_count
        self.required_count = required_count
        super().__init__(
            f"Only {matched_count} unique papers matched; {required_count} are required."
        )


@dataclass(frozen=True, slots=True)
class PaperRecord:
    """Catalog fields needed for selection and acquisition."""

    paper_id: str
    title: str
    year: int
    venue: str
    pdf_url: str
    abstract: str


@dataclass(frozen=True, slots=True)
class RankedPaper:
    """A catalog paper with deterministic selection evidence."""

    paper: PaperRecord
    matched_keywords: tuple[str, ...]
    selection_score: int
    selection_rank: int = 0


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """The intentionally small, tracked representation of a selected paper."""

    paper_id: str
    title: str
    year: int
    venue: str
    pdf_url: str
    matched_keywords: tuple[str, ...]
    selection_score: int
    selection_rank: int
    download_status: str
    local_pdf_path: str
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Matched candidates and the fixed-size selected prefix."""

    matched: tuple[RankedPaper, ...]
    selected: tuple[RankedPaper, ...]


MANIFEST_FIELDS = frozenset(field.name for field in fields(ManifestEntry))


def normalize_text(text: str) -> str:
    """Normalize case, Unicode, punctuation, and whitespace for phrase matching."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def keyword_matches(text: str) -> tuple[str, ...]:
    """Return each configured keyword at most once, in configured order."""
    padded_text = f" {normalize_text(text)} "
    return tuple(keyword for keyword in KEYWORDS if f" {normalize_text(keyword)} " in padded_text)


def score_paper(paper: PaperRecord) -> tuple[int, tuple[str, ...]]:
    """Score one paper as 2 × title matches + abstract matches."""
    title_matches = keyword_matches(paper.title)
    abstract_matches = keyword_matches(paper.abstract)
    all_matches = tuple(
        keyword for keyword in KEYWORDS if keyword in title_matches or keyword in abstract_matches
    )
    return 2 * len(title_matches) + len(abstract_matches), all_matches


def _deduplicate(records: Iterable[PaperRecord]) -> dict[str, PaperRecord]:
    """Choose deterministically if malformed catalog input repeats an ID."""
    unique: dict[str, PaperRecord] = {}
    for record in records:
        current = unique.get(record.paper_id)
        if current is None:
            unique[record.paper_id] = record
            continue
        current_score, _ = score_paper(current)
        new_score, _ = score_paper(record)
        current_key = (
            current_score,
            current.year,
            current.title,
            current.venue,
            current.pdf_url,
            current.abstract,
        )
        new_key = (
            new_score,
            record.year,
            record.title,
            record.venue,
            record.pdf_url,
            record.abstract,
        )
        if new_key > current_key:
            unique[record.paper_id] = record
    return unique


def rank_matching_papers(records: Iterable[PaperRecord]) -> tuple[RankedPaper, ...]:
    """Filter and rank unique papers by score, year, then Anthology ID."""
    ranked = []
    for paper in _deduplicate(records).values():
        if not MIN_YEAR <= paper.year <= MAX_YEAR:
            continue
        score, matches = score_paper(paper)
        if score:
            ranked.append(RankedPaper(paper, matches, score))

    ranked.sort(
        key=lambda candidate: (
            -candidate.selection_score,
            -candidate.paper.year,
            candidate.paper.paper_id,
        )
    )
    return tuple(ranked)


def select_fixed_corpus(
    records: Iterable[PaperRecord], target_count: int = TARGET_COUNT
) -> SelectionResult:
    """Select exactly ``target_count`` papers, or fail without broadening criteria."""
    matched = rank_matching_papers(records)
    if len(matched) < target_count:
        raise InsufficientMatchesError(len(matched), target_count)
    selected = tuple(
        RankedPaper(
            paper=candidate.paper,
            matched_keywords=candidate.matched_keywords,
            selection_score=candidate.selection_score,
            selection_rank=rank,
        )
        for rank, candidate in enumerate(matched[:target_count], start=1)
    )
    return SelectionResult(matched=matched, selected=selected)


def selected_from_manifest(
    content: Any, catalog_records: Iterable[PaperRecord]
) -> tuple[RankedPaper, ...]:
    """Validate and restore the fixed selection while enriching it with current abstracts."""
    if not isinstance(content, list) or len(content) != TARGET_COUNT:
        raise ValueError(f"Manifest must contain exactly {TARGET_COUNT} entries.")

    abstracts = {record.paper_id: record.abstract for record in catalog_records}
    selected = []
    for item in content:
        if not isinstance(item, dict) or set(item) != MANIFEST_FIELDS:
            raise ValueError("Manifest entry has missing or unsupported fields.")
        try:
            entry = ManifestEntry(
                paper_id=str(item["paper_id"]),
                title=str(item["title"]),
                year=int(item["year"]),
                venue=str(item["venue"]),
                pdf_url=str(item["pdf_url"]),
                matched_keywords=tuple(item["matched_keywords"]),
                selection_score=int(item["selection_score"]),
                selection_rank=int(item["selection_rank"]),
                download_status=str(item["download_status"]),
                local_pdf_path=str(item["local_pdf_path"]),
                sha256=item["sha256"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Manifest entry contains invalid values.") from error
        if not MIN_YEAR <= entry.year <= MAX_YEAR:
            raise ValueError(f"Manifest paper {entry.paper_id} is outside 2025–2026.")
        if not entry.matched_keywords or any(
            keyword not in KEYWORDS for keyword in entry.matched_keywords
        ):
            raise ValueError(f"Manifest paper {entry.paper_id} has invalid matched keywords.")
        valid_hash = isinstance(entry.sha256, str) and re.fullmatch(r"[0-9a-f]{64}", entry.sha256)
        if entry.download_status == "valid" and not valid_hash:
            raise ValueError(f"Manifest paper {entry.paper_id} has no valid SHA-256.")
        if entry.download_status == "failed" and entry.sha256 is not None:
            raise ValueError(f"Failed manifest paper {entry.paper_id} cannot have a SHA-256.")
        if entry.download_status not in {"valid", "failed"}:
            raise ValueError(f"Manifest paper {entry.paper_id} has an invalid download status.")
        paper = PaperRecord(
            paper_id=entry.paper_id,
            title=entry.title,
            year=entry.year,
            venue=entry.venue,
            pdf_url=entry.pdf_url,
            abstract=abstracts.get(entry.paper_id, ""),
        )
        selected.append(
            RankedPaper(
                paper=paper,
                matched_keywords=entry.matched_keywords,
                selection_score=entry.selection_score,
                selection_rank=entry.selection_rank,
            )
        )

    paper_ids = [candidate.paper.paper_id for candidate in selected]
    ranks = [candidate.selection_rank for candidate in selected]
    if len(set(paper_ids)) != TARGET_COUNT:
        raise ValueError("Manifest paper IDs must be unique.")
    if ranks != list(range(1, TARGET_COUNT + 1)):
        raise ValueError("Manifest selection ranks must be ordered from 1 through 50.")
    return tuple(selected)
