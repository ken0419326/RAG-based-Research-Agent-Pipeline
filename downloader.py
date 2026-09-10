"""Select and safely download the fixed ACL paper corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from config import AppConfig
from corpus import (
    TARGET_COUNT,
    InsufficientMatchesError,
    ManifestEntry,
    PaperRecord,
    RankedPaper,
    rank_matching_papers,
    select_fixed_corpus,
    selected_from_manifest,
)

DOWNLOAD_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    selected: int
    downloaded: int
    already_existing: int
    failed: int


@dataclass(frozen=True, slots=True)
class AcquisitionSummary:
    matched: int
    selected: int
    year_counts: dict[int, int]
    downloaded: int
    already_existing: int
    failed: int
    manifest_path: Path
    pdf_dir: Path


def paper_record_from_anthology(paper: Any) -> PaperRecord:
    """Adapt one ``acl_anthology`` paper without leaking library types downstream."""
    paper_id = str(paper.full_id)
    abstract = paper.abstract.as_text() if paper.abstract else ""
    venue = str(paper.parent.venue_acronym or paper.parent.full_id)
    return PaperRecord(
        paper_id=paper_id,
        title=paper.title.as_text(),
        year=int(paper.year),
        venue=venue,
        pdf_url=f"https://aclanthology.org/{paper_id}.pdf",
        abstract=abstract,
    )


def iter_catalog_records(papers: Iterable[Any]) -> Iterable[PaperRecord]:
    """Yield usable records; catalog entries with invalid years are outside the scope."""
    for paper in papers:
        try:
            yield paper_record_from_anthology(paper)
        except (AttributeError, TypeError, ValueError):
            continue


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(5) == b"%PDF-"
    except OSError:
        return False


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, content: Any) -> None:
    encoded = (json.dumps(content, ensure_ascii=False, indent=2) + "\n").encode()
    _atomic_write_bytes(path, encoded)


def _local_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def safe_filename(value: str) -> str:
    """Keep Anthology IDs readable while preventing directory traversal."""
    return "".join(
        character if character.isalnum() or character in ".-_" else "_" for character in value
    )


def _initial_manifest_entry(
    candidate: RankedPaper, pdf_path: Path, project_root: Path
) -> ManifestEntry:
    paper = candidate.paper
    return ManifestEntry(
        paper_id=paper.paper_id,
        title=paper.title,
        year=paper.year,
        venue=paper.venue,
        pdf_url=paper.pdf_url,
        matched_keywords=candidate.matched_keywords,
        selection_score=candidate.selection_score,
        selection_rank=candidate.selection_rank,
        download_status="pending",
        local_pdf_path=_local_path(pdf_path, project_root),
        sha256=None,
    )


def _write_companion_metadata(path: Path, paper: PaperRecord) -> None:
    _atomic_write_json(
        path,
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "year": paper.year,
            "venue": paper.venue,
            "url": paper.pdf_url,
            "abstract": paper.abstract,
        },
    )


def download_selected_papers(
    selected: tuple[RankedPaper, ...],
    *,
    raw_dir: Path,
    manifest_path: Path,
    project_root: Path,
    http_get: Callable[..., Any],
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> DownloadSummary:
    """Download selected PDFs and atomically write their tracked manifest."""
    if len(selected) != TARGET_COUNT:
        raise ValueError(f"Manifest requires exactly {TARGET_COUNT} selected papers.")

    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    downloaded = 0
    already_existing = 0
    failed = 0

    for candidate in selected:
        paper = candidate.paper
        filename = safe_filename(paper.paper_id)
        pdf_path = raw_dir / f"{filename}.pdf"
        json_path = raw_dir / f"{filename}.json"
        entry = _initial_manifest_entry(candidate, pdf_path, project_root)

        if is_valid_pdf(pdf_path):
            already_existing += 1
            entry = replace(entry, download_status="valid", sha256=sha256_file(pdf_path))
            _write_companion_metadata(json_path, paper)
            entries.append(entry)
            continue

        try:
            response = http_get(paper.pdf_url, timeout=timeout)
            response.raise_for_status()
            content = bytes(response.content)
            if not content.startswith(b"%PDF-"):
                raise ValueError("response does not begin with the PDF signature")
            _atomic_write_bytes(pdf_path, content)
            _write_companion_metadata(json_path, paper)
            downloaded += 1
            entry = replace(
                entry,
                download_status="valid",
                sha256=hashlib.sha256(content).hexdigest(),
            )
        except Exception as error:
            failed += 1
            entry = replace(entry, download_status="failed")
            print(f"Download failed for {paper.paper_id}: {error}", file=sys.stderr)
        entries.append(entry)

    _atomic_write_json(manifest_path, [entry.to_dict() for entry in entries])
    return DownloadSummary(
        selected=len(selected),
        downloaded=downloaded,
        already_existing=already_existing,
        failed=failed,
    )


def run_downloader(
    config: AppConfig | None = None,
    *,
    catalog_factory: Callable[[], Any] | None = None,
    http_get: Callable[..., Any] | None = None,
) -> AcquisitionSummary:
    """Discover, rank, select, and download the canonical corpus."""
    config = config or AppConfig.load()

    if catalog_factory is None:
        from acl_anthology import Anthology

        def catalog_factory() -> Any:
            return Anthology.from_repo(path=config.anthology_repo_dir, verbose=False)

    if http_get is None:
        import requests

        http_get = requests.get

    print("Loading ACL Anthology catalog...")
    anthology = catalog_factory()
    records = tuple(iter_catalog_records(anthology.papers()))
    matched = rank_matching_papers(records)
    if len(matched) < TARGET_COUNT:
        raise InsufficientMatchesError(len(matched))
    if config.corpus_manifest_path.is_file():
        manifest_content = json.loads(config.corpus_manifest_path.read_text(encoding="utf-8"))
        selected = selected_from_manifest(manifest_content, records)
    else:
        selected = select_fixed_corpus(records).selected
    year_counts = {
        year: sum(candidate.paper.year == year for candidate in selected) for year in (2025, 2026)
    }
    print(f"Matched: {len(matched)}; selected: {len(selected)}")

    downloads = download_selected_papers(
        selected,
        raw_dir=config.raw_dir,
        manifest_path=config.corpus_manifest_path,
        project_root=config.project_root,
        http_get=http_get,
    )
    summary = AcquisitionSummary(
        matched=len(matched),
        selected=downloads.selected,
        year_counts=year_counts,
        downloaded=downloads.downloaded,
        already_existing=downloads.already_existing,
        failed=downloads.failed,
        manifest_path=config.corpus_manifest_path,
        pdf_dir=config.raw_dir,
    )
    print(f"Year distribution: 2025={year_counts[2025]}, 2026={year_counts[2026]}")
    print(
        "Download counts: "
        f"selected={summary.selected}, downloaded={summary.downloaded}, "
        f"already existing={summary.already_existing}, failed={summary.failed}"
    )
    print(f"Manifest: {summary.manifest_path}")
    print(f"PDF directory: {summary.pdf_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        run_downloader()
    except InsufficientMatchesError as error:
        print(f"Corpus selection stopped: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Corpus acquisition failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
