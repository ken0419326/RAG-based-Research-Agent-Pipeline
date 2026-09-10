"""Manifest-driven PDF parsing and deterministic character chunking."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from corpus import MANIFEST_FIELDS, MAX_YEAR, MIN_YEAR, TARGET_COUNT, ManifestEntry

CHUNK_SIZE = 750
CHUNK_OVERLAP = 100
PREPARED_FILENAME = "chunks.jsonl"


class PreparationError(RuntimeError):
    """Raised when canonical inputs cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    """One independently chunked PDF page with complete source provenance."""

    chunk_id: str
    paper_id: str
    title: str
    year: int
    venue: str
    url: str
    page_number: int
    chunk_index: int
    pdf_sha256: str
    chunk_sha256: str
    abstract: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    selected_papers: int
    parsed_papers: int
    failed_papers: int
    empty_papers: int
    total_pages: int
    total_chunks: int
    output_path: Path


def load_canonical_manifest(
    manifest_path: Path, *, expected_count: int = TARGET_COUNT
) -> tuple[ManifestEntry, ...]:
    """Load the canonical manifest without accepting sidecar identity fields."""
    try:
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"Cannot read corpus manifest {manifest_path}: {error}") from error

    if not isinstance(content, list) or len(content) != expected_count:
        raise PreparationError(
            f"Corpus manifest must contain exactly {expected_count} selected papers."
        )

    entries = []
    for item in content:
        if not isinstance(item, dict) or set(item) != MANIFEST_FIELDS:
            raise PreparationError("Corpus manifest entry has missing or unsupported fields.")
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
            raise PreparationError("Corpus manifest entry contains invalid values.") from error
        entries.append(entry)

    paper_ids = [entry.paper_id for entry in entries]
    ranks = [entry.selection_rank for entry in entries]
    if len(set(paper_ids)) != expected_count:
        raise PreparationError("Corpus manifest paper IDs must be unique.")
    if ranks != list(range(1, expected_count + 1)):
        raise PreparationError("Corpus manifest ranks must be consecutive and ordered.")
    if any(not MIN_YEAR <= entry.year <= MAX_YEAR for entry in entries):
        raise PreparationError("Corpus manifest contains a paper outside 2025–2026.")
    return tuple(entries)


def chunk_page_text(
    text: str, *, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> tuple[str, ...]:
    """Split one page independently using deterministic character windows."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than it.")
    page_text = text.strip()
    if not page_text:
        return ()

    chunks = []
    start = 0
    while start < len(page_text):
        end = min(start + chunk_size, len(page_text))
        chunks.append(page_text[start:end])
        if end == len(page_text):
            break
        start = end - chunk_overlap
    return tuple(chunks)


def sanitize_unicode(text: str) -> str:
    """Replace invalid UTF-16 surrogate code points with the Unicode replacement character."""
    return "".join(
        "\N{REPLACEMENT CHARACTER}" if 0xD800 <= ord(character) <= 0xDFFF else character
        for character in text
    )


def deterministic_chunk_id(
    *, paper_id: str, pdf_sha256: str, page_number: int, chunk_index: int, text: str
) -> tuple[str, str]:
    """Return a persistent chunk ID and the exact chunk-text SHA-256."""
    chunk_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = "\0".join((paper_id, pdf_sha256, str(page_number), str(chunk_index), chunk_sha256))
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return chunk_id, chunk_sha256


def extract_pdf_pages(
    pdf_path: Path, *, reader_factory: Callable[[Path], Any] | None = None
) -> tuple[str, ...]:
    """Extract every PDF page in order with pypdf."""
    if reader_factory is None:
        from pypdf import PdfReader

        reader_factory = PdfReader
    reader = reader_factory(pdf_path)
    return tuple(sanitize_unicode(page.extract_text() or "") for page in reader.pages)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_abstract(sidecar_path: Path, paper_id: str) -> str:
    if not sidecar_path.exists():
        return ""
    try:
        content = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"Invalid JSON sidecar {sidecar_path}: {error}") from error
    if not isinstance(content, dict) or content.get("paper_id") != paper_id:
        raise PreparationError(f"JSON sidecar {sidecar_path} does not belong to paper {paper_id}.")
    abstract = content.get("abstract", "")
    if not isinstance(abstract, str):
        raise PreparationError(f"JSON sidecar {sidecar_path} has a non-text abstract.")
    return sanitize_unicode(abstract)


def _resolve_pdf_path(entry: ManifestEntry, *, project_root: Path, raw_dir: Path) -> Path:
    configured = Path(entry.local_pdf_path)
    path = configured if configured.is_absolute() else project_root / configured
    try:
        if path.resolve().parent != raw_dir.resolve():
            raise PreparationError(
                f"Manifest PDF for {entry.paper_id} is outside raw-data directory: {path}"
            )
    except OSError as error:
        raise PreparationError(f"Cannot resolve manifest PDF path {path}: {error}") from error
    return path


def build_page_chunks(
    entry: ManifestEntry, *, page_number: int, page_text: str, abstract: str
) -> tuple[PreparedChunk, ...]:
    if not isinstance(entry.sha256, str):
        raise PreparationError(f"Manifest paper {entry.paper_id} has no PDF SHA-256.")
    chunks = []
    for chunk_index, text in enumerate(chunk_page_text(page_text)):
        chunk_id, chunk_sha256 = deterministic_chunk_id(
            paper_id=entry.paper_id,
            pdf_sha256=entry.sha256,
            page_number=page_number,
            chunk_index=chunk_index,
            text=text,
        )
        chunks.append(
            PreparedChunk(
                chunk_id=chunk_id,
                paper_id=entry.paper_id,
                title=sanitize_unicode(entry.title),
                year=entry.year,
                venue=sanitize_unicode(entry.venue),
                url=sanitize_unicode(entry.pdf_url),
                page_number=page_number,
                chunk_index=chunk_index,
                pdf_sha256=entry.sha256,
                chunk_sha256=chunk_sha256,
                abstract=sanitize_unicode(abstract),
                text=text,
            )
        )
    return tuple(chunks)


def prepare_corpus(
    *,
    manifest_path: Path,
    project_root: Path,
    raw_dir: Path,
    output_path: Path,
    expected_count: int = TARGET_COUNT,
    reader_factory: Callable[[Path], Any] | None = None,
    report: Callable[[str], None] = print,
) -> PreparationSummary:
    """Prepare manifest-selected PDFs and atomically publish JSONL chunks."""
    entries = load_canonical_manifest(manifest_path, expected_count=expected_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    parsed_papers = 0
    failed_papers = 0
    empty_papers = 0
    total_pages = 0
    total_chunks = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for entry in entries:
                try:
                    if entry.download_status != "valid":
                        raise PreparationError(
                            f"manifest download status is {entry.download_status!r}"
                        )
                    pdf_path = _resolve_pdf_path(entry, project_root=project_root, raw_dir=raw_dir)
                    if not pdf_path.is_file():
                        raise PreparationError(f"PDF is missing: {pdf_path}")
                    actual_hash = _sha256_file(pdf_path)
                    if actual_hash != entry.sha256:
                        raise PreparationError(
                            f"PDF SHA-256 mismatch for {entry.paper_id}: {pdf_path}"
                        )
                    abstract = _load_abstract(pdf_path.with_suffix(".json"), entry.paper_id)
                    pages = extract_pdf_pages(pdf_path, reader_factory=reader_factory)
                    total_pages += len(pages)
                    paper_chunks = []
                    for page_number, page_text in enumerate(pages, start=1):
                        paper_chunks.extend(
                            build_page_chunks(
                                entry,
                                page_number=page_number,
                                page_text=page_text,
                                abstract=abstract,
                            )
                        )
                    serialized_chunks = [
                        json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n"
                        for chunk in paper_chunks
                    ]
                except Exception as error:
                    failed_papers += 1
                    report(f"Failed to prepare {entry.paper_id}: {error}")
                    continue

                if not paper_chunks:
                    empty_papers += 1
                    report(f"Empty PDF extraction for {entry.paper_id}: {pdf_path}")
                    continue
                for serialized_chunk in serialized_chunks:
                    stream.write(serialized_chunk)
                parsed_papers += 1
                total_chunks += len(paper_chunks)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return PreparationSummary(
        selected_papers=len(entries),
        parsed_papers=parsed_papers,
        failed_papers=failed_papers,
        empty_papers=empty_papers,
        total_pages=total_pages,
        total_chunks=total_chunks,
        output_path=output_path,
    )
