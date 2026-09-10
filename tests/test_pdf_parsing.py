from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from pypdf import PdfWriter

import data_update
import documents
from config import AppConfig
from data_update import DataUpdatePipeline
from documents import extract_pdf_pages, prepare_corpus, sanitize_unicode


class Page:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class Reader:
    def __init__(self, pages: list[Page]) -> None:
        self.pages = pages


def manifest_item(
    paper_id: str,
    pdf_path: Path,
    content: bytes,
    *,
    rank: int,
    project_root: Path,
) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Manifest title {paper_id}",
        "year": 2025 if rank % 2 else 2026,
        "venue": "Manifest Venue",
        "pdf_url": f"https://aclanthology.org/{paper_id}.pdf",
        "matched_keywords": ["empathy"],
        "selection_score": 2,
        "selection_rank": rank,
        "download_status": "valid",
        "local_pdf_path": pdf_path.relative_to(project_root).as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


def load_config() -> AppConfig:
    return AppConfig.load({}, load_env_file=False)


def test_manifest_metadata_and_validated_sidecar_abstract_propagate(tmp_path):
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    pdf_path = raw_dir / "2025.test.1.pdf"
    pdf_content = b"%PDF-test fixture"
    pdf_path.write_bytes(pdf_content)
    pdf_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "paper_id": "2025.test.1",
                "title": "Untrusted sidecar title",
                "year": 1999,
                "abstract": "Structured abstract",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "corpus/manifest.json"
    write_manifest(
        manifest_path,
        [manifest_item("2025.test.1", pdf_path, pdf_content, rank=1, project_root=tmp_path)],
    )
    output_path = tmp_path / "data/processed/chunks.jsonl"

    summary = prepare_corpus(
        manifest_path=manifest_path,
        project_root=tmp_path,
        raw_dir=raw_dir,
        output_path=output_path,
        expected_count=1,
        reader_factory=lambda path: Reader([Page("page body")]),
    )
    chunk = json.loads(output_path.read_text())

    assert summary.parsed_papers == 1
    assert chunk["paper_id"] == "2025.test.1"
    assert chunk["title"] == "Manifest title 2025.test.1"
    assert chunk["year"] == 2025
    assert chunk["venue"] == "Manifest Venue"
    assert chunk["url"] == "https://aclanthology.org/2025.test.1.pdf"
    assert chunk["page_number"] == 1
    assert chunk["chunk_index"] == 0
    assert chunk["pdf_sha256"] == hashlib.sha256(pdf_content).hexdigest()
    assert chunk["abstract"] == "Structured abstract"
    assert chunk["text"] == "page body"


def test_mismatched_sidecar_identity_is_rejected(tmp_path):
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    pdf_path = raw_dir / "2025.test.1.pdf"
    pdf_content = b"%PDF-test fixture"
    pdf_path.write_bytes(pdf_content)
    pdf_path.with_suffix(".json").write_text(
        json.dumps({"paper_id": "2025.test.other", "abstract": "Wrong paper"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "corpus/manifest.json"
    write_manifest(
        manifest_path,
        [manifest_item("2025.test.1", pdf_path, pdf_content, rank=1, project_root=tmp_path)],
    )
    messages = []

    summary = prepare_corpus(
        manifest_path=manifest_path,
        project_root=tmp_path,
        raw_dir=raw_dir,
        output_path=tmp_path / "data/processed/chunks.jsonl",
        expected_count=1,
        reader_factory=lambda path: Reader([Page("page body")]),
        report=messages.append,
    )

    assert summary.failed_papers == 1
    assert summary.parsed_papers == 0
    assert "does not belong" in messages[0]


def test_missing_corrupt_and_empty_pdfs_are_reported(tmp_path):
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    missing_path = raw_dir / "2025.test.missing.pdf"
    corrupt_path = raw_dir / "2025.test.corrupt.pdf"
    corrupt_content = b"%PDF-corrupt"
    corrupt_path.write_bytes(corrupt_content)
    empty_path = raw_dir / "2025.test.empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with empty_path.open("wb") as stream:
        writer.write(stream)
    empty_content = empty_path.read_bytes()
    manifest_path = tmp_path / "corpus/manifest.json"
    write_manifest(
        manifest_path,
        [
            manifest_item(
                "2025.test.missing", missing_path, b"missing", rank=1, project_root=tmp_path
            ),
            manifest_item(
                "2025.test.corrupt",
                corrupt_path,
                corrupt_content,
                rank=2,
                project_root=tmp_path,
            ),
            manifest_item(
                "2025.test.empty", empty_path, empty_content, rank=3, project_root=tmp_path
            ),
        ],
    )
    messages = []

    summary = prepare_corpus(
        manifest_path=manifest_path,
        project_root=tmp_path,
        raw_dir=raw_dir,
        output_path=tmp_path / "data/processed/chunks.jsonl",
        expected_count=3,
        report=messages.append,
    )

    assert summary.parsed_papers == 0
    assert summary.failed_papers == 2
    assert summary.empty_papers == 1
    assert summary.total_pages == 1
    assert summary.total_chunks == 0
    assert any("2025.test.missing" in message for message in messages)
    assert any("2025.test.corrupt" in message for message in messages)
    assert any("2025.test.empty" in message for message in messages)


def test_prepare_only_writes_jsonl_without_initializing_model_or_chroma(monkeypatch, tmp_path):
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    entries = []
    for rank in range(1, 51):
        paper_id = f"2025.test.{rank:02}"
        pdf_path = raw_dir / f"{paper_id}.pdf"
        content = f"%PDF-fixture-{rank}".encode()
        pdf_path.write_bytes(content)
        entries.append(manifest_item(paper_id, pdf_path, content, rank=rank, project_root=tmp_path))
    manifest_path = tmp_path / "corpus/manifest.json"
    write_manifest(manifest_path, entries)
    config = replace(
        load_config(),
        project_root=tmp_path,
        raw_dir=raw_dir,
        processed_dir=tmp_path / "data/processed",
        corpus_manifest_path=manifest_path,
    )
    pipeline = DataUpdatePipeline(config=config)
    monkeypatch.setattr(documents, "extract_pdf_pages", lambda path, reader_factory=None: ("x",))

    summary = pipeline.prepare_only()

    assert summary.selected_papers == 50
    assert summary.parsed_papers == 50
    assert summary.total_chunks == 50
    assert len((config.processed_dir / "chunks.jsonl").read_text().splitlines()) == 50
    assert pipeline.model is None
    assert pipeline.db_client is None
    assert pipeline.collection is None

    monkeypatch.setattr(data_update, "DataUpdatePipeline", lambda: pipeline)
    assert data_update.main(["--prepare-only"]) == 0
    assert pipeline.model is None
    assert pipeline.db_client is None
    assert pipeline.collection is None


def test_extract_pdf_pages_uses_pypdf_for_empty_page(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with pdf_path.open("wb") as stream:
        writer.write(stream)

    assert extract_pdf_pages(pdf_path) == ("",)


def test_invalid_pdf_text_surrogates_are_replaced_deterministically():
    assert sanitize_unicode("before\ud800after") == "before\N{REPLACEMENT CHARACTER}after"
