from __future__ import annotations

import hashlib
import json

from corpus import PaperRecord, select_fixed_corpus
from downloader import download_selected_papers


class Response:
    def __init__(self, content: bytes, *, failure: Exception | None = None) -> None:
        self.content = content
        self.failure = failure

    def raise_for_status(self) -> None:
        if self.failure:
            raise self.failure


def make_records() -> list[PaperRecord]:
    return [
        PaperRecord(
            paper_id=f"2025.test.{number:03}",
            title="Empathy",
            year=2025,
            venue="test-venue",
            pdf_url=f"https://aclanthology.org/2025.test.{number:03}.pdf",
            abstract="",
        )
        for number in range(50)
    ]


def test_download_counts_validation_and_manifest_output(tmp_path):
    selected = select_fixed_corpus(make_records()).selected
    raw_dir = tmp_path / "data/raw"
    manifest_path = tmp_path / "corpus/manifest.json"
    existing = raw_dir / f"{selected[0].paper.paper_id}.pdf"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"%PDF-existing")
    failed_id = selected[-1].paper.paper_id
    invalid_id = selected[-2].paper.paper_id
    calls = []

    def fake_get(url: str, *, timeout: float):
        calls.append((url, timeout))
        if failed_id in url:
            return Response(b"", failure=RuntimeError("HTTP 404"))
        if invalid_id in url:
            return Response(b"<html>not a PDF</html>")
        return Response(b"%PDF-mocked")

    summary = download_selected_papers(
        selected,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        project_root=tmp_path,
        http_get=fake_get,
    )
    manifest = json.loads(manifest_path.read_text())

    assert summary.selected == 50
    assert summary.downloaded == 47
    assert summary.already_existing == 1
    assert summary.failed == 2
    assert len(calls) == 49
    assert all(timeout == 30.0 for _, timeout in calls)
    assert len(manifest) == 50
    assert len({entry["paper_id"] for entry in manifest}) == 50
    assert {entry["download_status"] for entry in manifest} == {"valid", "failed"}
    assert next(entry for entry in manifest if entry["paper_id"] == failed_id)["sha256"] is None
    assert not (raw_dir / f"{failed_id}.pdf").exists()
    assert not (raw_dir / f"{invalid_id}.pdf").exists()
    assert manifest[0]["sha256"] == hashlib.sha256(b"%PDF-existing").hexdigest()
    assert set(manifest[0]) == {
        "paper_id",
        "title",
        "year",
        "venue",
        "pdf_url",
        "matched_keywords",
        "selection_score",
        "selection_rank",
        "download_status",
        "local_pdf_path",
        "sha256",
    }

    first_manifest = manifest_path.read_bytes()
    repeated = download_selected_papers(
        selected,
        raw_dir=raw_dir,
        manifest_path=manifest_path,
        project_root=tmp_path,
        http_get=fake_get,
    )

    assert repeated.downloaded == 0
    assert repeated.already_existing == 48
    assert repeated.failed == 2
    assert manifest_path.read_bytes() == first_manifest
