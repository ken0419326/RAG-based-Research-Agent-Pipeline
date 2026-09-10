from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from config import AppConfig
from generation import GenerationResult
from reporting import ReportError, load_checkpoint, render_source_table
from retrieval import RetrievalResult
from skill_builder import SCAN_QUESTIONS, SkillBuilder


def source() -> RetrievalResult:
    return RetrievalResult(
        rank=1,
        source_id="[S1]",
        distance=0.25,
        chunk_id="verified-chunk",
        paper_id="verified-paper",
        title="Verified | Paper",
        year=2026,
        venue="ACL",
        page=4,
        url="https://example.invalid/verified",
        text="verified passage",
    )


def test_report_source_table_uses_verified_metadata():
    table = render_source_table(
        [
            {
                "question_id": "scope",
                "sources": [
                    {
                        "source_id": "[S1]",
                        "paper_id": "verified-paper",
                        "title": "Verified | Paper",
                        "page": 4,
                        "chunk_id": "verified-chunk",
                        "url": "https://example.invalid/verified",
                    }
                ],
            }
        ]
    )

    assert "verified-paper" in table
    assert "Verified \\| Paper" in table
    assert "verified-chunk" in table
    assert "invented-paper" not in table


def test_checkpoint_index_mismatch_is_rejected(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "index_identity": "old-index",
                "completed_question_ids": [],
                "results": [],
            }
        )
    )

    with pytest.raises(ReportError, match="does not match the active index"):
        load_checkpoint(path, index_identity="new-index", question_ids={"scope"})


def test_skill_builder_writes_checkpoint_and_programmatic_source_table(tmp_path):
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    (chroma_path / "placeholder").write_text("fixture")
    config = replace(
        AppConfig.load({}, load_env_file=False),
        project_root=tmp_path,
        chroma_path=chroma_path,
        checkpoint_path=tmp_path / "checkpoint.json",
        llm_base_url="https://provider.example/v1",
        llm_api_key="fake-key",
        llm_model="fake-model",
    )
    verified_source = source()

    class FakeRetrievalService:
        active_index = SimpleNamespace(
            index_identity="a" * 64,
            embedding_model="test-embedding",
        )
        embedder = object()
        client = object()
        collection = object()

        def initialize(self):
            return None

        def retrieve(self, query, *, top_k):
            assert top_k == 2
            return [verified_source]

    class FakeGenerationService:
        client = object()

        def generate(self, query, sources, *, model_override):
            assert sources == [verified_source]
            return GenerationResult(
                answer="Narrative with an invented-paper phrase, but citation [S1].",
                cited_source_ids=("[S1]",),
                invalid_source_ids=(),
                cited_sources=(verified_source,),
            )

    builder = SkillBuilder(
        config=config,
        retrieval_service=FakeRetrievalService(),
        generation_service=FakeGenerationService(),
    )
    output_path = builder.build_skill("report.md")
    checkpoint = json.loads(config.checkpoint_path.read_text())
    report = output_path.read_text()
    source_table = report.split("## Source References", maxsplit=1)[1]

    assert checkpoint["index_identity"] == "a" * 64
    assert checkpoint["completed_question_ids"] == [item[0] for item in SCAN_QUESTIONS]
    assert "verified passage" not in config.checkpoint_path.read_text()
    assert "verified-paper" in source_table
    assert "Verified \\| Paper" in source_table
    assert "invented-paper" not in source_table
