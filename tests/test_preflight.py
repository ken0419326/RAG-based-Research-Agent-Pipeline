from dataclasses import replace

import pytest

import data_update
import rag_query
from config import AppConfig, ConfigurationError
from data_update import DataUpdatePipeline
from rag_query import RAGQuerySystem


def load_config() -> AppConfig:
    return AppConfig.load({}, load_env_file=False)


def test_ingestion_missing_directory_fails_before_heavy_initialization(tmp_path):
    config = replace(load_config(), raw_dir=tmp_path / "missing")
    pipeline = DataUpdatePipeline(config=config)

    with pytest.raises(ConfigurationError, match="Raw data directory"):
        pipeline.run(rebuild=True)

    assert pipeline.model is None
    assert pipeline.db_client is None
    assert pipeline.collection is None


def test_retrieval_missing_index_fails_before_heavy_initialization(tmp_path):
    config = replace(load_config(), chroma_path=tmp_path / "missing")
    rag = RAGQuerySystem(config=config)

    with pytest.raises(ConfigurationError, match="ChromaDB index"):
        rag.retrieve("test")

    assert rag.embed_model is None
    assert rag.db_client is None
    assert rag.collection is None


def test_generation_missing_configuration_does_not_create_client():
    rag = RAGQuerySystem(config=load_config())

    with pytest.raises(ConfigurationError, match="LLM generation"):
        rag.generate_answer("test", {"documents": [[]], "metadatas": [[]]})

    assert rag.client is None


def test_ingestion_cli_returns_nonzero_for_missing_input(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path / "missing"))

    assert data_update.main([]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_query_cli_returns_nonzero_for_missing_index(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "missing"))

    assert rag_query.main(["--query", "test"]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_query_cli_validates_llm_only_after_retrieval_preflight(monkeypatch, tmp_path, capsys):
    (tmp_path / "chroma.sqlite3").write_bytes(b"test fixture")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert rag_query.main(["--query", "test"]) == 2
    error = capsys.readouterr().err
    assert "LLM generation" in error
    assert "LLM_BASE_URL" in error
