from dataclasses import replace

import pytest

from config import PROJECT_ROOT, AppConfig, ConfigurationError


def load_config(values: dict[str, str] | None = None) -> AppConfig:
    return AppConfig.load(values or {}, load_env_file=False)


def test_default_paths_are_resolved_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.project_root == PROJECT_ROOT
    assert config.raw_dir == PROJECT_ROOT / "data/raw"
    assert config.processed_dir == PROJECT_ROOT / "data/processed"
    assert config.chroma_path == PROJECT_ROOT / "chroma_db"
    assert config.checkpoint_path == PROJECT_ROOT / "temp_insights.json"


def test_relative_overrides_are_project_root_relative():
    config = load_config(
        {
            "RAW_DATA_DIR": "local/raw",
            "PROCESSED_DATA_DIR": "local/processed",
            "CHROMA_PERSIST_DIR": "local/chroma",
            "REPORT_CHECKPOINT_PATH": "local/checkpoint.json",
        }
    )

    assert config.raw_dir == PROJECT_ROOT / "local/raw"
    assert config.processed_dir == PROJECT_ROOT / "local/processed"
    assert config.chroma_path == PROJECT_ROOT / "local/chroma"
    assert config.checkpoint_path == PROJECT_ROOT / "local/checkpoint.json"


def test_absolute_path_override_is_preserved(tmp_path):
    config = load_config({"RAW_DATA_DIR": str(tmp_path)})

    assert config.raw_dir == tmp_path


def test_ingestion_validation_does_not_require_llm_settings(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"test fixture")
    config = replace(load_config(), raw_dir=tmp_path)

    config.validate_ingestion()


def test_retrieval_validation_does_not_require_llm_settings(tmp_path):
    (tmp_path / "chroma.sqlite3").write_bytes(b"test fixture")
    config = replace(load_config(), chroma_path=tmp_path)

    config.validate_retrieval()


def test_generation_validation_reports_only_missing_variable_names():
    config = load_config({"LLM_API_KEY": "do-not-print-this-value"})

    with pytest.raises(ConfigurationError) as error:
        config.validate_generation()

    message = str(error.value)
    assert "LLM_BASE_URL" in message
    assert "LLM_MODEL" in message
    assert "LLM_API_KEY" not in message
    assert "do-not-print-this-value" not in message


def test_generation_validation_accepts_provider_neutral_settings():
    config = load_config(
        {
            "LLM_BASE_URL": "https://provider.invalid/v1",
            "LLM_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
        }
    )

    config.validate_generation()


def test_model_override_satisfies_generation_model_requirement():
    config = load_config(
        {
            "LLM_BASE_URL": "https://provider.invalid/v1",
            "LLM_API_KEY": "test-key",
        }
    )

    config.validate_generation(model_override="test-model")
