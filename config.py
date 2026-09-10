"""Shared, side-effect-free application configuration and preflight checks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigurationError(RuntimeError):
    """Raised when a command cannot safely start with the current configuration."""


def _project_path(value: str | None, default: str) -> Path:
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Configuration shared by downloader, ingestion, retrieval, and reporting."""

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    corpus_manifest_path: Path
    anthology_repo_dir: Path
    chroma_path: Path
    collection_name: str
    embedding_model: str
    checkpoint_path: Path
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None

    @classmethod
    def load(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        load_env_file: bool = True,
    ) -> AppConfig:
        """Load configuration without creating paths or external clients."""
        if load_env_file:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env", override=False)

        values = os.environ if environ is None else environ
        return cls(
            project_root=PROJECT_ROOT,
            raw_dir=_project_path(values.get("RAW_DATA_DIR"), "data/raw"),
            processed_dir=_project_path(values.get("PROCESSED_DATA_DIR"), "data/processed"),
            corpus_manifest_path=_project_path(
                values.get("CORPUS_MANIFEST_PATH"), "corpus/manifest.json"
            ),
            anthology_repo_dir=_project_path(
                values.get("ACL_ANTHOLOGY_REPO_DIR"), ".cache/acl-anthology"
            ),
            chroma_path=_project_path(values.get("CHROMA_PERSIST_DIR"), "chroma_db"),
            collection_name=values.get("CHROMA_COLLECTION", "acl_research"),
            embedding_model=values.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
            checkpoint_path=_project_path(
                values.get("REPORT_CHECKPOINT_PATH"), "temp_insights.json"
            ),
            llm_base_url=values.get("LLM_BASE_URL") or None,
            llm_api_key=values.get("LLM_API_KEY") or None,
            llm_model=values.get("LLM_MODEL") or None,
        )

    def validate_ingestion(self) -> None:
        """Validate ingestion inputs without requiring an LLM or loading a model."""
        if not self.raw_dir.is_dir():
            raise ConfigurationError(
                f"Raw data directory does not exist: {self.raw_dir}. "
                "Run downloader.py first or set RAW_DATA_DIR."
            )
        supported_files = (
            path
            for path in self.raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt"}
        )
        if not any(supported_files):
            raise ConfigurationError(f"No supported input files found in: {self.raw_dir}")

    def validate_retrieval(self) -> None:
        """Validate retrieval inputs without requiring an LLM or loading a model."""
        if not self.chroma_path.is_dir() or not any(self.chroma_path.iterdir()):
            raise ConfigurationError(
                f"ChromaDB index does not exist or is empty: {self.chroma_path}. "
                "Run data_update.py first or set CHROMA_PERSIST_DIR."
            )

    def validate_generation(self, *, model_override: str | None = None) -> None:
        """Validate optional LLM settings only when generation is requested."""
        missing = []
        if not self.llm_base_url:
            missing.append("LLM_BASE_URL")
        if not self.llm_api_key:
            missing.append("LLM_API_KEY")
        if not (model_override or self.llm_model):
            missing.append("LLM_MODEL")
        if missing:
            names = ", ".join(missing)
            raise ConfigurationError(
                f"LLM generation requires these environment variables: {names}."
            )
