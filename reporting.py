"""Pure report rendering and atomic checkpoint/output helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1


class ReportError(RuntimeError):
    """Raised when report state is corrupt, stale, or cannot be written safely."""


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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
    except OSError as error:
        raise ReportError(f"Could not write {path}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    serialized = json.dumps(content, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, serialized)


def new_checkpoint(index_identity: str) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "index_identity": index_identity,
        "completed_question_ids": [],
        "results": [],
    }


def load_checkpoint(path: Path, *, index_identity: str, question_ids: set[str]) -> dict[str, Any]:
    if not path.exists():
        return new_checkpoint(index_identity)
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"Report checkpoint is corrupt: {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ReportError(f"Report checkpoint must be a JSON object: {path}")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ReportError("Report checkpoint schema version is unsupported.")
    if checkpoint.get("index_identity") != index_identity:
        raise ReportError(
            "Report checkpoint index identity does not match the active index; "
            "move or remove the stale checkpoint before restarting."
        )
    completed = checkpoint.get("completed_question_ids")
    results = checkpoint.get("results")
    if not isinstance(completed, list) or not all(isinstance(item, str) for item in completed):
        raise ReportError("Report checkpoint has invalid completed_question_ids.")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise ReportError("Report checkpoint has invalid results.")
    if len(set(completed)) != len(completed) or not set(completed) <= question_ids:
        raise ReportError("Report checkpoint contains unknown or duplicate question IDs.")
    result_ids = [item.get("question_id") for item in results]
    if result_ids != completed:
        raise ReportError("Report checkpoint results do not match completed question IDs.")
    return checkpoint


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_source_table(results: list[dict[str, Any]]) -> str:
    """Render only source metadata copied from verified retrieval results."""
    rows = []
    for result in results:
        question_id = result["question_id"]
        for source in result["sources"]:
            rows.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        question_id,
                        source["source_id"],
                        source["paper_id"],
                        source["title"],
                        source["page"],
                        source["chunk_id"],
                        source["url"],
                    )
                )
                + " |"
            )
    header = (
        "| Question | Source ID | Paper ID | Title | Page | Chunk ID | URL |\n"
        "|---|---|---|---|---:|---|---|"
    )
    return "\n".join([header, *rows]) if rows else f"{header}\n| — | — | — | — | — | — | — |"


def render_report(
    results: list[dict[str, Any]],
    *,
    index_identity: str,
    embedding_model: str,
) -> str:
    sections = []
    for result in results:
        invalid = result["invalid_source_ids"]
        warning = f"\n\n> Invalid source IDs reported: {', '.join(invalid)}" if invalid else ""
        sections.append(f"## {result['question']}\n\n{result['answer']}{warning}")
    source_table = render_source_table(results)
    return (
        "# 同理心與價值對齊研究報告\n\n"
        "> 此報告由固定 corpus 的 active index 產生；來源表完全由 verified "
        "retrieval metadata 程式化建立。Citation ID validation 不代表內容事實正確。\n\n"
        f"- Index identity: `{index_identity}`\n"
        f"- Embedding model: `{embedding_model}`\n\n"
        + "\n\n".join(sections)
        + "\n\n## Source References\n\n"
        + source_table
        + "\n"
    )
