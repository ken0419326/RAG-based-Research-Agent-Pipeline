import os
import subprocess
import sys
from pathlib import Path

from config import PROJECT_ROOT


def relative_tree(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    return {entry.relative_to(path) for entry in path.rglob("*")}


def test_application_imports_have_no_filesystem_side_effects(tmp_path):
    script = """
import config
import downloader
import data_update
import rag_query
import skill_builder
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    data_before = relative_tree(PROJECT_ROOT / "data")
    chroma_before = relative_tree(PROJECT_ROOT / "chroma_db")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
    assert relative_tree(PROJECT_ROOT / "data") == data_before
    assert relative_tree(PROJECT_ROOT / "chroma_db") == chroma_before


def test_config_root_is_stable_in_a_different_current_directory(tmp_path):
    script = "from config import AppConfig; print(AppConfig.load(load_env_file=False).project_root)"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert Path(result.stdout.strip()) == PROJECT_ROOT


def test_ingestion_cli_runs_from_a_different_current_directory(tmp_path):
    missing_raw_dir = tmp_path / "missing-raw"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["RAW_DATA_DIR"] = str(missing_raw_dir)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "data_update.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert str(missing_raw_dir) in result.stderr
    assert list(tmp_path.iterdir()) == []
