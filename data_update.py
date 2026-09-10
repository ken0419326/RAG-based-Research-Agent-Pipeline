"""Prepare corpus chunks and build validated local indexes."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from config import AppConfig, ConfigurationError
from documents import PREPARED_FILENAME, PreparationError, PreparationSummary, prepare_corpus
from indexing import IndexBuildError, IndexBuildSummary, build_index


class DataUpdatePipeline:
    """Command-level orchestration with lazy model and vector-store initialization."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.load()
        # Retained as observable state for import/preflight tests and compatibility.
        self.model: Any | None = None
        self.db_client: Any | None = None
        self.collection: Any | None = None

    def prepare_only(self) -> PreparationSummary:
        """Create structured JSONL chunks without loading embedding or vector clients."""
        self.config.validate_preparation()
        return prepare_corpus(
            manifest_path=self.config.corpus_manifest_path,
            project_root=self.config.project_root,
            raw_dir=self.config.raw_dir,
            output_path=self.config.processed_dir / PREPARED_FILENAME,
        )

    def build_index(
        self,
        *,
        embedder_factory=None,
        client_factory=None,
        now=None,
    ) -> IndexBuildSummary:
        """Build a new collection and activate it only after validation."""
        self.config.validate_index_build()
        options = {}
        if embedder_factory is not None:
            options["embedder_factory"] = embedder_factory
        if client_factory is not None:
            options["client_factory"] = client_factory
        if now is not None:
            options["now"] = now
        return build_index(
            chunks_path=self.config.processed_dir / PREPARED_FILENAME,
            corpus_manifest_path=self.config.corpus_manifest_path,
            chroma_path=self.config.chroma_path,
            collection_prefix=self.config.collection_name,
            embedding_model=self.config.embedding_model,
            index_manifest_path=self.config.chroma_path / "index_manifest.json",
            active_pointer_path=self.config.chroma_path / "active_index.json",
            **options,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--build-index", action="store_true")
    args = parser.parse_args(argv)
    try:
        pipeline = DataUpdatePipeline()
        if args.prepare_only:
            summary = pipeline.prepare_only()
            print(
                "Preparation summary: "
                f"selected={summary.selected_papers}, parsed={summary.parsed_papers}, "
                f"failed={summary.failed_papers}, empty={summary.empty_papers}, "
                f"pages={summary.total_pages}, chunks={summary.total_chunks}"
            )
            print(f"JSONL output: {summary.output_path}")
            return 1 if summary.failed_papers or summary.empty_papers else 0

        summary = pipeline.build_index()
        manifest = summary.manifest
        print(
            "Index summary: "
            f"model={manifest.embedding_model}, dimension={manifest.embedding_dimension}, "
            f"papers={manifest.paper_count}, chunks={manifest.chunk_count}, "
            f"collection={manifest.collection_name}, validated={summary.validated}"
        )
        print(f"Chroma directory: {summary.chroma_path}")
        print(f"Index manifest: {summary.index_manifest_path}")
        print(f"Active pointer: {summary.active_pointer_path}")
        print(f"Build duration: {summary.duration_seconds:.3f} seconds")
    except (ConfigurationError, PreparationError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except IndexBuildError as error:
        print(f"Index build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
