"""Build a checkpointed, provenance-backed Markdown research report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from config import AppConfig, ConfigurationError
from generation import GenerationError, GenerationService
from reporting import (
    ReportError,
    atomic_write_json,
    atomic_write_text,
    load_checkpoint,
    render_report,
)
from retrieval import RetrievalError, RetrievalService

SCAN_QUESTIONS = (
    (
        "scope",
        "這個知識庫涵蓋哪些關於同理心(empathy)或價值對齊(value alignment)的主要概念和子主題？",
    ),
    ("trends", "在這些論文中，目前最重要的研究方向或趨勢為何？"),
    ("methods", "這批論文中主要提到的核心工具、模型架構、框架、或方法論有哪些？"),
    ("entities", "主要的作者、研究機構、或資料來源有哪些？"),
)


class SkillBuilder:
    def __init__(
        self,
        model: str | None = None,
        config: AppConfig | None = None,
        *,
        retrieval_service: RetrievalService | None = None,
        generation_service: GenerationService | None = None,
    ) -> None:
        self.config = config or AppConfig.load()
        self.model = model
        self.retrieval_service = retrieval_service or RetrievalService(self.config)
        self.generation_service = generation_service or GenerationService(self.config)
        self.scan_queries = [question for _, question in SCAN_QUESTIONS]
        # Compatibility state remains lazy and observable by existing preflight tests.
        self.embed_model: Any | None = None
        self.db_client: Any | None = None
        self.collection: Any | None = None
        self.client: Any | None = None

    def _initialize_retrieval(self) -> None:
        self.config.validate_retrieval()
        self.retrieval_service.initialize()
        self.embed_model = self.retrieval_service.embedder
        self.db_client = self.retrieval_service.client
        self.collection = self.retrieval_service.collection

    def _initialize_generation(self) -> str:
        self.config.validate_generation(model_override=self.model)
        return self.model or self.config.llm_model or ""

    def build_skill(
        self, output_file: str = "reports/research_report.md", *, top_k: int = 2
    ) -> Path:
        self._initialize_retrieval()
        self._initialize_generation()
        active_index = self.retrieval_service.active_index
        if active_index is None:
            raise ReportError(
                "Active index metadata is unavailable after retrieval initialization."
            )

        output_path = Path(output_file).expanduser()
        if not output_path.is_absolute():
            output_path = self.config.project_root / output_path
        question_ids = {question_id for question_id, _ in SCAN_QUESTIONS}
        checkpoint = load_checkpoint(
            self.config.checkpoint_path,
            index_identity=active_index.index_identity,
            question_ids=question_ids,
        )
        completed = set(checkpoint["completed_question_ids"])

        for question_id, question in SCAN_QUESTIONS:
            if question_id in completed:
                continue
            sources = self.retrieval_service.retrieve(question, top_k=top_k)
            generated = self.generation_service.generate(
                question,
                sources,
                model_override=self.model,
            )
            if generated.invalid_source_ids:
                raise ReportError(
                    f"Question {question_id} returned invalid source IDs: "
                    f"{', '.join(generated.invalid_source_ids)}."
                )
            if not generated.cited_source_ids:
                raise ReportError(f"Question {question_id} returned no valid source citations.")
            self.client = self.generation_service.client
            checkpoint["completed_question_ids"].append(question_id)
            checkpoint["results"].append(
                {
                    "question_id": question_id,
                    "question": question,
                    "answer": generated.answer,
                    "cited_source_ids": list(generated.cited_source_ids),
                    "invalid_source_ids": list(generated.invalid_source_ids),
                    "sources": [
                        {
                            "rank": source.rank,
                            "source_id": source.source_id,
                            "distance": source.distance,
                            "chunk_id": source.chunk_id,
                            "paper_id": source.paper_id,
                            "title": source.title,
                            "year": source.year,
                            "venue": source.venue,
                            "page": source.page,
                            "url": source.url,
                        }
                        for source in generated.cited_sources
                    ],
                }
            )
            atomic_write_json(self.config.checkpoint_path, checkpoint)

        report = render_report(
            checkpoint["results"],
            index_identity=active_index.index_identity,
            corpus_manifest_sha256=active_index.corpus_manifest_sha256,
            embedding_model=active_index.embedding_model,
            generation_model=self.model or self.config.llm_model or "",
        )
        atomic_write_text(output_path, report)
        return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="reports/research_report.md")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args(argv)

    try:
        builder = SkillBuilder(model=args.model)
        output_path = builder.build_skill(output_file=args.output)
        print(f"Report written: {output_path}")
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except RetrievalError as error:
        print(f"Retrieval error: {error}", file=sys.stderr)
        return 2
    except GenerationError as error:
        print(f"Generation error: {error}", file=sys.stderr)
        return 1
    except ReportError as error:
        print(f"Report error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
