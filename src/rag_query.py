import argparse
import json
import sys
from typing import Any

from config import AppConfig, ConfigurationError
from generation import GenerationError, GenerationResult, GenerationService
from hybrid_retrieval import DENSE, RETRIEVAL_CONFIGURATIONS, PaperRetrievalService
from retrieval import RetrievalError, RetrievalResult, RetrievalService


class RAGQuerySystem:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.load()
        self.embed_model: Any | None = None
        self.db_client: Any | None = None
        self.collection: Any | None = None
        self.client: Any | None = None
        self.history = []
        self.retrieval_service: RetrievalService | None = None
        self.generation_service: GenerationService | None = None

    def _initialize_retrieval(self) -> None:
        if self.embed_model is not None:
            return

        self.config.validate_retrieval()
        service = RetrievalService(self.config)
        service.initialize()
        self.retrieval_service = service
        self.embed_model = service.embedder
        self.db_client = service.client
        self.collection = service.collection

    def _initialize_generation(self, model_override: str | None = None) -> str:
        self.config.validate_generation(model_override=model_override)
        if self.generation_service is None:
            self.generation_service = GenerationService(self.config)
        return model_override or self.config.llm_model or ""

    def retrieve(self, query, top_k=5):
        """從資料庫檢索相關片段"""
        self._initialize_retrieval()
        return self.retrieval_service.retrieve(query, top_k=top_k)

    def generate_answer(
        self,
        query: str,
        context_results: list[RetrievalResult],
        model: str | None = None,
    ) -> GenerationResult:
        """Generate a source-aware answer while retaining only process-local history."""
        self._initialize_generation(model_override=model)
        result = self.generation_service.generate(
            query,
            context_results,
            history=self.history[-6:],
            model_override=model,
        )
        self.client = self.generation_service.client
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": result.answer})
        return result

    def ask(self, query: str, *, top_k: int = 5, model: str | None = None) -> GenerationResult:
        sources = self.retrieve(query, top_k=top_k)
        return self.generate_answer(query, sources, model=model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="輸入你的問題")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument(
        "--retrieval-config",
        choices=RETRIEVAL_CONFIGURATIONS,
        default=DENSE,
        help="Paper retrieval strategy; applies only to --retrieval-only.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    try:
        config = AppConfig.load()
        if args.retrieval_only:
            if not args.query:
                parser.error("--retrieval-only requires --query")
            if args.retrieval_config == DENSE:
                service = RetrievalService(config)
                results = service.retrieve(args.query, top_k=args.top_k)
            else:
                service = PaperRetrievalService(config)
                results = service.retrieve(
                    args.query,
                    configuration=args.retrieval_config,
                    top_k=args.top_k,
                )
            _print_retrieval_results(results, json_output=args.json_output)
            return 0

        if args.retrieval_config != DENSE:
            parser.error("--retrieval-config applies only to --retrieval-only")

        config.validate_retrieval()
        config.validate_generation(model_override=args.model)
        rag = RAGQuerySystem(config=config)

        if args.query:
            result = rag.ask(args.query, top_k=args.top_k, model=args.model)
            _print_generation_result(result)
        else:
            print("已進入互動模式 (輸入 exit 離開)")
            while True:
                u_input = input("\n問題: ")
                if u_input.lower() in ["exit", "quit"]:
                    break
                result = rag.ask(u_input, top_k=args.top_k, model=args.model)
                _print_generation_result(result)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except RetrievalError as exc:
        print(f"Retrieval error: {exc}", file=sys.stderr)
        return 2
    except GenerationError as exc:
        print(f"Generation error: {exc}", file=sys.stderr)
        return 1
    return 0


def _print_retrieval_results(results: list[RetrievalResult], *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
        return
    for result in results:
        print(
            f"{result.source_id} rank={result.rank} distance={result.distance:.6f}\n"
            f"{result.title} ({result.year}, {result.venue}), page {result.page}\n"
            f"paper_id={result.paper_id} chunk_id={result.chunk_id}\n"
            f"URL: {result.url}\n"
            f"{result.text}\n"
        )


def _print_generation_result(result: GenerationResult) -> None:
    cited_titles = [source.title for source in result.cited_sources]
    print(f"\n回答：\n{result.answer}")
    print(f"\n有效引用 ID：{list(result.cited_source_ids)}")
    print(f"無效引用 ID：{list(result.invalid_source_ids)}")
    print(f"引用論文：{cited_titles}")
    print(
        "引用 ID 驗證："
        f"{'通過' if result.citation_validation_passed else '失敗'}"
        "（僅驗證 ID，不代表內容正確性）"
    )


if __name__ == "__main__":
    raise SystemExit(main())
