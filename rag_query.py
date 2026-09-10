import argparse
import json
import sys
from typing import Any

from config import AppConfig, ConfigurationError
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
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
            )
        return model_override or self.config.llm_model or ""

    def retrieve(self, query, top_k=5):
        """從資料庫檢索相關片段"""
        self._initialize_retrieval()
        return self.retrieval_service.query_raw(query, top_k=top_k)

    def generate_answer(self, query, context_results, model=None):
        """組裝 Prompt 並透過 OpenAI SDK 呼叫 LLM"""
        effective_model = self._initialize_generation(model_override=model)

        context_list = []
        sources = []
        for i in range(len(context_results["documents"][0])):
            doc = context_results["documents"][0][i]
            meta = context_results["metadatas"][0][i]
            source_info = f"[{meta.get('title', 'Unknown')}, {meta.get('year', 'N/A')}]"
            context_list.append(f"來源 {i + 1} {source_info}:\n{doc}")
            sources.append(source_info)

        context_str = "\n\n".join(context_list)

        system_prompt = (
            "你是一位專業的 NLP 研究助理。請根據下方的參考資料回答問題。\n"
            "若資料不足請直說。回答需專業且精確，並在適當時機引用來源標籤。"
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-6:])

        user_content = f"--- 參考資料 ---\n{context_str}\n\n--- 問題 ---\n{query}"
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=effective_model,
            messages=messages,
        )

        answer = response.choices[0].message.content

        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})

        return answer, list(set(sources))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="輸入你的問題")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)

    try:
        config = AppConfig.load()
        if args.retrieval_only:
            if not args.query:
                parser.error("--retrieval-only requires --query")
            service = RetrievalService(config)
            results = service.retrieve(args.query, top_k=args.top_k)
            _print_retrieval_results(results, json_output=args.json_output)
            return 0

        config.validate_retrieval()
        config.validate_generation(model_override=args.model)
        rag = RAGQuerySystem(config=config)

        if args.query:
            context = rag.retrieve(args.query, top_k=args.top_k)
            answer, sources = rag.generate_answer(args.query, context, model=args.model)
            print(f"\n回答：\n{answer}\n\n引用來源：{sources}")
        else:
            print("已進入互動模式 (輸入 exit 離開)")
            while True:
                u_input = input("\n問題: ")
                if u_input.lower() in ["exit", "quit"]:
                    break
                context = rag.retrieve(u_input, top_k=args.top_k)
                answer, sources = rag.generate_answer(u_input, context, model=args.model)
                print(f"\n回答：\n{answer}\n\n引用來源：{sources}")
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except RetrievalError as exc:
        print(f"Retrieval error: {exc}", file=sys.stderr)
        return 2
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


if __name__ == "__main__":
    raise SystemExit(main())
