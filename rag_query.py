import argparse
import sys
from typing import Any

from config import AppConfig, ConfigurationError


class RAGQuerySystem:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.load()
        self.embed_model: Any | None = None
        self.db_client: Any | None = None
        self.collection: Any | None = None
        self.client: Any | None = None
        self.history = []

    def _initialize_retrieval(self) -> None:
        if self.embed_model is not None:
            return

        self.config.validate_retrieval()
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.embed_model = SentenceTransformer(self.config.embedding_model)
        self.db_client = chromadb.PersistentClient(path=str(self.config.chroma_path))
        self.collection = self.db_client.get_or_create_collection(name=self.config.collection_name)

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
        query_vector = self.embed_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        return results

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
    args = parser.parse_args(argv)

    try:
        config = AppConfig.load()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
