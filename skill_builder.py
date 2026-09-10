import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from config import AppConfig, ConfigurationError


class SkillBuilder:
    def __init__(self, model: str | None = None, config: AppConfig | None = None):
        self.config = config or AppConfig.load()
        self.model = model
        self.embed_model: Any | None = None
        self.db_client: Any | None = None
        self.collection: Any | None = None
        self.client: Any | None = None
        self.scan_queries = [
            "這個知識庫涵蓋哪些關於同理心(empathy)或價值對齊(value alignment)的主要概念和子主題？",
            "在這些論文中，目前最重要的研究方向或趨勢為何？",
            "這批論文中主要提到的核心工具、模型架構、框架、或方法論有哪些？",
            "主要的作者、研究機構、或資料來源有哪些？",
        ]

    def _initialize_retrieval(self) -> None:
        if self.embed_model is not None:
            return

        self.config.validate_retrieval()
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.embed_model = SentenceTransformer(self.config.embedding_model)
        self.db_client = chromadb.PersistentClient(path=str(self.config.chroma_path))
        self.collection = self.db_client.get_or_create_collection(name=self.config.collection_name)

    def _initialize_generation(self) -> str:
        self.config.validate_generation(model_override=self.model)
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
            )
        return self.model or self.config.llm_model or ""

    def rag_retrieve(self, query, top_k=2):
        self._initialize_retrieval()
        query_vector = self.embed_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_k)
        return "\n".join(results["documents"][0])

    def build_skill(self, output_file="skill.md"):
        self.config.validate_retrieval()
        effective_model = self._initialize_generation()
        checkpoint_path = self.config.checkpoint_path
        output_path = Path(output_file).expanduser()
        if not output_path.is_absolute():
            output_path = self.config.project_root / output_path

        if checkpoint_path.exists():
            print(f"偵測到暫存檔 {checkpoint_path}，正在載入已完成的洞察...")
            with checkpoint_path.open("r", encoding="utf-8") as f:
                intermediate_insights = json.load(f)
        else:
            intermediate_insights = []

        start_idx = len(intermediate_insights)
        if start_idx < len(self.scan_queries):
            print(f"開始主題掃描 (從第 {start_idx + 1} 個問題開始)...")
            for i in range(start_idx, len(self.scan_queries)):
                q = self.scan_queries[i]
                print(f"正在分析維度 {i + 1}/{len(self.scan_queries)}...")

                context = self.rag_retrieve(q)

                try:
                    response = self.client.chat.completions.create(
                        model=effective_model,
                        messages=[
                            {"role": "system", "content": "你是一個專業的學術知識提取器。"},
                            {"role": "user", "content": f"參考資料：\n{context}\n\n問題：{q}"},
                        ],
                        timeout=120,
                    )
                    insight = response.choices[0].message.content
                    intermediate_insights.append({"query": q, "insight": insight})

                    with checkpoint_path.open("w", encoding="utf-8") as f:
                        json.dump(intermediate_insights, f, ensure_ascii=False, indent=2)

                    print(f"維度 {i + 1} 完成並已存檔。")
                    time.sleep(5)
                except Exception as e:
                    print(f"維度 {i + 1} 失敗: {e}。請稍後重新執行程式。")
                    return

        print("正在進行最後的 Skill.md 整合...")
        all_context = "\n\n".join(
            [f"### {item['query']}\n{item['insight']}" for item in intermediate_insights]
        )

        today = date.today().strftime("%Y-%m-%d")

        final_prompt = f"""
        你是一個專業的 AI Agent 技能構建專家（Skill Builder）。
        請將下方的研究洞察整合為一份結構嚴謹、專業的 Markdown 文件。

        ### 嚴格格式要求：
        1. 使用 Markdown 標題語法（# 和 ##）。
        2. 必須包含以下所有章節，不得遺漏：

        # Skill: [同理心與價值對齊研究綜述]

        ## Metadata
        - **知識領域**：自然語言處理 / 同理心計算 / AI 價值對齊
        - **資料來源數量**：約 30 份 ACL 論文摘要
        - **最後更新時間**：{today}
        - **適用 Agent 類型**：學術研究助手 / 情感運算顧問

        ## Overview（一段話摘要）
        [請根據資料總結 200 字以內的摘要，說明此知識庫的核心範圍]

        ## Core Concepts（核心概念）
        [條列 5–15 個最關鍵概念，如：Cognitive Empathy, Affective Empathy, Reward Modeling 等，並附 1-2 句說明]

        ## Key Trends（最新趨勢）
        [條列 3–10 個目前最重要的發展方向，例如：大型語言模型的同理心評測、多元文化價值對齊等]

        ## Key Entities（重要實體）
        [將作者、機構、工具(如 PyTorch)、框架、資料集等分類條列]

        ## Methodology & Best Practices（方法論與最佳實踐）
        [總結論文中常用的評估指標(如 BLEU, ROUGE, 人工評測)或實驗流程]

        ## Knowledge Gaps & Limitations（知識邊界）
        [說明此 Skill 的侷限，例如：資料僅限於 ACL 論文摘要、缺乏 2026 年以後的資料等]

        ## Example Q&A（代表性問答）
        [列出 3–5 組具代表性的問題與簡短答案]

        ## Source References（來源索引）
        [列出主要的論文標題與 ID]

        ---
        以下是供你整合的原始研究洞察內容：
        {all_context}
        """

        try:
            final_res = self.client.chat.completions.create(
                model=effective_model,
                messages=[{"role": "user", "content": final_prompt}],
            )
            with output_path.open("w", encoding="utf-8") as f:
                f.write(final_res.choices[0].message.content)

            print(f"成功生成 {output_path}！")
            if checkpoint_path.exists():
                checkpoint_path.unlink()

        except Exception as e:
            print(f"整合階段失敗: {e}。暫存檔已保留，請稍後重試整合。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="skill.md")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args(argv)

    try:
        config = AppConfig.load()
        config.validate_retrieval()
        config.validate_generation(model_override=args.model)
        builder = SkillBuilder(model=args.model, config=config)
        builder.build_skill(output_file=args.output)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
