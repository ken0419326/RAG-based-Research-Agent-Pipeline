import os
import argparse
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 載入環境變數
load_dotenv()

# --- 配置 ---
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

class RAGQuerySystem:
    def __init__(self):
        # 1. 初始化 Embedding Model
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 2. 連接 Vector DB
        self.db_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.db_client.get_or_create_collection(name="acl_research")
        
        # 3. 初始化 OpenAI Client (LiteLLM 接口)
        self.client = OpenAI(
            api_key=os.getenv("LITELLM_API_KEY"),
            base_url=os.getenv("LITELLM_BASE_URL") # 填入 https://litellm.netdb.csie.ncku.edu.tw
        )
        
        # 4. 對話歷史
        self.history = []

    def retrieve(self, query, top_k=5):
        """從資料庫檢索相關片段"""
        query_vector = self.embed_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        return results

    def generate_answer(self, query, context_results, model="gpt-oss:20b"):
        """組裝 Prompt 並透過 OpenAI SDK 呼叫 LLM"""
        
        context_list = []
        sources = []
        for i in range(len(context_results['documents'][0])):
            doc = context_results['documents'][0][i]
            meta = context_results['metadatas'][0][i]
            source_info = f"[{meta.get('title', 'Unknown')}, {meta.get('year', 'N/A')}]"
            context_list.append(f"來源 {i+1} {source_info}:\n{doc}")
            sources.append(source_info)

        context_str = "\n\n".join(context_list)

        system_prompt = (
            "你是一位專業的 NLP 研究助理。請根據下方的參考資料回答問題。\n"
            "若資料不足請直說。回答需專業且精確，並在適當時機引用來源標籤。"
        )

        # 建立訊息清單
        messages = [{"role": "system", "content": system_prompt}]
        # 加入歷史紀錄 (最近 3 輪 = 6 條)
        messages.extend(self.history[-6:])
        
        user_content = f"--- 參考資料 ---\n{context_str}\n\n--- 問題 ---\n{query}"
        messages.append({"role": "user", "content": user_content})

        # 呼叫 API (符合你提供的範例寫法)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )

        answer = response.choices[0].message.content
        
        # 儲存對話
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        
        return answer, list(set(sources))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="輸入你的問題")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-oss:20b")
    args = parser.parse_args()

    rag = RAGQuerySystem()

    if args.query:
        # 單次查詢
        context = rag.retrieve(args.query, top_k=args.top_k)
        answer, sources = rag.generate_answer(args.query, context, model=args.model)
        print(f"\n回答：\n{answer}\n\n引用來源：{sources}")
    else:
        # 互動模式
        print("已進入互動模式 (輸入 exit 離開)")
        while True:
            u_input = input("\n問題: ")
            if u_input.lower() in ['exit', 'quit']: break
            context = rag.retrieve(u_input, top_k=args.top_k)
            answer, sources = rag.generate_answer(u_input, context, model=args.model)
            print(f"\n回答：\n{answer}\n\n引用來源：{sources}")

if __name__ == "__main__":
    main()