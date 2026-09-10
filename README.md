# Build-Your-Personal-RAG

## 1. 專案簡介
* **知識主題**：本專案聚焦於「ACL Anthology 2025–2026 年論文中關於同理心 (Empathy)、情緒理解與價值對齊 (Value Alignment) 的研究」。
* **選擇理由**：隨著大型語言模型 (LLM) 的普及，如何讓模型展現人類般的同理心並符合人類價值觀（Value Alignment）是當前 NLP 領域的核心課題。
* **資料來源**：固定 corpus manifest 定義 50 份從 ACL Anthology 選出的論文，格式為 PDF。
* **技術選型**：
    * **LLM 接口**：可選擇使用 OpenAI SDK 介接使用者設定的 OpenAI-compatible provider。
    * **向量資料庫**：ChromaDB (Persistent Mode)。
    * **嵌入模型**：paraphrase-multilingual-MiniLM-L12-v2。

## 2. 系統架構說明
```mermaid
graph LR
    A[data/raw PDF] --> B[data_update.py]
    B --> C{RecursiveCharacter\nChunking}
    C --> D[Sentence-Transformer\nMiniLM-L12]
    D --> E[(ChromaDB)]
    E --> F[rag_query.py]
    F --> G[OpenAI-compatible LLM optional]
    G --> H[skill_builder.py]
    H --> I[skill.md]
```

## 3. 設計決策說明 (Design Decisions)

* **Chunking 策略**：
    * PDF 逐頁獨立切分，不讓 chunk 跨越 page boundary。
    * **設定參數**：chunk_size=750, chunk_overlap=100，單位皆為 characters。
    * **決策理由**：固定 character windows 與 overlap 可產生可重現的 chunk boundaries。
* **Embedding 模型選擇**：
    * 選用 paraphrase-multilingual-MiniLM-L12-v2。
    * **決策理由**：此多語系模型在語意對齊上表現優異，且體積適中，適合本地 CPU 環境。
* **Vector DB 選型**：
    * 選擇 **ChromaDB**。
    * **決策理由**：提供嵌入式存儲，無需透過 Docker 啟動服務，複現性最高且適合小型研究專案。
* **Retrieval 策略**：
    * **Top-k 設定**：預設為 3。
    * **決策理由**：在提供足夠脈絡與控制模型回應時間（Latency）之間取得最佳平衡，避免過長的 Context 導致推論超時。
* **Prompt Engineering**：
    * **設計邏輯**：強制要求模型根據參考資料回答，並列出引用來源。若資料不足則必須誠實回答「不知道」。
    * **決策理由**：有效抑制 LLM 產生幻覺（Hallucination）。
* **Idempotency 設計**：
    * **實作方式**：data_update.py 透過 --rebuild 參數確保冪等性。
    * **決策理由**：執行時若帶此參數，會清空舊目錄重新構建，確保索引與當前資料夾內容完全一致。

## 4. 環境設定與執行方式

### 4-1. Python 版本與虛擬環境
* `pyproject.toml` 目前限定 **Python 3.11–3.12**；正式支援聲明仍需在後續 CI matrix 完成驗證。Python 3.13 不在目前範圍內。

```bash
# ① 確認 Python 與 uv 版本
python3 --version
uv --version

# ② 依 uv.lock 安裝 runtime 與 development dependencies
uv sync --locked --all-groups

# ③ 建立本機設定檔（不得提交真實 API key）
cp .env.example .env
```

Corpus acquisition、ingestion 與 indexing 不需要 LLM credentials。目前的問答與報告生成則需要在 `.env` 設定 provider-neutral 的 `LLM_BASE_URL`、`LLM_API_KEY` 與 `LLM_MODEL`。Groq 可作為 OpenAI-compatible provider 的範例，但 application logic 不依賴 Groq；endpoint、model availability、free tier 與 rate limits 可能變動，使用前請查閱 provider 的最新官方文件。

所有相對路徑均以 project root 解析，而不是呼叫命令時的 current working directory。

### 4-2. Vector DB 啟動
本專案使用 **ChromaDB (Embedded Mode)**，無需啟動 Docker 容器。資料將儲存於專案目錄下的 chroma_db/。

### 4-3. 完整執行流程
```bash
# ④ 下載原始資料
uv run python downloader.py

# ⑤ 解析 manifest 中的 PDF 並產生 structured chunks（不載入 embedding/ChromaDB）
uv run python data_update.py --prepare-only

# ⑥ 全量重建索引
uv run python data_update.py --rebuild

# ⑦ 測試 RAG 問答（需要 LLM_* 設定）
uv run python rag_query.py

# ⑧ 生成 Skill 文件（需要 LLM_* 設定）
uv run python skill_builder.py --output skill.md
```

`--prepare-only` 僅處理 fixed manifest 的 50 份 PDF，逐頁使用 `pypdf` 擷取全文並將 provenance-rich chunks 原子寫入 Git-ignored 的 `data/processed/chunks.jsonl`。摘要輸出會分別列出 selected、parsed、failed、empty、pages 與 chunks；此命令不需要 embedding model、ChromaDB 或 LLM credentials。

#### Downloader 行為

`downloader.py` 搜尋 ACL Anthology 中符合 `2025 <= paper.year <= 2026` 的所有 venue，對 normalized title 與 abstract 進行指定 keyword matching。Ranking score 為 `2 × title keyword matches + abstract keyword matches`，再依 score descending、year descending、Anthology paper ID ascending 排序並選出固定 top 50 unique paper IDs。Standalone `value`、`alignment` 與 `sentiment` 不屬於 keywords。

Canonical manifest 首次寫入 `corpus/manifest.json` 後即固定 paper IDs；後續執行仍會取得 catalog 並報告 matched count，但不會因 catalog 新增項目而替換 fixed selection。下載的 PDF 與 associated JSON metadata 寫入 `data/raw/` 並由 Git 忽略。若首次選擇時 matched papers 少於 50，命令會停止，不會放寬條件。

執行摘要中的 `selected` 固定表示 manifest 的 50 篇論文；`downloaded` 是本次成功下載且通過 PDF signature 驗證的數量；`already existing` 是無須重複下載的既有有效 PDF；`failed` 是 HTTP 或內容驗證失敗的數量，且不會被計入成功下載。

## 5. 資料來源聲明 (Data Sources Statement)

| 來源名稱 | 類型 | 授權 / 合規依據 | 數量 |
| :--- | :--- | :--- | :--- |
| ACL Anthology  | PDF | CC BY 4.0 | 固定 manifest 50 篇 |
