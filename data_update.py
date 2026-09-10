import argparse
import json
import re
import shutil
import sys
from contextlib import suppress
from typing import Any

from config import AppConfig, ConfigurationError
from documents import PREPARED_FILENAME, PreparationError, PreparationSummary, prepare_corpus


class DataUpdatePipeline:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.load()
        self.model: Any | None = None
        self.db_client: Any | None = None
        self.collection: Any | None = None
        self.text_splitter: Any | None = None
        self.progress: Any | None = None

    def _initialize(self) -> None:
        if self.model is not None:
            return

        import chromadb
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm

        print(f"正在載入 Embedding 模型: {self.config.embedding_model}...")
        self.model = SentenceTransformer(self.config.embedding_model)
        self.db_client = chromadb.PersistentClient(path=str(self.config.chroma_path))
        self.collection = self.db_client.get_or_create_collection(name=self.config.collection_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
        self.progress = tqdm

    def clean_text(self, text):
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def process_content(self, rebuild=False):
        if rebuild:
            print(f"正在清空 {self.config.processed_dir}...")
            shutil.rmtree(self.config.processed_dir, ignore_errors=True)
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)

        print(f"正在檢查目錄: {self.config.raw_dir}")

        raw_files = [
            path.name
            for path in self.config.raw_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt"}
        ]
        print(f"找到 {len(raw_files)} 個原始檔案: {raw_files}")

        if not raw_files:
            print("錯誤：在 data/raw 中找不到任何 PDF, MD 或 TXT 檔案！請確認路徑。")
            return

        for filename in self.progress(raw_files, desc="解析檔案"):
            base_name = filename.rsplit(".", 1)[0]
            ext = filename.rsplit(".", 1)[1].lower()
            txt_path = self.config.processed_dir / f"{base_name}.txt"
            json_meta_path = self.config.raw_dir / f"{base_name}.json"

            if txt_path.exists() and not rebuild:
                continue

            metadata_part = ""
            body_part = ""

            if json_meta_path.exists():
                try:
                    with json_meta_path.open("r", encoding="utf-8") as f:
                        m = json.load(f)
                        metadata_part += f"Title: {m.get('title', '')}\n"
                        metadata_part += f"Year: {m.get('year', '')}\n"
                        metadata_part += f"ID: {m.get('paper_id', '')}\n"
                        metadata_part += f"Abstract: {m.get('abstract', '')}\n"
                        metadata_part += "--- CONTENT_START ---\n"
                except Exception:
                    pass
            else:
                print(f"找不到對應的 JSON 描述檔: {base_name}.json")

            try:
                if ext == "pdf":
                    from pypdf import PdfReader

                    reader = PdfReader(self.config.raw_dir / filename)
                    for page in reader.pages:
                        p_text = page.extract_text() or ""
                        if "References" in p_text:
                            body_part += p_text.split("References")[0]
                            break
                        body_part += p_text + "\n"
                else:
                    with (self.config.raw_dir / filename).open("r", encoding="utf-8") as f:
                        body_part = f.read()

                with txt_path.open("w", encoding="utf-8") as f:
                    f.write(metadata_part + self.clean_text(body_part))

            except Exception as e:
                print(f"無法解析 {filename}: {e}")

    def index_data(self, rebuild=False):
        if rebuild:
            print("重置 Vector DB...")
            with suppress(Exception):
                self.db_client.delete_collection(self.config.collection_name)
            self.collection = self.db_client.create_collection(self.config.collection_name)

        txt_files = [
            path.name
            for path in self.config.processed_dir.iterdir()
            if path.is_file() and path.suffix == ".txt"
        ]
        print(f"準備向量化 {len(txt_files)} 個處理後的文字檔...")

        for txt_file in self.progress(txt_files, desc="建立索引"):
            path = self.config.processed_dir / txt_file
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            meta, content_body, is_body = {"source": txt_file}, "", False
            for line in lines:
                if "--- CONTENT_START ---" in line:
                    is_body = True
                    continue
                if not is_body:
                    if line.startswith("Title: "):
                        meta["title"] = line[7:].strip()
                    elif line.startswith("Year: "):
                        meta["year"] = line[6:].strip()
                    elif line.startswith("ID: "):
                        meta["paper_id"] = line[4:].strip()
                    elif line.startswith("Abstract: "):
                        content_body += line[10:]
                else:
                    content_body += line

            chunks = self.text_splitter.split_text(content_body.strip())
            if not chunks:
                continue

            embeddings = self.model.encode(chunks).tolist()
            ids = [f"{txt_file}#c{i}" for i in range(len(chunks))]
            metadatas = [meta.copy() for _ in range(len(chunks))]
            self.collection.upsert(
                ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
            )

    def run(self, rebuild):
        self.config.validate_ingestion()
        self._initialize()
        self.process_content(rebuild)
        self.index_data(rebuild)
        print(f"完成！目前 DB 片段總數: {self.collection.count()}")

    def prepare_only(self) -> PreparationSummary:
        """Create structured JSONL chunks without loading embedding or vector clients."""
        self.config.validate_preparation()
        return prepare_corpus(
            manifest_path=self.config.corpus_manifest_path,
            project_root=self.config.project_root,
            raw_dir=self.config.raw_dir,
            output_path=self.config.processed_dir / PREPARED_FILENAME,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
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
        pipeline.run(rebuild=args.rebuild)
    except (ConfigurationError, PreparationError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
