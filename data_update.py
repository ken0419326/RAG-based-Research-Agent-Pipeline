import os, argparse, chromadb, shutil, json, re
from tqdm import tqdm
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- 設定 ---
RAW_DIR, PROCESSED_DIR = "data/raw", "data/processed"
CHROMA_PATH = "chroma_db"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

class DataUpdatePipeline:
    def __init__(self):
        print(f"📦 正在載入 Embedding 模型: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.db_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.db_client.get_or_create_collection(name="acl_research")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', '', text)  
        text = re.sub(r'\s+', ' ', text)     
        return text.strip()

    def process_content(self, rebuild=False):
        if rebuild:
            print(f"🧹正在清空 {PROCESSED_DIR}...")
            shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
            os.makedirs(PROCESSED_DIR, exist_ok=True)

        # 這裡印出路徑，確認程式看哪裡
        abs_raw_path = os.path.abspath(RAW_DIR)
        print(f"📂 正在檢查目錄: {abs_raw_path}")
        
        raw_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.pdf', '.md', '.txt'))]
        print(f"找到 {len(raw_files)} 個原始檔案: {raw_files}")

        if not raw_files:
            print("❌ 錯誤：在 data/raw 中找不到任何 PDF, MD 或 TXT 檔案！請確認路徑。")
            return

        for filename in tqdm(raw_files, desc="解析檔案"):
            base_name = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1].lower()
            txt_path = os.path.join(PROCESSED_DIR, f"{base_name}.txt")
            json_meta_path = os.path.join(RAW_DIR, f"{base_name}.json")
            
            if os.path.exists(txt_path) and not rebuild:
                continue

            metadata_part = ""
            body_part = ""
            
            # 偵測 JSON
            if os.path.exists(json_meta_path):
                try:
                    with open(json_meta_path, 'r', encoding='utf-8') as f:
                        m = json.load(f)
                        metadata_part += f"Title: {m.get('title', '')}\n"
                        metadata_part += f"Year: {m.get('year', '')}\n"
                        metadata_part += f"ID: {m.get('paper_id', '')}\n"
                        metadata_part += f"Abstract: {m.get('abstract', '')}\n"
                        metadata_part += "--- CONTENT_START ---\n"
                except: pass
            else:
                print(f"  ⚠️ 找不到對應的 JSON 描述檔: {base_name}.json")

            try:
                if ext == "pdf":
                    reader = PdfReader(os.path.join(RAW_DIR, filename))
                    for page in reader.pages:
                        p_text = page.extract_text() or ""
                        if "References" in p_text:
                            body_part += p_text.split("References")[0]; break
                        body_part += p_text + "\n"
                else:
                    with open(os.path.join(RAW_DIR, filename), 'r', encoding='utf-8') as f:
                        body_part = f.read()
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(metadata_part + self.clean_text(body_part))
                    
            except Exception as e:
                print(f"❌ 無法解析 {filename}: {e}")

    def index_data(self, rebuild=False):
        if rebuild:
            print("🔄 重置 Vector DB...")
            try: self.db_client.delete_collection("acl_research")
            except: pass
            self.collection = self.db_client.create_collection("acl_research")

        txt_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.txt')]
        print(f"📝 準備向量化 {len(txt_files)} 個處理後的文字檔...")

        for txt_file in tqdm(txt_files, desc="建立索引"):
            path = os.path.join(PROCESSED_DIR, txt_file)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            meta, content_body, is_body = {"source": txt_file}, "", False
            for l in lines:
                if "--- CONTENT_START ---" in l:
                    is_body = True
                    continue
                if not is_body:
                    if l.startswith("Title: "): meta["title"] = l[7:].strip()
                    elif l.startswith("Year: "): meta["year"] = l[6:].strip()
                    elif l.startswith("ID: "): meta["paper_id"] = l[4:].strip()
                    elif l.startswith("Abstract: "): content_body += l[10:] 
                else:
                    content_body += l

            chunks = self.text_splitter.split_text(content_body.strip())
            if not chunks: continue
            
            embeddings = self.model.encode(chunks).tolist()
            ids = [f"{txt_file}#c{i}" for i in range(len(chunks))]
            metadatas = [meta.copy() for _ in range(len(chunks))]
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

    def run(self, rebuild):
        self.process_content(rebuild)
        self.index_data(rebuild)
        print(f"🚀 完成！目前 DB 片段總數: {self.collection.count()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    DataUpdatePipeline().run(rebuild=args.rebuild)