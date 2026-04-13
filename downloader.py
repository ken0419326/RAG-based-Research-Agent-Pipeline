import os, time, requests, json
from acl_anthology import Anthology

# --- 設定 ---
KEYWORDS = ["empathy", "empathetic", "value", "emotional support", "emotion recognition", "value alignment"]
START_YEAR = 2025
TARGET_COUNT = 30
RAW_DIR = "data/raw" # 爬蟲只負責把原始物料丟進 raw

os.makedirs(RAW_DIR, exist_ok=True)

def run_downloader():
    print("正在初始化 ACL Anthology 數據...")
    # 若本地沒有資料，這會從 GitHub 下載索引，可能需要一點時間
    anthology = Anthology.from_repo()
    found = 0

    print(f"開始掃描 {START_YEAR} 年後的論文...")

    for paper in anthology.papers():
        # 1. 年份篩選
        try:
            if int(paper.year) < START_YEAR: continue
        except: continue

        # 2. 關鍵字篩選
        title_str = str(paper.title)
        title_lower = title_str.lower()
        
        if any(k in title_lower for k in KEYWORDS):
            paper_id = paper.full_id
            pdf_url = f"https://aclanthology.org/{paper_id}.pdf"
            
            # 檔名清理
            clean_t = "".join(x for x in title_str if x.isalnum() or x==" ").replace(" ", "_")[:50]
            filename = f"{paper.year}_{paper_id}_{clean_t}"
            
            pdf_path = os.path.join(RAW_DIR, f"{filename}.pdf")
            json_path = os.path.join(RAW_DIR, f"{filename}.json") # 關鍵：將元數據存為 JSON
            
            download_triggered = False

            # 3. 下載 PDF (如果不存在)
            if not os.path.exists(pdf_path):
                try:
                    r = requests.get(pdf_url, timeout=15)
                    if r.status_code == 200:
                        with open(pdf_path, 'wb') as f: 
                            f.write(r.content)
                        print(f"[{found+1}] PDF 下載成功: {paper_id}")
                        download_triggered = True
                except Exception as e:
                    print(f"PDF 下載失敗 {paper_id}: {e}")
            else:
                download_triggered = True # 已存在視為成功

            # 4. 儲存 Metadata 與 Abstract (如果不存在)
            # 即使 PDF 失敗，有 Abstract 也能做基礎 RAG
            if not os.path.exists(json_path):
                abstract_text = paper.abstract.as_text() if paper.abstract else ""
                metadata = {
                    "title": title_str,
                    "year": paper.year,
                    "paper_id": paper_id,
                    "abstract": abstract_text,
                    "url": pdf_url
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                if not abstract_text:
                    print(f"  ⚠️ 警告: {paper_id} 缺乏摘要")
            
            # 5. 計數與延遲
            found += 1
            time.sleep(0.5) # 稍微縮短延遲

            if found >= TARGET_COUNT: 
                break

    print(f"\n✅ 任務完成！")
    print(f"原始檔案路徑: {RAW_DIR}")
    print(f"共計處理: {found} 份論文原始資料 (PDF + JSON)。")

if __name__ == "__main__":
    run_downloader()