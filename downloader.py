import json
import time

from config import AppConfig

KEYWORDS = [
    "empathy",
    "empathetic",
    "value",
    "emotional support",
    "emotion recognition",
    "value alignment",
]
START_YEAR = 2025
TARGET_COUNT = 30


def run_downloader(config: AppConfig | None = None):
    config = config or AppConfig.load()
    raw_dir = config.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    import requests
    from acl_anthology import Anthology

    print("正在初始化 ACL Anthology 數據...")
    anthology = Anthology.from_repo()
    found = 0

    print(f"開始掃描 {START_YEAR} 年後的論文...")

    for paper in anthology.papers():
        try:
            if int(paper.year) < START_YEAR:
                continue
        except Exception:
            continue

        title_str = str(paper.title)
        title_lower = title_str.lower()

        if any(k in title_lower for k in KEYWORDS):
            paper_id = paper.full_id
            pdf_url = f"https://aclanthology.org/{paper_id}.pdf"

            clean_t = "".join(x for x in title_str if x.isalnum() or x == " ").replace(" ", "_")[
                :50
            ]
            filename = f"{paper.year}_{paper_id}_{clean_t}"

            pdf_path = raw_dir / f"{filename}.pdf"
            json_path = raw_dir / f"{filename}.json"

            if not pdf_path.exists():
                try:
                    r = requests.get(pdf_url, timeout=15)
                    if r.status_code == 200:
                        with pdf_path.open("wb") as f:
                            f.write(r.content)
                        print(f"[{found + 1}] PDF 下載成功: {paper_id}")
                except Exception as e:
                    print(f"PDF 下載失敗 {paper_id}: {e}")

            if not json_path.exists():
                abstract_text = paper.abstract.as_text() if paper.abstract else ""
                metadata = {
                    "title": title_str,
                    "year": paper.year,
                    "paper_id": paper_id,
                    "abstract": abstract_text,
                    "url": pdf_url,
                }
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                if not abstract_text:
                    print(f"警告: {paper_id} 缺乏摘要")

            found += 1
            time.sleep(0.5)

            if found >= TARGET_COUNT:
                break

    print(f"原始檔案路徑: {raw_dir}")
    print(f"共計處理: {found} 份論文原始資料 (PDF + JSON)。")


if __name__ == "__main__":
    run_downloader()
