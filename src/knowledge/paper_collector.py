"""Collect papers from HuggingFace Daily Papers API and arXiv search."""

import json
import re
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote_plus

from ..utils.log import get_logger

LOGGER = get_logger(__name__)

HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
ARXIV_SEARCH_URL = "https://export.arxiv.org/api/query"


def _http_get(url: str, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "FARS-KB/1.0"})
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except (URLError, TimeoutError) as exc:
            LOGGER.warning("HTTP GET %s attempt %d failed: %s", url, attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def collect_hf_daily(max_papers: int = 100) -> list[dict]:
    """Fetch recent papers from HuggingFace daily papers API."""
    raw = _http_get(HF_DAILY_URL)
    if not raw:
        LOGGER.error("failed to fetch HF daily papers")
        return []

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        LOGGER.error("HF daily papers response is not valid JSON")
        return []

    papers = []
    for item in items[:max_papers]:
        paper = item.get("paper", item)
        entry = {
            "paper_id": paper.get("id", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("summary", paper.get("abstract", "")),
            "authors": [a.get("name", a) if isinstance(a, dict) else str(a)
                        for a in paper.get("authors", [])],
            "url": "https://arxiv.org/abs/" + paper.get("id", ""),
            "date": paper.get("publishedAt", paper.get("published", "")),
            "source": "huggingface_daily",
        }
        if entry["paper_id"] and entry["title"]:
            papers.append(entry)

    LOGGER.info("collected %d papers from HF daily", len(papers))
    return papers


def collect_arxiv(keywords: list[str], max_papers: int = 100) -> list[dict]:
    """Search arXiv for papers matching keywords."""
    papers = []
    per_keyword = max(10, max_papers // len(keywords)) if keywords else 0

    for kw in keywords:
        query = quote_plus(f'all:"{kw}"')
        url = f"{ARXIV_SEARCH_URL}?search_query={query}&start=0&max_results={per_keyword}&sortBy=submittedDate&sortOrder=descending"
        raw = _http_get(url)
        if not raw:
            continue

        for entry_match in re.finditer(r"<entry>(.*?)</entry>", raw, re.DOTALL):
            block = entry_match.group(1)

            arxiv_id_m = re.search(r"<id>.*?/abs/([^<]+)</id>", block)
            title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL)
            summary_m = re.search(r"<summary>(.*?)</summary>", block, re.DOTALL)
            published_m = re.search(r"<published>(.*?)</published>", block)

            authors = re.findall(r"<name>(.*?)</name>", block)

            arxiv_id = arxiv_id_m.group(1).strip() if arxiv_id_m else ""
            title = title_m.group(1).strip().replace("\n", " ") if title_m else ""
            abstract = summary_m.group(1).strip().replace("\n", " ") if summary_m else ""
            published = published_m.group(1).strip() if published_m else ""

            if not arxiv_id or not title:
                continue

            papers.append({
                "paper_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "date": published,
                "source": "arxiv_search",
                "keyword": kw,
            })

        time.sleep(1)

    seen = set()
    deduped = []
    for p in papers:
        if p["paper_id"] not in seen:
            seen.add(p["paper_id"])
            deduped.append(p)

    LOGGER.info("collected %d unique papers from arXiv (%d keywords)", len(deduped), len(keywords))
    return deduped[:max_papers]


def collect_papers(
    keywords: list[str],
    max_papers: int = 200,
    pool_dir: Optional[Path] = None,
) -> list[dict]:
    """Collect papers from all sources, deduplicate, and save to pool."""
    hf_papers = collect_hf_daily(max_papers=max_papers // 2)
    arxiv_papers = collect_arxiv(keywords, max_papers=max_papers // 2)

    seen = set()
    all_papers = []
    for p in hf_papers + arxiv_papers:
        pid = p["paper_id"]
        if pid not in seen:
            seen.add(pid)
            all_papers.append(p)

    LOGGER.info("total unique papers collected: %d", len(all_papers))

    if pool_dir:
        pool_dir.mkdir(parents=True, exist_ok=True)
        index_path = pool_dir / "metadata.json"

        existing = []
        if index_path.exists():
            try:
                existing = json.loads(index_path.read_text())
            except json.JSONDecodeError:
                existing = []

        existing_ids = {p["paper_id"] for p in existing}
        new_papers = [p for p in all_papers if p["paper_id"] not in existing_ids]
        merged = existing + new_papers

        index_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        LOGGER.info("saved %d papers to %s (%d new)", len(merged), index_path, len(new_papers))

    return all_papers
