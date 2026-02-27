"""Extract usable text from papers. Uses abstract as primary source,
with optional full-text fetch from arXiv HTML."""

import re
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

from ..utils.log import get_logger

LOGGER = get_logger(__name__)


def _fetch_arxiv_html(arxiv_id: str) -> Optional[str]:
    """Try to fetch the HTML version of an arXiv paper."""
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        req = Request(url, headers={"User-Agent": "FARS-KB/1.0"})
        with urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError):
        pass
    return None


def _html_to_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text(paper: dict, save_dir: Optional[Path] = None) -> str:
    """Extract text for a single paper. Returns combined title + abstract + body."""
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    if save_dir:
        cached = save_dir / f"{paper_id.replace('/', '_')}.txt"
        if cached.exists():
            return cached.read_text()

    parts = [f"Title: {title}", f"Abstract: {abstract}"]

    html = _fetch_arxiv_html(paper_id)
    if html and len(html) > 5000:
        body = _html_to_text(html)
        if len(body) > len(abstract) * 2:
            body = body[:15000]
            parts.append(f"Body: {body}")
            LOGGER.info("fetched full text for %s (%d chars)", paper_id, len(body))
    else:
        LOGGER.info("using abstract only for %s", paper_id)

    text = "\n\n".join(parts)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        cached = save_dir / f"{paper_id.replace('/', '_')}.txt"
        cached.write_text(text)

    return text


def extract_all(papers: list[dict], save_dir: Path, rate_limit: float = 1.0) -> dict[str, str]:
    """Extract text for all papers with rate limiting. Returns {paper_id: text}."""
    results = {}
    for i, paper in enumerate(papers):
        pid = paper.get("paper_id", "")
        if not pid:
            continue

        text = extract_text(paper, save_dir=save_dir)
        results[pid] = text

        if i > 0 and i % 10 == 0:
            LOGGER.info("extracted %d/%d papers", i, len(papers))
        if rate_limit > 0:
            time.sleep(rate_limit)

    LOGGER.info("text extraction complete: %d papers", len(results))
    return results
