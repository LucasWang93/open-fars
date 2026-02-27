"""Extract method units from paper text using GPT-4o."""

import hashlib
import json
from typing import Optional

from ..llm.router import get_router
from ..llm.prompts import METHOD_EXTRACTION_SYSTEM, METHOD_EXTRACTION_USER
from ..utils.log import get_logger
from .schemas import MethodUnit

LOGGER = get_logger(__name__)


def _make_unit_id(name: str, paper_id: str) -> str:
    raw = f"{name}::{paper_id}"
    return "mu_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def extract_method_units(
    paper_id: str,
    paper_title: str,
    text: str,
    max_units: int = 10,
    categories: Optional[list[str]] = None,
) -> list[MethodUnit]:
    """Use GPT-4o to extract method units from a paper's text."""
    if len(text) > 12000:
        text = text[:12000]

    cat_hint = ""
    if categories:
        cat_hint = "Categories to focus on: " + ", ".join(categories)

    user_prompt = METHOD_EXTRACTION_USER.format(
        paper_title=paper_title,
        paper_text=text,
        max_units=max_units,
        categories_hint=cat_hint,
    )

    router = get_router("azure_gpt4o")
    raw = router.generate(METHOD_EXTRACTION_SYSTEM, user_prompt, json_mode=True)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        LOGGER.error("method extraction returned non-JSON for %s", paper_id)
        return []

    units_data = data.get("method_units", data.get("methods", []))
    if isinstance(units_data, dict):
        units_data = [units_data]

    units = []
    for item in units_data[:max_units]:
        try:
            unit = MethodUnit(
                unit_id=_make_unit_id(item.get("name", ""), paper_id),
                name=item.get("name", "unknown"),
                category=item.get("category", "other"),
                description=item.get("description", ""),
                inputs=item.get("inputs", []),
                outputs=item.get("outputs", []),
                paper_source=paper_id,
                paper_title=paper_title,
                confidence=item.get("confidence", 0.7),
            )
            units.append(unit)
        except Exception as exc:
            LOGGER.warning("skipping malformed unit from %s: %s", paper_id, exc)

    LOGGER.info("extracted %d method units from %s", len(units), paper_id)
    return units


def extract_all(
    papers: list[dict],
    texts: dict[str, str],
    max_units_per_paper: int = 10,
    categories: Optional[list[str]] = None,
) -> list[MethodUnit]:
    """Extract method units from all papers."""
    all_units = []
    for paper in papers:
        pid = paper.get("paper_id", "")
        text = texts.get(pid, "")
        if not text or len(text) < 100:
            continue

        units = extract_method_units(
            paper_id=pid,
            paper_title=paper.get("title", ""),
            text=text,
            max_units=max_units_per_paper,
            categories=categories,
        )
        all_units.extend(units)

    LOGGER.info("total method units extracted: %d from %d papers", len(all_units), len(papers))
    return all_units
