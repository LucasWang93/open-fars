"""Pydantic schemas for knowledge base entities."""

from typing import Optional
from pydantic import BaseModel, Field


class MethodUnit(BaseModel):
    unit_id: str = ""
    name: str
    category: str
    description: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    paper_source: str = ""
    paper_title: str = ""
    confidence: float = 0.8


class MethodRelation(BaseModel):
    from_id: str
    to_id: str
    relation_type: str  # "co_occurs_with", "improves_upon", "complementary_to"
    weight: float = 1.0
    paper_source: str = ""


class ResearchPattern(BaseModel):
    pattern_id: str = ""
    components: list[str] = Field(default_factory=list)
    component_names: list[str] = Field(default_factory=list)
    expected_benefit: str = ""
    evidence_count: int = 0
    quality_score: float = 0.0
    mappable_actions: list[str] = Field(default_factory=list)
    source_papers: list[str] = Field(default_factory=list)


class ExperimentRecord(BaseModel):
    project_id: str
    pattern_id: Optional[str] = None
    actions: list[str] = Field(default_factory=list)
    hypothesis: str = ""
    outcome: str = ""  # "success", "negative", "failed"
    eval_loss: Optional[float] = None
    timestamp: str = ""
