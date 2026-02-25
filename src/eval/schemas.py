"""Pydantic schemas for metrics validation."""

from typing import Optional
from pydantic import BaseModel


class PrimaryMetric(BaseModel):
    name: str
    value: float
    higher_is_better: bool


class RunMetrics(BaseModel):
    run_id: str
    group: str = "unknown"
    primary_metric: PrimaryMetric
    secondary_metrics: dict = {}
    seed: int = 0
    config_hash: str = ""
    status: str = "SUCCESS"
