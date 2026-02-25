"""JSON schemas for agent outputs (used in gate validation)."""

IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "hypothesis": {"type": "string"},
    },
    "required": ["actions", "hypothesis"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_summary": {"type": "string"},
        "variables": {"type": "array", "items": {"type": "string"}},
        "control": {"type": "string"},
        "treatment": {"type": "string"},
        "metric": {"type": "string"},
        "budget_estimate_minutes": {"type": "integer"},
    },
    "required": ["plan_summary", "variables", "control", "treatment", "metric"],
}

METRICS_SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "primary_metric": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
                "higher_is_better": {"type": "boolean"},
            },
            "required": ["name", "value", "higher_is_better"],
        },
        "secondary_metrics": {"type": "object"},
        "seed": {"type": "integer"},
        "config_hash": {"type": "string"},
        "status": {"type": "string", "enum": ["SUCCESS", "FAIL"]},
    },
    "required": ["run_id", "primary_metric", "status"],
}
