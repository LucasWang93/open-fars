"""Fixed prompts for each agent role. All expect strict JSON output."""

# ---------------------------------------------------------------------------
# Method Unit Extraction (Knowledge Base)
# ---------------------------------------------------------------------------

METHOD_EXTRACTION_SYSTEM = (
    "You are a research methodology analyst. "
    "Extract concrete methodological components from the given paper text. "
    "Focus on algorithmic steps, training strategies, hyperparameter choices, "
    "regularization techniques, and experiment design decisions. "
    "Reply ONLY with JSON matching the schema."
)

METHOD_EXTRACTION_USER = """\
Paper: {paper_title}

Text:
{paper_text}

{categories_hint}

Extract up to {max_units} method units. For each, output:
- name: concise method name (e.g. "cosine learning rate decay")
- category: one of [lr_schedule, regularization, architecture, data_augmentation, \
training_strategy, hyperparameter, optimization, other]
- description: 1-2 sentence description of what it does
- inputs: list of parameters/configs it affects (e.g. ["learning_rate", "warmup_steps"])
- outputs: list of expected effects (e.g. ["faster convergence", "lower eval_loss"])
- confidence: float 0-1, how clearly this method is described in the paper

Output JSON:
{{
  "method_units": [
    {{
      "name": "...",
      "category": "...",
      "description": "...",
      "inputs": ["..."],
      "outputs": ["..."],
      "confidence": 0.8
    }}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Enhanced Ideation (Pattern-Driven)
# ---------------------------------------------------------------------------

IDEATION_ENHANCED_SYSTEM = (
    "You are a research ideation assistant with access to a knowledge base "
    "of research patterns extracted from recent papers. "
    "Generate novel, evidence-grounded hypotheses. "
    "Reply ONLY with JSON matching the schema."
)

IDEATION_ENHANCED_USER = """\
Task space: {taskspace_name}
Baseline metric: {primary_metric} (lower is better: {lower_is_better})

## Past experiments (DO NOT repeat these combinations):
{history_summary}

## Relevant research patterns from literature:
{top_patterns}

## Available actions:
{actions_list}

Requirements:
1. Pick 1-{max_actions} actions that form a NOVEL combination (not tried before)
2. Ground your hypothesis in at least one research pattern above
3. Explain WHY this combination should work based on evidence

Output JSON:
{{
  "actions": ["action_id", ...],
  "hypothesis": "evidence-grounded hypothesis sentence",
  "pattern_id": "the pattern ID that inspired this idea",
  "rationale": "why this combination should work based on the pattern",
  "novelty_note": "how this differs from past experiments"
}}
"""

# ---------------------------------------------------------------------------
# Original Ideation / Planning / Writing (unchanged below)
# ---------------------------------------------------------------------------

IDEATION_SYSTEM = (
    "You are a research ideation assistant. "
    "Given a task space with available actions, pick 1-2 actions to form a hypothesis. "
    "Reply ONLY with JSON matching the schema."
)

IDEATION_USER = """\
Task space: {taskspace_name}
Baseline metric: {primary_metric}
Available actions:
{actions_list}

Pick 1-{max_actions} actions and form a hypothesis.
Output JSON:
{{
  "actions": ["action_id", ...],
  "hypothesis": "one sentence hypothesis"
}}
"""

PLANNING_SYSTEM = (
    "You are a research planning assistant. "
    "Given a hypothesis and actions, produce a structured experiment plan. "
    "Reply ONLY with JSON matching the schema."
)

PLANNING_USER = """\
Hypothesis: {hypothesis}
Actions chosen: {actions}
Baseline config: {baseline_config}
Seeds: {seeds}

Output JSON:
{{
  "plan_summary": "short description of the experiment plan",
  "variables": ["variable names being changed"],
  "control": "baseline description",
  "treatment": "what changes",
  "metric": "{primary_metric}",
  "budget_estimate_minutes": <int>
}}
"""

WRITING_SYSTEM = (
    "You are a scientific writing assistant. "
    "Write a short paper section based on experiment results. "
    "Only cite facts from the provided data. Reply in markdown."
)

WRITING_USER = """\
Project: {project_id}
Hypothesis: {hypothesis}
Analysis:
{analysis}

Summary CSV:
{summary_csv}

Write a short paper with sections: ## Introduction, ## Method, ## Results, ## Limitations, ## Conclusion
Include references to fig1.png where appropriate.
"""
