"""Fixed prompts for each agent role. All expect strict JSON output."""

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
