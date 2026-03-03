"""Fixed prompts for each agent role. All expect strict JSON output."""

# ---------------------------------------------------------------------------
# Method Unit Extraction (Knowledge Base)
# ---------------------------------------------------------------------------

METHOD_EXTRACTION_SYSTEM = (
    "You are a research methodology analyst specializing in GUI agents and web navigation. "
    "Extract concrete methodological components from the given paper text. "
    "Focus on action prediction strategies, grounding techniques, training methods, "
    "prompt designs, data augmentation, and evaluation approaches. "
    "Reply ONLY with JSON matching the schema."
)

METHOD_EXTRACTION_USER = """\
Paper: {paper_title}

Text:
{paper_text}

{categories_hint}

Extract up to {max_units} method units. For each, output:
- name: concise method name (e.g. "chain-of-thought action decomposition")
- category: one of [grounding, action_prediction, training_strategy, prompt_design, \
data_augmentation, evaluation, architecture, other]
- description: 1-2 sentence description of what it does
- inputs: list of parameters/configs it affects (e.g. ["prompt_template", "html_parsing"])
- outputs: list of expected effects (e.g. ["higher element accuracy", "better action F1"])
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
# Enhanced Ideation (Free-form Research Proposal)
# ---------------------------------------------------------------------------

IDEATION_ENHANCED_SYSTEM = (
    "You are a research scientist specializing in GUI agents for web navigation. "
    "You propose novel, evidence-grounded research ideas for improving action planning. "
    "Your ideas should be specific, executable, and testable on the Mind2Web benchmark. "
    "Reply ONLY with JSON matching the schema."
)

IDEATION_ENHANCED_USER = """\
## Research Context
Domain: GUI Agent — Action Planning for Web Navigation
Benchmark: Mind2Web (predict correct action + target element given task + webpage)
Base model: {base_model}
Primary metric: {primary_metric} (higher is better: {higher_is_better})

## Exploration Dimensions (you may innovate along any of these):
{dimensions_text}

## Baseline Configuration:
{baseline_text}

## Past Experiments (DO NOT repeat; learn from failures):
{history_summary}

## Relevant Research Patterns from Literature:
{top_patterns}

## Requirements:
1. Propose ONE novel research idea that differs from all past experiments
2. Ground your hypothesis in at least one research pattern from literature
3. Specify concrete choices for each exploration dimension
4. The method must be executable: fine-tune {base_model} on Mind2Web, evaluate with standard metrics
5. Explain WHY this approach should improve over the baseline
6. If past experiments failed, analyze WHY and avoid the same pitfalls

Output JSON:
{{
  "title": "short descriptive title of the research idea",
  "hypothesis": "one clear, testable hypothesis",
  "method": {{
    "training_strategy": "chosen strategy",
    "data_processing": "how to represent webpage state",
    "prompt_design": "how to format the task for the model",
    "model_config": "model/adapter configuration",
    "augmentation": "data augmentation strategy",
    "key_innovation": "1-2 sentences: what is novel about this approach"
  }},
  "config_hints": {{
    "learning_rate": <float>,
    "num_train_epochs": <int>,
    "lora_rank": <int>,
    "max_seq_length": <int>,
    "batch_size": <int>
  }},
  "pattern_id": "the pattern ID that inspired this idea (or null)",
  "rationale": "why this combination should work, citing evidence",
  "novelty_note": "how this differs from past experiments"
}}
"""

# ---------------------------------------------------------------------------
# Planning (Convert free-form idea to executable config)
# ---------------------------------------------------------------------------

PLANNING_SYSTEM = (
    "You are an ML experiment planning assistant. "
    "Convert a research idea into a precise, executable experiment configuration. "
    "The config must be directly loadable by the training pipeline. "
    "Reply ONLY with JSON matching the schema."
)

PLANNING_USER = """\
## Research Idea
Title: {title}
Hypothesis: {hypothesis}
Method: {method_json}
Config hints: {config_hints_json}

## Baseline Config (for reference):
{baseline_config}

## Constraints:
- Base model: {base_model}
- Benchmark: Mind2Web
- Seeds: {seeds}
- Max GPU hours: {max_gpu_hours}
- Max train samples: {max_train_samples}
- Max eval samples: {max_eval_samples}

## Your task:
Create a complete experiment config with baseline and treatment groups.
The treatment should implement the research idea's method.
Ensure all values are concrete (no placeholders).

Output JSON:
{{
  "plan_summary": "2-3 sentence description of what will be compared",
  "variables": ["list of variables being changed from baseline"],
  "baseline": {{
    "training_strategy": "vanilla_sft",
    "data_processing": "html_simplified",
    "prompt_design": "standard",
    "model_config": "lora_r16",
    "augmentation": "none",
    "train": {{
      "learning_rate": <float>,
      "num_train_epochs": <int>,
      "warmup_ratio": <float>,
      "per_device_train_batch_size": <int>,
      "max_seq_length": <int>,
      "gradient_accumulation_steps": <int>
    }},
    "lora": {{
      "rank": <int>,
      "alpha": <int>,
      "dropout": <float>,
      "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }},
    "data": {{
      "max_train_samples": <int>,
      "max_eval_samples": <int>
    }}
  }},
  "treatment": {{
    "training_strategy": "...",
    "data_processing": "...",
    "prompt_design": "...",
    "model_config": "...",
    "augmentation": "...",
    "train": {{ ... }},
    "lora": {{ ... }},
    "data": {{ ... }}
  }},
  "metric": "{primary_metric}",
  "budget_estimate_minutes": <int>
}}
"""

# ---------------------------------------------------------------------------
# REVISE (Failure Analysis + Idea Refinement)
# ---------------------------------------------------------------------------

REVISE_SYSTEM = (
    "You are a research debugging expert. Analyze why an experiment failed "
    "and propose a refined research idea that avoids the same pitfalls. "
    "Reply ONLY with JSON matching the ideation schema."
)

REVISE_USER = """\
## Original Research Idea
Title: {title}
Hypothesis: {hypothesis}
Method: {method_json}

## Failure History (most recent first):
{failure_history}

## Error Messages:
{error_messages}

## Requirements:
1. Analyze WHY the experiment failed
2. Propose a REVISED idea that keeps the core research direction
3. Simplify the approach to make it more robust
4. Specifically address each failure point
5. If resource errors (OOM), reduce model/data size
6. If logic errors (NaN, parse), use safer defaults

Output JSON (same schema as ideation):
{{
  "title": "revised title",
  "hypothesis": "revised hypothesis",
  "method": {{
    "training_strategy": "...",
    "data_processing": "...",
    "prompt_design": "...",
    "model_config": "...",
    "augmentation": "...",
    "key_innovation": "what changed from the original and why"
  }},
  "config_hints": {{
    "learning_rate": <float>,
    "num_train_epochs": <int>,
    "lora_rank": <int>,
    "max_seq_length": <int>,
    "batch_size": <int>
  }},
  "pattern_id": null,
  "rationale": "why the revised approach should work",
  "novelty_note": "what changed from the failed attempt"
}}
"""

# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

WRITING_SYSTEM = (
    "You are a scientific writing assistant specializing in GUI agent research. "
    "Write a short paper section based on experiment results. "
    "Only cite facts from the provided data. Reply in markdown."
)

WRITING_USER = """\
Project: {project_id}
Title: {title}
Hypothesis: {hypothesis}
Method: {method_description}

Analysis:
{analysis}

Summary CSV:
{summary_csv}

Write a short paper with sections: ## Introduction, ## Method, ## Results, ## Limitations, ## Conclusion
Include references to fig1.png where appropriate.
Focus on what was tested, what was found, and what it means for GUI agent research.
"""
