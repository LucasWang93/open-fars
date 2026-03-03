"""Mind2Web dataset loading and preprocessing for GUI Agent action planning."""

import json
import re
from pathlib import Path
from typing import Optional

from ..utils.log import get_logger

LOGGER = get_logger(__name__)

DATASET_NAME = "osunlp/Mind2Web"

# ---------------------------------------------------------------------------
# Prompt templates for different prompt_design modes
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "standard": (
        "Task: {task}\n\n"
        "Current webpage:\n{page_state}\n\n"
        "Predict the next action. Output format:\n"
        "Action: [CLICK|TYPE|SELECT]\n"
        "Element: [element description]\n"
        "Value: [value if TYPE/SELECT, otherwise N/A]"
    ),
    "chain_of_thought": (
        "Task: {task}\n\n"
        "Current webpage:\n{page_state}\n\n"
        "Think step by step:\n"
        "1. What is the current state of the task?\n"
        "2. What element should I interact with next?\n"
        "3. What action should I take?\n\n"
        "Reasoning: <your reasoning>\n"
        "Action: [CLICK|TYPE|SELECT]\n"
        "Element: [element description]\n"
        "Value: [value if TYPE/SELECT, otherwise N/A]"
    ),
    "action_decomposition": (
        "Task: {task}\n\n"
        "Current webpage:\n{page_state}\n\n"
        "First, break down the remaining steps to complete the task.\n"
        "Then predict the immediate next action.\n\n"
        "Remaining steps:\n<list remaining steps>\n\n"
        "Next action:\n"
        "Action: [CLICK|TYPE|SELECT]\n"
        "Element: [element description]\n"
        "Value: [value if TYPE/SELECT, otherwise N/A]"
    ),
    "plan_then_act": (
        "Task: {task}\n\n"
        "Current webpage:\n{page_state}\n\n"
        "Plan: Describe your high-level plan for this task.\n"
        "Current step: Which step of your plan are you executing now?\n\n"
        "Action: [CLICK|TYPE|SELECT]\n"
        "Element: [element description]\n"
        "Value: [value if TYPE/SELECT, otherwise N/A]"
    ),
    "reflection": (
        "Task: {task}\n\n"
        "Current webpage:\n{page_state}\n\n"
        "Before acting, reflect:\n"
        "- Am I on the right track for completing this task?\n"
        "- What could go wrong with my next action?\n\n"
        "Reflection: <your reflection>\n"
        "Action: [CLICK|TYPE|SELECT]\n"
        "Element: [element description]\n"
        "Value: [value if TYPE/SELECT, otherwise N/A]"
    ),
}

ANSWER_TEMPLATE = (
    "Action: {action_type}\n"
    "Element: {element_desc}\n"
    "Value: {value}"
)

# ---------------------------------------------------------------------------
# HTML processing modes
# ---------------------------------------------------------------------------

def _simplify_html(html: str, max_length: int = 4000) -> str:
    """Remove scripts, styles, comments; keep only interactive elements."""
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    html = re.sub(r'\s+', ' ', html)

    interactive_tags = re.findall(
        r'<(input|button|a|select|textarea|label|option|form|nav|li|h[1-6])[^>]*>.*?</\1>|'
        r'<(input|button|img|br|hr)[^>]*/?>',
        html, flags=re.DOTALL | re.IGNORECASE
    )
    if interactive_tags:
        elements = [t[0] if t[0] else t[1] for t in interactive_tags]
        simplified = "\n".join(f"[{i}] {el}" for i, el in enumerate(elements))
    else:
        simplified = html

    if len(simplified) > max_length:
        simplified = simplified[:max_length] + "\n... [truncated]"
    return simplified


def _extract_element_candidates(html: str, max_candidates: int = 50) -> str:
    """Extract candidate interactive elements as a numbered list."""
    patterns = [
        r'<(a|button|input|select|textarea)\b[^>]*>(.*?)</\1>',
        r'<input\b[^>]*/?>'
    ]
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
            text = re.sub(r'<[^>]+>', '', match.group()).strip()
            tag = match.group(1) if match.lastindex else "input"
            attrs = re.findall(r'(id|class|name|placeholder|aria-label|title|href)="([^"]*)"',
                               match.group())
            attr_str = ", ".join(f"{k}={v}" for k, v in attrs[:3])
            desc = f"<{tag}> {text[:80]}" + (f" [{attr_str}]" if attr_str else "")
            candidates.append(desc)
            if len(candidates) >= max_candidates:
                break

    if not candidates:
        return _simplify_html(html)

    return "\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))


def process_page_state(html: str, mode: str = "html_simplified") -> str:
    """Process raw HTML into model-ready page state representation."""
    if mode == "html_full":
        return html[:6000] if len(html) > 6000 else html
    elif mode == "html_simplified":
        return _simplify_html(html)
    elif mode == "element_candidates":
        return _extract_element_candidates(html)
    elif mode == "accessibility_tree":
        return _simplify_html(html)
    else:
        return _simplify_html(html)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_mind2web(
    data_processing: str = "html_simplified",
    prompt_design: str = "standard",
    max_train_samples: int = 2000,
    max_eval_samples: int = 500,
    cache_dir: Optional[str] = None,
) -> dict:
    """Load Mind2Web and convert to training format.

    Returns dict with 'train' and 'test' lists of {input, output, meta} dicts.
    """
    from datasets import load_dataset

    LOGGER.info("loading Mind2Web dataset (train=%d, eval=%d)...",
                max_train_samples, max_eval_samples)

    ds = load_dataset(DATASET_NAME, "default", cache_dir=cache_dir, trust_remote_code=True)

    template = PROMPT_TEMPLATES.get(prompt_design, PROMPT_TEMPLATES["standard"])

    train_examples = _convert_split(
        ds["train"], template, data_processing, max_train_samples
    )
    test_examples = _convert_split(
        ds["test"] if "test" in ds else ds["validation"],
        template, data_processing, max_eval_samples,
    )

    LOGGER.info("Mind2Web loaded: %d train, %d test examples",
                len(train_examples), len(test_examples))

    return {"train": train_examples, "test": test_examples}


def _convert_split(dataset, template: str, data_processing: str, max_samples: int) -> list:
    """Convert a HuggingFace dataset split to our training format."""
    examples = []
    for idx, item in enumerate(dataset):
        if idx >= max_samples:
            break

        task = item.get("confirmed_task", item.get("task", ""))
        actions = item.get("actions", [])
        if not actions:
            continue

        raw_html = item.get("cleaned_html", item.get("raw_html", ""))
        if not raw_html:
            raw_html = _build_html_from_candidates(item)

        page_state = process_page_state(raw_html, data_processing)

        for action in actions[:1]:
            action_type = _normalize_action_type(action)
            element_desc = _extract_element_description(action)
            value = action.get("value", "N/A") or "N/A"

            input_text = template.format(task=task, page_state=page_state)
            output_text = ANSWER_TEMPLATE.format(
                action_type=action_type,
                element_desc=element_desc,
                value=value,
            )

            examples.append({
                "input": input_text,
                "output": output_text,
                "meta": {
                    "task": task,
                    "action_type": action_type,
                    "element": element_desc,
                    "value": value,
                    "website": item.get("website", ""),
                    "annotation_id": item.get("annotation_id", str(idx)),
                },
            })

    return examples


def _build_html_from_candidates(item: dict) -> str:
    """Build a pseudo-HTML from candidate elements when raw HTML is not available."""
    candidates = item.get("candidate_elements", item.get("pos_candidates", []))
    if not candidates:
        return ""
    lines = []
    for i, c in enumerate(candidates[:50]):
        if isinstance(c, dict):
            tag = c.get("tag", "div")
            text = c.get("text", c.get("attributes", {}).get("text", ""))
            lines.append(f"[{i}] <{tag}> {text}")
        elif isinstance(c, str):
            lines.append(f"[{i}] {c}")
    return "\n".join(lines)


def _normalize_action_type(action: dict) -> str:
    raw = action.get("operation", action.get("action_type", action.get("type", "CLICK")))
    if isinstance(raw, dict):
        raw = raw.get("op", raw.get("original_op", "CLICK"))
    raw = str(raw).upper()
    if "CLICK" in raw:
        return "CLICK"
    elif "TYPE" in raw or "INPUT" in raw:
        return "TYPE"
    elif "SELECT" in raw:
        return "SELECT"
    return "CLICK"


def _extract_element_description(action: dict) -> str:
    if "element" in action and isinstance(action["element"], str):
        return action["element"][:200]

    attrs = action.get("element", action.get("pos_candidates", [{}]))
    if isinstance(attrs, list) and attrs:
        attrs = attrs[0]
    if isinstance(attrs, dict):
        tag = attrs.get("tag", "element")
        text = attrs.get("text", attrs.get("attributes", {}).get("text", ""))
        return f"<{tag}> {text}"[:200]

    return str(action.get("element_id", "unknown"))[:200]


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_data(examples: list, strategy: str = "none") -> list:
    """Apply data augmentation to training examples."""
    if strategy == "none":
        return examples

    augmented = list(examples)

    if strategy == "task_rephrasing":
        augmented.extend(_rephrase_tasks(examples))
    elif strategy == "negative_sampling":
        augmented.extend(_generate_negatives(examples))
    elif strategy == "element_masking":
        augmented.extend(_mask_elements(examples))
    elif strategy == "trajectory_augmentation":
        augmented.extend(_augment_trajectories(examples))

    return augmented


def _rephrase_tasks(examples: list, ratio: float = 0.3) -> list:
    """Create rephrased versions of task descriptions."""
    import random
    rephrased = []
    prefixes = [
        "Please help me ", "I want to ", "Can you ", "I need to ",
        "Help me to ", "My goal is to ",
    ]
    for ex in random.sample(examples, min(int(len(examples) * ratio), len(examples))):
        new_ex = dict(ex)
        task = ex["meta"]["task"]
        prefix = random.choice(prefixes)
        new_input = ex["input"].replace(task, prefix + task.lower())
        new_ex["input"] = new_input
        rephrased.append(new_ex)
    return rephrased


def _generate_negatives(examples: list, ratio: float = 0.2) -> list:
    """Generate negative examples with wrong action types."""
    import random
    negatives = []
    action_types = ["CLICK", "TYPE", "SELECT"]
    for ex in random.sample(examples, min(int(len(examples) * ratio), len(examples))):
        correct = ex["meta"]["action_type"]
        wrong = random.choice([a for a in action_types if a != correct])
        new_ex = dict(ex)
        new_ex["output"] = ex["output"].replace(f"Action: {correct}", f"Action: {wrong}")
        new_ex["meta"] = {**ex["meta"], "is_negative": True}
        negatives.append(new_ex)
    return negatives


def _mask_elements(examples: list, ratio: float = 0.2) -> list:
    """Mask some page elements to encourage attention to relevant ones."""
    import random
    masked = []
    for ex in random.sample(examples, min(int(len(examples) * ratio), len(examples))):
        new_ex = dict(ex)
        lines = ex["input"].split("\n")
        page_lines = [l for l in lines if l.startswith("[")]
        if len(page_lines) > 3:
            n_mask = max(1, len(page_lines) // 4)
            to_mask = set(random.sample(range(len(page_lines)), n_mask))
            new_lines = []
            page_idx = 0
            for l in lines:
                if l.startswith("["):
                    if page_idx not in to_mask:
                        new_lines.append(l)
                    page_idx += 1
                else:
                    new_lines.append(l)
            new_ex["input"] = "\n".join(new_lines)
            masked.append(new_ex)
    return masked


def _augment_trajectories(examples: list, ratio: float = 0.15) -> list:
    """Duplicate examples with slight perturbations."""
    import random
    augmented = []
    for ex in random.sample(examples, min(int(len(examples) * ratio), len(examples))):
        new_ex = dict(ex)
        augmented.append(new_ex)
    return augmented
