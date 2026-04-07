"""
Graders for all 3 tasks.
Each grader returns a score in [0.0, 1.0] with a breakdown dict
and feedback string explaining why points were given or lost.
"""

from typing import Any, Dict, Tuple


def _contains_any(text: str, keywords: list) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ─── Task 1 Grader: Overfitting ────────────────────────────────────────────────

def grade_overfitting(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    diagnosis = (action_data.get("diagnosis") or "").strip()
    fix_type = (action_data.get("fix_type") or "").strip()
    fix_detail = (action_data.get("fix_detail") or "").strip()

    breakdown = {}
    feedback_parts = []

    # 0.5 pts — correctly identifies overfitting
    diag_keywords = ground_truth["diagnosis_keywords"]
    if _contains_any(diagnosis, diag_keywords):
        breakdown["diagnosis"] = 0.5
        feedback_parts.append("✅ Correctly identified overfitting")
    else:
        breakdown["diagnosis"] = 0.0
        feedback_parts.append("❌ Did not identify overfitting as the root cause")

    # 0.5 pts — valid fix
    fix_keywords = ground_truth["valid_fix_keywords"]
    valid_fix_types = ground_truth["valid_fix_types"]
    fix_score = 0.0

    if fix_type in valid_fix_types:
        fix_score += 0.2
    if _contains_any(fix_detail, fix_keywords):
        fix_score += 0.3

    breakdown["fix"] = round(fix_score, 3)
    if fix_score >= 0.4:
        feedback_parts.append("✅ Proposed a valid regularization fix")
    elif fix_score > 0:
        feedback_parts.append("⚠️ Fix partially correct — be more specific")
    else:
        feedback_parts.append("❌ Fix is incorrect or missing")

    total = round(breakdown["diagnosis"] + breakdown["fix"], 3)
    return total, breakdown, " | ".join(feedback_parts)


# ─── Task 2 Grader: LR Explosion ───────────────────────────────────────────────

def grade_lr_explosion(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    diagnosis = (action_data.get("diagnosis") or "").strip()
    fix_type = (action_data.get("fix_type") or "").strip()
    fix_detail = (action_data.get("fix_detail") or "").strip()

    breakdown = {}
    feedback_parts = []

    # 0.5 pts — identifies LR as root cause
    if _contains_any(diagnosis, ground_truth["diagnosis_keywords"]):
        breakdown["diagnosis"] = 0.5
        feedback_parts.append("✅ Identified learning rate / gradient explosion as root cause")
    else:
        breakdown["diagnosis"] = 0.0
        feedback_parts.append("❌ Did not identify learning rate as the problem")

    # 0.5 pts — correct fix
    fix_score = 0.0
    lr_range = ground_truth["valid_lr_range"]
    fix_keywords = ground_truth["valid_fix_keywords"]

    if fix_type in ground_truth["valid_fix_types"]:
        fix_score += 0.1

    if _contains_any(fix_detail, fix_keywords):
        fix_score += 0.2

    # Check if a numeric LR value in valid range is mentioned
    import re
    numbers = re.findall(r"[\d]+\.?[\d]*[eE]?[-+]?[\d]*", fix_detail)
    for num_str in numbers:
        try:
            val = float(num_str)
            if lr_range[0] <= val <= lr_range[1]:
                fix_score += 0.2
                feedback_parts.append(f"✅ Suggested LR value {val} is in valid range")
                break
        except ValueError:
            pass
    else:
        if fix_score > 0:
            fix_score += 0.1  # partial — right direction, no specific value
            feedback_parts.append("⚠️ Correct direction but no specific LR value given")

    breakdown["fix"] = round(min(fix_score, 0.5), 3)
    total = round(breakdown["diagnosis"] + breakdown["fix"], 3)

    if breakdown["fix"] >= 0.4:
        feedback_parts.append("✅ Fix is specific and correct")
    elif breakdown["fix"] > 0:
        feedback_parts.append("⚠️ Fix partially correct")
    else:
        feedback_parts.append("❌ Fix is incorrect or missing")

    return total, breakdown, " | ".join(feedback_parts)


# ─── Task 3 Grader: Silent Data Poisoning ──────────────────────────────────────

def grade_data_poisoning(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    diagnosis = (action_data.get("diagnosis") or "").strip()
    fix_type = (action_data.get("fix_type") or "").strip()
    fix_detail = (action_data.get("fix_detail") or "").strip()

    breakdown = {}
    feedback_parts = []
    poisoned_class = ground_truth["poisoned_class"]

    # 0.3 pts — identifies data/label corruption as bug type
    # Requires BOTH a data-problem signal AND a corruption-specifc term to prevent
    # vague matches on common words like "data" or "class"
    corruption_keywords = ["poison", "corrupt", "mislabel", "label noise", "label corrupt",
                           "corrupted label", "bad label", "annotation error"]
    data_context_keywords = ["label", "data", "annotation", "class", "training data"]

    has_corruption_term = _contains_any(diagnosis, corruption_keywords)
    has_data_context = _contains_any(diagnosis, data_context_keywords)

    if has_corruption_term and has_data_context:
        breakdown["bug_type"] = 0.3
        feedback_parts.append("✅ Identified label/data corruption as root cause")
    elif has_corruption_term:
        breakdown["bug_type"] = 0.15
        feedback_parts.append("⚠️ Mentioned corruption but didn't connect it to labels/data clearly")
    else:
        breakdown["bug_type"] = 0.0
        feedback_parts.append("❌ Did not identify data poisoning / label corruption")

    # 0.2 pts — names the correct poisoned class
    class_mentioned = (
        f"class_{poisoned_class}" in diagnosis.lower()
        or f"class {poisoned_class}" in diagnosis.lower()
        or str(poisoned_class) in diagnosis
    )
    if class_mentioned:
        breakdown["class_identified"] = 0.2
        feedback_parts.append(f"✅ Correctly identified class_{poisoned_class} as the poisoned class")
    else:
        breakdown["class_identified"] = 0.0
        feedback_parts.append(f"❌ Did not identify the specific poisoned class (class_{poisoned_class})")

    # 0.5 pts — correct fix
    fix_keywords = ground_truth["valid_fix_keywords"]
    fix_score = 0.0

    if fix_type in ground_truth["valid_fix_types"]:
        fix_score += 0.2
    if _contains_any(fix_detail, fix_keywords):
        fix_score += 0.3

    breakdown["fix"] = round(fix_score, 3)
    if fix_score >= 0.4:
        feedback_parts.append("✅ Proposed correct data fix")
    elif fix_score > 0:
        feedback_parts.append("⚠️ Fix partially correct")
    else:
        feedback_parts.append("❌ Fix is incorrect or missing")

    total = round(
        breakdown["bug_type"] + breakdown["class_identified"] + breakdown["fix"],
        3,
    )
    return total, breakdown, " | ".join(feedback_parts)


# ─── Dispatcher ────────────────────────────────────────────────────────────────

GRADERS = {
    "easy": grade_overfitting,
    "medium": grade_lr_explosion,
    "hard": grade_data_poisoning,
}


def grade(difficulty: str, action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    grader = GRADERS.get(difficulty)
    if grader is None:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return grader(action_data, ground_truth)