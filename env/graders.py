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


# ─── Task 6 Grader: Bad Initialization (NaN loss) ────────────────────────────

def grade_nan_init(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    diagnosis = (action_data.get("diagnosis") or "").strip()
    fix_type = (action_data.get("fix_type") or "").strip()
    fix_detail = (action_data.get("fix_detail") or "").strip()

    breakdown = {}
    feedback_parts = []

    # 0.5 pts — identifies bad initialization as root cause
    if _contains_any(diagnosis, ground_truth["diagnosis_keywords"]):
        breakdown["diagnosis"] = 0.5
        feedback_parts.append("✅ Correctly identified bad weight initialization as root cause")
    else:
        breakdown["diagnosis"] = 0.0
        feedback_parts.append("❌ Did not identify weight initialization as the problem")

    # 0.5 pts — valid fix
    fix_score = 0.0
    if fix_type in ground_truth["valid_fix_types"]:
        fix_score += 0.2
    if _contains_any(fix_detail, ground_truth["valid_fix_keywords"]):
        fix_score += 0.3

    breakdown["fix"] = round(fix_score, 3)
    if fix_score >= 0.4:
        feedback_parts.append("✅ Proposed valid initialization fix (Xavier/Kaiming/correct std)")
    elif fix_score > 0:
        feedback_parts.append("⚠️ Fix partially correct — specify correct init scheme or std value")
    else:
        feedback_parts.append("❌ Fix is incorrect or missing")

    total = round(breakdown["diagnosis"] + breakdown["fix"], 3)
    return total, breakdown, " | ".join(feedback_parts)


# ─── Task 4 Grader: Class Imbalance ───────────────────────────────────────────

def grade_class_imbalance(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    diagnosis = (action_data.get("diagnosis") or "").strip()
    fix_type = (action_data.get("fix_type") or "").strip()
    fix_detail = (action_data.get("fix_detail") or "").strip()

    breakdown = {}
    feedback_parts = []

    # 0.5 pts — identifies class imbalance as root cause
    if _contains_any(diagnosis, ground_truth["diagnosis_keywords"]):
        breakdown["diagnosis"] = 0.5
        feedback_parts.append("✅ Correctly identified class imbalance as root cause")
    else:
        breakdown["diagnosis"] = 0.0
        feedback_parts.append("❌ Did not identify class imbalance")

    # 0.5 pts — valid fix
    fix_score = 0.0
    if fix_type in ground_truth["valid_fix_types"]:
        fix_score += 0.2
    if _contains_any(fix_detail, ground_truth["valid_fix_keywords"]):
        fix_score += 0.3

    breakdown["fix"] = round(fix_score, 3)
    if fix_score >= 0.4:
        feedback_parts.append("✅ Proposed a valid class-balancing fix")
    elif fix_score > 0:
        feedback_parts.append("⚠️ Fix partially correct — be more specific")
    else:
        feedback_parts.append("❌ Fix is incorrect or missing")

    total = round(breakdown["diagnosis"] + breakdown["fix"], 3)
    return total, breakdown, " | ".join(feedback_parts)


# ─── Task 5 Grader: Catastrophic Forgetting ───────────────────────────────────

def grade_forgetting(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    diagnosis = (action_data.get("diagnosis") or "").strip()
    fix_type = (action_data.get("fix_type") or "").strip()
    fix_detail = (action_data.get("fix_detail") or "").strip()

    breakdown = {}
    feedback_parts = []

    # 0.5 pts — identifies catastrophic forgetting
    if _contains_any(diagnosis, ground_truth["diagnosis_keywords"]):
        breakdown["diagnosis"] = 0.5
        feedback_parts.append("✅ Identified catastrophic forgetting as root cause")
    else:
        breakdown["diagnosis"] = 0.0
        feedback_parts.append("❌ Did not identify catastrophic forgetting")

    # 0.5 pts — valid continual learning fix
    fix_score = 0.0
    if fix_type in ground_truth["valid_fix_types"]:
        fix_score += 0.2
    if _contains_any(fix_detail, ground_truth["valid_fix_keywords"]):
        fix_score += 0.3

    breakdown["fix"] = round(fix_score, 3)
    if fix_score >= 0.4:
        feedback_parts.append("✅ Proposed a valid continual learning fix (EWC/freeze/replay)")
    elif fix_score > 0:
        feedback_parts.append("⚠️ Fix partially correct — mention specific technique")
    else:
        feedback_parts.append("❌ Fix is incorrect or missing")

    total = round(breakdown["diagnosis"] + breakdown["fix"], 3)
    return total, breakdown, " | ".join(feedback_parts)


# ─── LLM Grader (with keyword fallback) ───────────────────────────────────────

import os
import json

def llm_grade(
    action_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    keyword_score: float,
    keyword_breakdown: Dict,
    keyword_feedback: str,
) -> Tuple[float, Dict, str]:
    """
    Optional LLM-based grading. Activates when USE_LLM_GRADING=true and HF_TOKEN is set.
    Falls back to keyword grading on any error (API unavailable, quota exceeded, etc.).

    The LLM scores 0.0–1.0 and its score is blended 60/40 with the keyword score
    to ensure reproducibility while capturing open-ended reasoning quality.
    """
    use_llm = os.environ.get("USE_LLM_GRADING", "false").lower() == "true"
    hf_token = os.environ.get("HF_TOKEN", "")

    if not use_llm or not hf_token:
        return keyword_score, keyword_breakdown, keyword_feedback

    try:
        from openai import OpenAI
        api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        model = os.environ.get("GRADER_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        client = OpenAI(base_url=api_base, api_key=hf_token)

        prompt = f"""You are an expert ML debugging judge. Score the agent's diagnosis from 0.0 to 1.0.

RUBRIC:
- 0.5 points: Did the agent correctly identify the ROOT CAUSE? (must name the specific issue)
- 0.5 points: Did the agent propose a SPECIFIC, ACTIONABLE fix? (vague answers score lower)

FEW-SHOT EXAMPLES:

Example 1 (PERFECT — score 1.0):
  Bug: overfitting
  Root cause: No regularization on small dataset
  Agent diagnosis: "The model is overfitting — train loss drops to 0.05 but val loss rises after epoch 10"
  Agent fix: "Add dropout=0.3 and weight_decay=1e-4 to prevent overfitting on this small dataset"
  Score: 1.0 — correctly names overfitting AND proposes specific config values

Example 2 (WRONG — score 0.0):
  Bug: learning_rate_explosion
  Root cause: LR=0.5 too high for SGD causing gradient explosion
  Agent diagnosis: "The model performance could be better with more training"
  Agent fix: "Try different hyperparameters"
  Score: 0.0 — completely missed the root cause, fix is non-specific

Example 3 (PARTIAL — score 0.5):
  Bug: class_imbalance
  Root cause: 95% majority class, model predicts majority always
  Agent diagnosis: "There seems to be a data quality issue with some classes"
  Agent fix: "Use weighted loss function"
  Score: 0.5 — vaguely identified data issue (half credit) but named a specific fix

NOW SCORE THIS:
  BUG TYPE: {ground_truth.get('bug_type', 'unknown')}
  ROOT CAUSE: {ground_truth.get('root_cause', '')}

  AGENT DIAGNOSIS: {action_data.get('diagnosis', '')}
  AGENT FIX TYPE: {action_data.get('fix_type', '')}
  AGENT FIX DETAIL: {action_data.get('fix_detail', '')}

Respond ONLY with a JSON object:
{{"score": 0.0, "reasoning": "one sentence explanation"}}
"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        llm_score = float(result.get("score", keyword_score))
        llm_score = max(0.0, min(1.0, llm_score))

        # 60% LLM + 40% keyword for robustness
        blended = round(0.6 * llm_score + 0.4 * keyword_score, 3)
        reasoning = result.get("reasoning", "")
        blended_feedback = f"{keyword_feedback} | 🤖 LLM score: {llm_score:.2f} ({reasoning[:80]})"
        breakdown = {**keyword_breakdown, "llm_score": llm_score, "blended_score": blended}
        return blended, breakdown, blended_feedback

    except Exception as e:
        # Graceful fallback — keyword grading always works
        return keyword_score, keyword_breakdown, f"{keyword_feedback} | ⚠️ LLM grader unavailable, using keyword score"


# ─── Dispatcher ────────────────────────────────────────────────────────────────

# Map bug_type → grader function (supports multiple tasks per difficulty)
BUG_TYPE_GRADERS = {
    "overfitting":             grade_overfitting,
    "bad_initialization":      grade_nan_init,
    "learning_rate_explosion": grade_lr_explosion,
    "silent_data_poisoning":   grade_data_poisoning,
    "class_imbalance":         grade_class_imbalance,
    "catastrophic_forgetting": grade_forgetting,
}

# Fallback by difficulty if bug_type is missing
DIFFICULTY_GRADERS = {
    "easy":   grade_overfitting,
    "medium": grade_lr_explosion,
    "hard":   grade_data_poisoning,
}


def grade(difficulty: str, action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    """
    Grade a diagnose action. Routes by bug_type for precise grading,
    falls back to difficulty if bug_type is not present.
    Applies LLM grading if USE_LLM_GRADING=true.
    """
    bug_type = ground_truth.get("bug_type", "")
    grader = BUG_TYPE_GRADERS.get(bug_type) or DIFFICULTY_GRADERS.get(difficulty)
    if grader is None:
        raise ValueError(f"No grader found for bug_type='{bug_type}' difficulty='{difficulty}'")

    keyword_score, keyword_breakdown, keyword_feedback = grader(action_data, ground_truth)
    return llm_grade(action_data, ground_truth, keyword_score, keyword_breakdown, keyword_feedback)