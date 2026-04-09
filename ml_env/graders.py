"""
Graders for all 3 tasks.
Each grader returns a score in [0.0, 1.0] with a breakdown dict
and feedback string explaining why points were given or lost.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def _contains_any(text: str, keywords: list) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ─── Task 1 Grader: Data Leakage ────────────────────────────────────────────────

def grade_data_leakage(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    diagnosis = (action_data.get("diagnosis") or "").strip()
    fix = (action_data.get("fix") or "").strip()
    
    breakdown = {}
    feedback_parts = []
    
    # 0.5 pts — correctly identifies data leakage
    diag_keywords = ground_truth["diagnosis_keywords"]
    if _contains_any(diagnosis, diag_keywords):
        breakdown["diagnosis"] = 0.5
        feedback_parts.append("✅ Correctly identified data leakage / target leak")
    else:
        breakdown["diagnosis"] = 0.0
        feedback_parts.append("❌ Did not identify data leakage as the root cause")
    
    # 0.5 pts — valid fix
    fix_keywords = ground_truth["valid_fix_keywords"]
    valid_fix_types = ground_truth["valid_fix_types"]
    fix_score = 0.0
    
    # Check if any of the valid categories are mentioned
    if _contains_any(fix, valid_fix_types):
        fix_score += 0.2
    if _contains_any(fix, fix_keywords):
        fix_score += 0.3
    
    breakdown["fix"] = round(fix_score, 3)
    if fix_score >= 0.4:
        feedback_parts.append("✅ Proposed valid fix to remove leaking feature")
    elif fix_score > 0:
        feedback_parts.append("⚠️ Fix partially correct — be more specific about which feature")
    else:
        feedback_parts.append("❌ Fix is incorrect or missing")

    total = round(breakdown["diagnosis"] + breakdown["fix"], 3)
    return total, breakdown, " | ".join(feedback_parts)


# ─── Task 2 Grader: FP16 Underflow ───────────────────────────────────────────────

def grade_fp16_underflow(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    diagnosis = (action_data.get("diagnosis") or "").strip()
    fix_type = (action_data.get("fix_type") or "").strip()
    fix_detail = (action_data.get("fix_detail") or "").strip()

    breakdown = {}
    feedback_parts = []

    # 0.5 pts — identifies FP16 underflow or missing scaler
    if _contains_any(diagnosis, ground_truth["diagnosis_keywords"]):
        breakdown["diagnosis"] = 0.5
        feedback_parts.append("✅ Identified fp16 underflow or missing gradient scaler")
    else:
        breakdown["diagnosis"] = 0.0
        feedback_parts.append("❌ Did not identify precision/underflow issue")

    # 0.5 pts — correct fix
    fix_score = 0.0
    fix_keywords = ground_truth["valid_fix_keywords"]

    if fix_type in ground_truth["valid_fix_types"]:
        fix_score += 0.2

    if _contains_any(fix_detail, fix_keywords):
        fix_score += 0.3

    breakdown["fix"] = round(fix_score, 3)
    total = round(breakdown["diagnosis"] + breakdown["fix"], 3)

    if breakdown["fix"] >= 0.4:
        feedback_parts.append("✅ Fix is specific and correct (scaler or bf16)")
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
    Optional LLM-based grading with robust fallback.
    
    Tries multiple JSON extraction strategies:
    1. Regex for JSON object in response
    2. Plain text score extraction (Score: 0.85)
    3. Fallback to keyword grading
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
- 0.5 points: Did the agent correctly identify the ROOT CAUSE?
- 0.5 points: Did the agent propose a SPECIFIC, ACTIONABLE fix?

BUG TYPE: {ground_truth.get('bug_type', 'unknown')}
ROOT CAUSE: {ground_truth.get('root_cause', '')}

AGENT DIAGNOSIS: {action_data.get('diagnosis', '')}
AGENT FIX: {action_data.get('fix_detail', '')}

Respond ONLY with a JSON object: {{"score": 0.0, "reasoning": "brief explanation"}}
"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
            timeout=10.0,
        )

        raw = response.choices[0].message.content.strip()
        llm_score, reasoning = _parse_llm_response(raw)
        
        if llm_score is None:
            logger.warning(f"Failed to parse LLM response: {raw[:100]}")
            return keyword_score, keyword_breakdown, keyword_feedback
        
        # ✅ Blend scores
        blended = round(0.6 * llm_score + 0.4 * keyword_score, 3)
        blended_feedback = (
            f"{keyword_feedback} | 🤖 LLM: {llm_score:.2f} ({reasoning[:60]}...)"
        )
        breakdown = {
            **keyword_breakdown,
            "llm_score": llm_score,
            "blended_score": blended,
        }
        return blended, breakdown, blended_feedback

    except TimeoutError:
        logger.error("LLM API timeout")
        return keyword_score, keyword_breakdown, keyword_feedback
    except Exception as e:
        logger.error(f"LLM grading failed: {e}")
        return keyword_score, keyword_breakdown, keyword_feedback


def _parse_llm_response(raw: str) -> Tuple[float | None, str]:
    """
    Extract score from LLM response with multiple fallback strategies.
    
    Returns:
        (score, reasoning) where score in [0.0, 1.0] or None if parsing failed
    """
    
    # ✅ STRATEGY 1: Extract JSON object using regex
    json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, dict):
                score = result.get("score")
                if isinstance(score, (int, float)):
                    score = float(score)
                    score = max(0.0, min(1.0, score))
                    reasoning = result.get("reasoning", "LLM graded")
                    return score, reasoning
        except json.JSONDecodeError:
            pass
    
    # ✅ STRATEGY 2: Extract from markdown JSON block
    for block in re.findall(r'```(?:json)?\s*(\{.*?\})', raw, re.DOTALL):
        try:
            result = json.loads(block)
            if isinstance(result, dict):
                score = result.get("score")
                if isinstance(score, (int, float)):
                    score = float(score)
                    score = max(0.0, min(1.0, score))
                    reasoning = result.get("reasoning", "LLM graded")
                    return score, reasoning
        except json.JSONDecodeError:
            continue
    
    # ✅ STRATEGY 3: Extract plain text score
    score_match = re.search(r'(?:score|Score)\s*[:\"]?\s*([\d.]+)', raw, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
            reasoning = "Extracted from plain text"
            return score, reasoning
        except ValueError:
            pass
    
    # ✅ STRATEGY 4: Extract reasoning even without score
    reason_match = re.search(r'(?:reasoning|Reasoning)[:\"]?\s*["\']?([^"\']+)', raw)
    reasoning = reason_match.group(1) if reason_match else "Parse failed"
    
    # ❌ Failed to parse anything
    logger.debug(f"Could not extract score from: {raw[:200]}")
    return None, reasoning


# ─── Dispatcher ────────────────────────────────────────────────────────────────

# Map bug_type → grader function (supports multiple tasks per difficulty)
BUG_TYPE_GRADERS = {
    "data_leakage":            grade_data_leakage,
    "bad_initialization":      grade_nan_init,
    "fp16_underflow":          grade_fp16_underflow,
    "silent_data_poisoning":   grade_data_poisoning,
    "class_imbalance":         grade_class_imbalance,
    "catastrophic_forgetting": grade_forgetting,
}

# Fallback by difficulty if bug_type is missing
DIFFICULTY_GRADERS = {
    "easy":   grade_data_leakage,
    "medium": grade_fp16_underflow,
    "hard":   grade_data_poisoning,
}


def grade(difficulty: str, action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    """
    Internal grader dispatcher.
    """
    bug_type = ground_truth.get("bug_type", "")
    grader = BUG_TYPE_GRADERS.get(bug_type) or DIFFICULTY_GRADERS.get(difficulty)
    if grader is None:
        raise ValueError(f"No grader found for bug_type='{bug_type}' difficulty='{difficulty}'")

    keyword_score, keyword_breakdown, keyword_feedback = grader(action_data, ground_truth)
    return llm_grade(action_data, ground_truth, keyword_score, keyword_breakdown, keyword_feedback)


def grade_task(action_data: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[float, Dict, str]:
    """
    Standard 2-argument grader entry point for OpenEnv.
    """
    difficulty = ground_truth.get("difficulty", "easy")
    return grade(difficulty, action_data, ground_truth)