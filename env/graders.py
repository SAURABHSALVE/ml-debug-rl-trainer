import re
from typing import Dict, Any
from env.models import Action


def grade_overfitting(action: Action, ground_truth: Dict[str, Any]) -> Dict:
    score = 0.0
    diagnosis_score = 0.0
    fix_score = 0.0
    reasons = []

    # grader receives a diagnose action; reads diagnosis / fix_detail / fix_type
    diag_lower = (action.diagnosis or "").lower()
    fix_lower = ((action.fix_detail or "") + " " + (action.fix_type or "")).lower()

    # Diagnosis (0.0 – 0.5)
    if "overfit" in diag_lower:
        diagnosis_score = 0.5
        reasons.append("✓ Correctly identified overfitting")
    elif "val" in diag_lower and ("loss" in diag_lower or "accuracy" in diag_lower):
        diagnosis_score = 0.25
        reasons.append("⚠ Partial: noticed val metric but didn't name overfitting")
    elif "generaliz" in diag_lower or "regulariz" in diag_lower:
        diagnosis_score = 0.3
        reasons.append("⚠ Partial: related concept, not exact diagnosis")

    # Fix quality (0.0 – 0.5)
    if any(kw in fix_lower for kw in ["dropout", "weight_decay", "l2"]):
        fix_score = 0.5
        reasons.append("✓ Correct fix: regularization")
    elif "augment" in fix_lower:
        fix_score = 0.4
        reasons.append("✓ Valid fix: data augmentation")
    elif "early stop" in fix_lower:
        fix_score = 0.35
        reasons.append("✓ Valid fix: early stopping")
    elif "dataset" in fix_lower and ("more" in fix_lower or "larger" in fix_lower):
        fix_score = 0.3
        reasons.append("⚠ Partial: more data is valid but vague")
    elif "config_change" in fix_lower:
        fix_score = 0.1
        reasons.append("⚠ Correct fix_type but fix_detail insufficient")

    score = diagnosis_score + fix_score
    return {
        "score": round(score, 3),
        "diagnosis_score": diagnosis_score,
        "fix_score": fix_score,
        "reasoning": " | ".join(reasons) or "No matching signals found in response",
    }


def grade_lr_schedule(action: Action, ground_truth: Dict[str, Any]) -> Dict:
    diagnosis_score = 0.0
    fix_score = 0.0
    reasons = []

    diag_lower = (action.diagnosis or "").lower()
    fix_lower = (action.fix_detail or "").lower()

    # Diagnosis (0.5)
    if "learning rate" in diag_lower or "lr" in diag_lower:
        if any(w in diag_lower for w in ["high", "large", "explod", "unstable", "nan", "diverge"]):
            diagnosis_score = 0.5
            reasons.append("✓ Correctly identified high LR causing instability")
        else:
            diagnosis_score = 0.3
            reasons.append("⚠ Mentioned LR but missed the 'too high' qualifier")
    elif "gradient" in diag_lower and "explod" in diag_lower:
        diagnosis_score = 0.35
        reasons.append("⚠ Identified gradient explosion (effect, not cause)")
    elif "nan" in diag_lower or "instab" in diag_lower:
        diagnosis_score = 0.2
        reasons.append("⚠ Noticed instability but didn't identify cause")

    # Fix: proposed value in correct range? (0.5)
    numbers = re.findall(r'[\d.e\-]+', fix_lower)
    low, high = ground_truth["correct_value_range"]
    for num_str in numbers:
        try:
            val = float(num_str)
            if low <= val <= high:
                fix_score = 0.5
                reasons.append(f"✓ Proposed valid LR value: {val}")
                break
            elif val < low:
                fix_score = 0.35
                reasons.append(f"⚠ Proposed LR {val} is in valid direction but too small")
                break
        except ValueError:
            pass

    if fix_score == 0.0:
        also_ok = ground_truth.get("also_acceptable", [])
        if any(k in fix_lower for k in also_ok):
            fix_score = 0.25
            reasons.append("⚠ Proposed valid supporting fix (warmup/clip) but missed LR correction")

    score = diagnosis_score + fix_score
    return {
        "score": round(score, 3),
        "diagnosis_score": diagnosis_score,
        "fix_score": fix_score,
        "reasoning": " | ".join(reasons) or "No matching signals found",
    }


def grade_data_poisoning(action: Action, ground_truth: Dict[str, Any]) -> Dict:
    diagnosis_score = 0.0
    fix_score = 0.0
    reasons = []

    diag_lower = (action.diagnosis or "").lower()
    fix_lower = (action.fix_detail or "").lower()
    fix_type = (action.fix_type or "").lower()
    target_class = ground_truth["poisoned_class"]

    # Bug type identification (0.3)
    if any(w in diag_lower for w in ["label", "annotation", "corrupt", "poison", "mislabel"]):
        diagnosis_score += 0.3
        reasons.append("✓ Identified data/label quality issue")
    elif "data" in diag_lower and ("quality" in diag_lower or "issue" in diag_lower):
        diagnosis_score += 0.15
        reasons.append("⚠ Generic data issue, not specific enough")

    # Correct class identification (0.2)
    class_patterns = [f"class_{target_class}", f"class {target_class}", str(target_class)]
    if any(p in diag_lower or p in fix_lower for p in class_patterns):
        diagnosis_score += 0.2
        reasons.append(f"✓ Correctly identified class_{target_class} as affected")
    else:
        for c in range(5):
            if c != target_class and (
                f"class_{c}" in diag_lower or f"class {c}" in diag_lower
            ):
                diagnosis_score += 0.05
                reasons.append(f"✗ Identified wrong class (class_{c})")
                break

    # Fix quality (0.5)
    if fix_type == "data_fix":
        fix_score += 0.2
        reasons.append("✓ Correct fix_type: data_fix")

    if any(w in fix_lower for w in ["re-annotate", "reannotate", "relabel", "re-label"]):
        fix_score += 0.3
        reasons.append("✓ Proposed re-annotation as fix")
    elif any(w in fix_lower for w in ["filter", "remove", "clean", "audit"]):
        fix_score += 0.25
        reasons.append("✓ Proposed data filtering/audit")
    elif "inspect" in fix_lower or "review" in fix_lower:
        fix_score += 0.15
        reasons.append("⚠ Proposed inspection — valid but not a concrete fix")

    score = min(1.0, diagnosis_score + fix_score)
    return {
        "score": round(score, 3),
        "diagnosis_score": diagnosis_score,
        "fix_score": fix_score,
        "reasoning": " | ".join(reasons) or "No matching signals found",
    }


GRADER_MAP = {
    "easy": grade_overfitting,
    "medium": grade_lr_schedule,
    "hard": grade_data_poisoning,
}
