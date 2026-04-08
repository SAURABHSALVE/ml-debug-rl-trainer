"""
Tool implementations for the ML Debug environment.

Each function returns:
  (result_str, intermediate_signal)

intermediate_signal is a small reward (−0.01 – 0.03):
  +0.01–0.03 for calls that reveal information relevant to the actual bug
  −0.01      for redundant identical calls (RL penalty)
"""
import re
import yaml
from typing import Set, Tuple, List, Optional
from ml_env.models import Action
from ml_env.tasks import Task


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_epochs(epochs_str: Optional[str], max_epochs: int) -> Tuple[int, int]:
    """Parse "1-10", "15-20", "all", or a single number into (start, end)."""
    if not epochs_str or epochs_str.strip().lower() == "all":
        return 1, max_epochs
    s = epochs_str.strip()
    if "-" in s:
        parts = s.split("-", 1)
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            return 1, max_epochs
    try:
        e = int(s)
        return e, e
    except ValueError:
        return 1, max_epochs


# ── 1. fetch_logs ─────────────────────────────────────────────────────────────

def _fetch_logs(task: Task, epochs_str: Optional[str]) -> Tuple[str, float]:
    logs = task.training_logs
    start, end = _parse_epochs(epochs_str, len(logs))
    s = max(0, start - 1)
    e = min(len(logs), end)
    if s >= e:
        return f"No logs for epoch range {epochs_str!r}. Max epoch: {len(logs)}.", 0.0

    lines = logs[s:e]
    result = "\n".join(lines)

    bug = task.ground_truth.get("bug_type", "")
    signal = 0.0
    if bug == "overfitting":
        signal = 0.02 if (any("val_loss" in l for l in lines) and end >= 10) else 0.0
    elif bug == "lr_too_high":
        signal = 0.02 if any("nan" in l.lower() or "UNSTABLE" in l for l in lines) else 0.0
    elif bug == "label_corruption":
        signal = 0.02 if any("WARNING" in l or "label_consistency" in l for l in lines) else 0.0

    return result, signal


# ── 2. fetch_config ───────────────────────────────────────────────────────────

def _fetch_config(task: Task, keys: Optional[List[str]]) -> Tuple[str, float]:
    config = yaml.safe_load(task.config_yaml)

    if not keys:
        return task.config_yaml, 0.01

    found, missing = {}, []
    for k in keys:
        if k in config:
            found[k] = config[k]
        else:
            missing.append(k)

    lines = [f"{k}: {v}" for k, v in found.items()]
    if missing:
        lines.append(f"Not found: {missing}. Available: {list(config.keys())}")
    result = "\n".join(lines) if lines else "No matching keys found."

    bug = task.ground_truth.get("bug_type", "")
    buggy_key = task.ground_truth.get("buggy_key", "")
    correct_keys = task.ground_truth.get("correct_fix_keys", [])

    if bug == "lr_too_high" and buggy_key in (keys or []):
        signal = 0.03
    elif bug == "overfitting" and any(k in correct_keys for k in (keys or [])):
        signal = 0.02
    else:
        signal = 0.01

    return result, signal


# ── 3. fetch_loss_curve ───────────────────────────────────────────────────────

def _fetch_loss_curve(task: Task, split: Optional[str]) -> Tuple[str, float]:
    lc = task.loss_curve
    split = (split or "all").lower()

    if split == "all":
        selected = lc
    elif split in lc:
        selected = {split: lc[split]}
    else:
        available = list(lc.keys())
        return f"Split '{split}' not found. Available: {available}", 0.0

    lines = []
    for name, vals in selected.items():
        valid = [x for x in vals if x == x and x != 0.0]
        if not valid:
            lines.append(f"{name}: all NaN/zero (training collapsed)")
            continue
        first3 = [f"{v:.4f}" for v in valid[:3]]
        last3  = [f"{v:.4f}" for v in valid[-3:]]
        nan_count = sum(1 for v in vals if v != v or v == 0.0)
        trend = "↑ RISING" if valid[-1] > valid[0] * 1.1 else "↓ falling"
        line = f"{name}: [{', '.join(first3)} ... {', '.join(last3)}]  trend={trend}"
        if nan_count:
            line += f"  ⚠ {nan_count} NaN/collapsed epochs"
        lines.append(line)

    result = "\n".join(lines)

    bug = task.ground_truth.get("bug_type", "")
    poison_cls = task.ground_truth.get("poisoned_class")
    if bug == "overfitting" and "val" in split:
        signal = 0.03
    elif bug == "lr_too_high" and ("train" in split or split == "all"):
        signal = 0.02
    elif bug == "label_corruption" and poison_cls is not None and f"class_{poison_cls}" in selected:
        signal = 0.03
    else:
        signal = 0.01

    return result, signal


# ── 4. fetch_diagnostics ─────────────────────────────────────────────────────

def _fetch_diagnostics(task: Task, check: Optional[str]) -> Tuple[str, float]:
    check = (check or "trends").lower()
    train = task.loss_curve.get("train", [])
    val   = task.loss_curve.get("val",   [])
    logs  = task.training_logs
    bug   = task.ground_truth.get("bug_type", "")

    if check == "overfitting":
        if not train or not val:
            return "No loss data available.", 0.0
        vt = [v for v in val if v == v and v > 0]
        tr = [v for v in train if v == v and v > 0]
        if not vt or not tr:
            return "Loss data all NaN/zero.", 0.0
        min_val_epoch = val.index(min(vt)) + 1
        final_gap = vt[-1] - tr[-1]
        trend = "DIVERGING ⚠" if final_gap > 0.15 else "CONVERGING"
        result = (
            f"Train loss: {tr[0]:.4f} → {tr[-1]:.4f}\n"
            f"Val   loss: {vt[0]:.4f} → {vt[-1]:.4f}  (min at epoch {min_val_epoch})\n"
            f"Final train/val gap: {final_gap:.4f}\n"
            f"Verdict: {trend}"
        )
        signal = 0.03 if bug == "overfitting" else 0.01

    elif check == "gradients":
        norms = [float(m.group(1))
                 for l in logs
                 for m in [re.search(r'grad_norm=([\d.]+)', l)] if m]
        nan_c = sum(1 for l in logs if "nan" in l.lower())
        inf_c = sum(1 for l in logs if "inf" in l.lower())
        if not norms:
            result = f"No grad_norm in logs. NaN epochs: {nan_c}, Inf epochs: {inf_c}"
        else:
            result = (
                f"Grad norms — min: {min(norms):.3f}, max: {max(norms):.3f}, last: {norms[-1]:.3f}\n"
                f"NaN/Inf epochs: {nan_c} / {inf_c}\n"
                f"Verdict: {'EXPLODING ⚠' if norms[-1] > 10 or inf_c > 0 else 'STABLE'}"
            )
        signal = 0.03 if bug == "lr_too_high" else 0.01

    elif check == "trends":
        nan_epochs = sum(1 for l in logs if "nan" in l.lower())
        lines = [f"Total epochs: {len(train)}"]
        if nan_epochs:
            lines.append(f"Training instability: {nan_epochs} epochs with NaN/inf ⚠")
        vt = [v for v in train if v > 0]
        vv = [v for v in val if v > 0]
        if vt:
            lines.append(f"Train loss range: [{min(vt):.4f}, {max(vt):.4f}]")
        if vv:
            lines.append(f"Val   loss range: [{min(vv):.4f}, {max(vv):.4f}]")
        result = "\n".join(lines)
        signal = 0.01

    elif check == "class_balance":
        class_curves = {k: v for k, v in task.loss_curve.items()
                        if k not in ("train", "val")}
        if not class_curves:
            return "No per-class accuracy data for this task.", 0.0
        lines = ["Per-class val accuracy (final epoch):"]
        for cls, vals in sorted(class_curves.items()):
            final  = vals[-1]
            spread = max(vals) - min(vals)
            status = "STAGNANT ⚠ possible label corruption" if spread < 0.12 and final < 0.5 else "OK"
            lines.append(f"  {cls}: {final:.3f}  [{status}]")
        result = "\n".join(lines)
        signal = 0.03 if bug == "label_corruption" else 0.01

    else:
        return (
            f"Unknown check '{check}'. "
            f"Use: overfitting | gradients | trends | class_balance", 0.0
        )

    return result, signal


# ── 5. fetch_class_data ───────────────────────────────────────────────────────

def _fetch_class_data(task: Task, class_id: int) -> Tuple[str, float]:
    key = f"class_{class_id}"
    all_keys = [k for k in task.loss_curve if k.startswith("class_")]

    if key not in task.loss_curve:
        return (
            f"No data for class_{class_id}. Available: {sorted(all_keys)}", 0.0
        )

    vals      = task.loss_curve[key]
    avg       = sum(vals) / len(vals)
    final     = vals[-1]
    spread    = max(vals) - min(vals)
    suspicious = final < 0.5 and spread < 0.12

    result = (
        f"class_{class_id} accuracy over {len(vals)} epochs:\n"
        f"  Final: {final:.3f}   Average: {avg:.3f}\n"
        f"  Range: [{min(vals):.3f}, {max(vals):.3f}]  spread={spread:.3f}\n"
        f"  Assessment: {'SUSPICIOUS — stagnant, likely label corruption ⚠' if suspicious else 'Normal learning curve'}"
    )

    poisoned = task.ground_truth.get("poisoned_class")
    signal = 0.03 if (poisoned is not None and class_id == poisoned) else 0.0
    return result, signal


# ── Dispatcher ────────────────────────────────────────────────────────────────

def execute_action(
    action: Action,
    task: Task,
    actions_used: Set[str],
) -> Tuple[str, float]:
    """
    Execute an investigation action and return (result_string, intermediate_signal).
    Redundant identical calls receive a −0.01 penalty (RL penalization).
    """
    # Cache key — excludes terminal-only fields
    cache_key = str(action.model_dump(
        exclude={"confidence", "diagnosis", "fix_type", "fix_detail"}
    ))
    already_used = cache_key in actions_used
    actions_used.add(cache_key)

    atype = action.action_type

    if atype == "fetch_logs":
        result, sig = _fetch_logs(task, action.epochs)

    elif atype == "fetch_config":
        result, sig = _fetch_config(task, action.keys)

    elif atype == "fetch_loss_curve":
        result, sig = _fetch_loss_curve(task, action.split)

    elif atype == "fetch_diagnostics":
        result, sig = _fetch_diagnostics(task, action.check)

    elif atype == "fetch_class_data":
        if action.class_id is None:
            return "Error: class_id is required for fetch_class_data.", 0.0
        result, sig = _fetch_class_data(task, action.class_id)

    else:
        return f"Unknown action_type: {atype!r}", 0.0

    if already_used:
        sig = -0.01   # RL penalty for redundant calls

    return result, sig
