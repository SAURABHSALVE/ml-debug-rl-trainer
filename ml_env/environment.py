"""
MLDebugEnv — core environment logic.

Episode flow:
  reset()  → returns Task 1 (easy) observation with ONLY the task description
  step()   → agent calls investigation tools OR calls diagnose (terminal per task)
             investigation tools reveal partial data, cost one step
             diagnose ends the current task, moves to next task
  state()  → returns current episode state

Key design:
  - Agent starts with ZERO data — only the task description
  - Agent must CHOOSE which tools to call (each costs a step)
  - Wrong / redundant calls waste budget
  - Agent must call diagnose within budget or task scores 0
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from ml_env.graders import grade
from ml_env.models import Action, Observation, Reward
from ml_env.tasks import (
    generate_data_leakage_task,
    generate_poisoning_task,
    generate_class_imbalance_task,
    generate_forgetting_task,
    generate_nan_init_task,
    generate_fp16_underflow_task,
)

# Task pool by difficulty bracket — 3 per episode (1 easy, 1 medium, 1 hard)
TASK_POOL = {
    "easy":   [generate_data_leakage_task, generate_nan_init_task],
    "medium": [generate_fp16_underflow_task, generate_class_imbalance_task],
    "hard":   [generate_poisoning_task, generate_forgetting_task],
}

# Tools that reveal data — ordered from most to least relevant per task
AVAILABLE_TOOLS = [
    "fetch_logs",
    "fetch_config",
    "fetch_loss_curve",
    "fetch_gpu_metrics",
    "fetch_class_metrics",
    "diagnose",
]

# Which tools give a +0.05 reward signal per bug type (keyed by bug_type for precision)
RELEVANT_TOOLS_BY_BUG = {
    "data_leakage":            {"fetch_class_metrics", "fetch_loss_curve"},
    "bad_initialization":      {"fetch_logs", "fetch_config"},
    "fp16_underflow":          {"fetch_logs", "fetch_config"},
    "class_imbalance":         {"fetch_class_metrics", "fetch_config"},
    "silent_data_poisoning":   {"fetch_class_metrics", "fetch_logs"},
    "catastrophic_forgetting": {"fetch_logs", "fetch_loss_curve"},
}
# Fallback by difficulty
RELEVANT_TOOLS = {
    "easy":   {"fetch_loss_curve", "fetch_config"},
    "medium": {"fetch_logs", "fetch_config"},
    "hard":   {"fetch_class_metrics", "fetch_logs"},
}


class MLDebugEnv:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)

        # Episode state
        self._task_index: int = 0
        self._current_task: Optional[Dict[str, Any]] = None
        self._task_step: int = 0
        self._action_history: List[str] = []
        self._revealed_data: Dict[str, Any] = {}
        self._called_tools: List[str] = []

        # Scores per task
        self._scores: Dict[str, float] = {}

        # Global episode budget (instead of per-task)
        self._episode_budget = 16
        self._episode_steps = 0

        # ✅ FIX: Pre-generate tasks at init so list_tasks() works before reset()
        self._tasks: List[Dict[str, Any]] = self._generate_task_set(seed)

    # ─── Task Generation ───────────────────────────────────────────────────────

    def _generate_task_set(self, seed: int) -> List[Dict[str, Any]]:
        """Generate 1 easy + 1 medium + 1 hard task. Used at init and reset."""
        rng = random.Random(seed)
        s = rng.randint(1, 9999)
        easy_gen   = rng.choice(TASK_POOL["easy"])
        medium_gen = rng.choice(TASK_POOL["medium"])
        hard_gen   = rng.choice(TASK_POOL["hard"])
        return [
            easy_gen(seed=s),
            medium_gen(seed=s + 1),
            hard_gen(seed=s + 2),
        ]

    # ─── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode. Picks 1 easy + 1 medium + 1 hard task from the pool."""
        s = self._rng.randint(1, 9999)
        # Pick one generator from each difficulty bracket
        easy_gen   = self._rng.choice(TASK_POOL["easy"])
        medium_gen = self._rng.choice(TASK_POOL["medium"])
        hard_gen   = self._rng.choice(TASK_POOL["hard"])
        self._tasks = [
            easy_gen(seed=s),
            medium_gen(seed=s + 1),
            hard_gen(seed=s + 2),
        ]
        self._task_index = 0
        self._scores = {}
        self._episode_steps = 0

        logger.info(
            f"Episode reset: easy={self._tasks[0]['task_id']}, "
            f"medium={self._tasks[1]['task_id']}, hard={self._tasks[2]['task_id']}"
        )

        return self._load_task(self._tasks[0])

    def _load_task(self, task: Dict[str, Any]) -> Observation:
        self._current_task = task
        self._task_step = 0
        self._action_history = []
        self._revealed_data = {}
        self._called_tools = []
        difficulty = task["difficulty"]

        return Observation(
            task_id=task["task_id"],
            difficulty=difficulty,
            description=task["description"],
            step_number=self._episode_steps,
            max_steps=self._episode_budget,
            steps_remaining=max(0, self._episode_budget - self._episode_steps),
            tool_result=None,
            loss_curve={"train": [], "val": []},
            class_metrics={},
            logs=[],
            config={},
            gpu_metrics={"memory_mb": [], "util_pct": []},
            action_history=[],
            available_tools=AVAILABLE_TOOLS,
        )

    # ─── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step()")

        task = self._current_task
        difficulty = task["difficulty"]

        # ✅ FIX: Increment FIRST (globally and locally)
        self._task_step += 1
        self._episode_steps += 1

        # ✅ Then check against limit (global episode limit)
        if self._episode_steps > self._episode_budget and action.action_type != "diagnose":
            reward = Reward(
                score=0.0,
                breakdown={},
                feedback=f"Global episode budget exhausted after {self._episode_budget} steps. Score = 0.",
                total=0.0,
            )
            done, info, next_obs = self._advance_or_end(reward, difficulty, task)
            return next_obs, reward, done, info

        # ── Investigation action ──────────────────────────────────────────────
        if action.action_type != "diagnose":
            tool_result, intermediate_reward = self._handle_tool_call(action, task)
            self._action_history.append(action.action_type)

            reward = Reward(
                score=intermediate_reward,
                breakdown={"tool_call": intermediate_reward},
                feedback=f"Tool call: {action.action_type} | intermediate reward: {intermediate_reward}",
                total=intermediate_reward,
            )
            obs = self._make_obs(task, tool_result=tool_result)
            return obs, reward, False, {"task_step": self._task_step}

        # ── Diagnose (terminal) ───────────────────────────────────────────────
        action_data = {
            "diagnosis": action.diagnosis or "",
            "fix_type": action.fix_type or "",
            "fix_detail": action.fix_detail or "",
            "confidence": action.confidence or 0.0,
        }
        logger.info(f"GRADER INPUT: {action_data}")
        score, breakdown, feedback = grade(difficulty, action_data, task["ground_truth"])

        # 🚨 Guessing Penalty / Priority 1 - Reward investigation path
        bug_type = task.get("ground_truth", {}).get("bug_type", "")
        relevant_set = RELEVANT_TOOLS_BY_BUG.get(bug_type) or RELEVANT_TOOLS.get(difficulty, set())
        relevant_called = set([t for t in self._called_tools if t in relevant_set])

        if len(relevant_set) > 0:
            investigation_ratio = len(relevant_called) / len(relevant_set)
        else:
            investigation_ratio = 1.0

        # Scale diagnosis score to max 0.5, give up to 0.5 for proper investigation
        base_score = score
        investigation_score = round(investigation_ratio * 0.5, 3)
        scaled_diagnosis_score = round(base_score * 0.5, 3)

        score = scaled_diagnosis_score + investigation_score
        breakdown["investigation_score"] = investigation_score
        breakdown["diagnosis_scaled"] = scaled_diagnosis_score

        if investigation_ratio == 0:
            feedback += f" | ❌ Guess penalty: no relevant tools used (investigation=0.0)"
        elif investigation_ratio < 1.0:
            feedback += f" | ⚠️ Partial investigation: missed some key evidence (investigation={investigation_score})"
        else:
            feedback += f" | 🔍 Full investigation score (investigation={investigation_score})"

        # Efficiency bonus: got it right within very few steps
        efficiency_bonus = 0.0
        if base_score >= 0.8 and self._task_step <= 3:
            efficiency_bonus = 0.05
            feedback += " | ⚡ Efficiency bonus: solved in few steps"

        # Trajectory bonus for hard task
        trajectory_bonus = 0.0
        if difficulty == "hard":
            easy_score = self._scores.get("easy", 0.0)
            medium_score = self._scores.get("medium", 0.0)
            if easy_score > 0.7 and medium_score > 0.6:
                trajectory_bonus = 0.05
                feedback += " | 🏆 Trajectory bonus: consistent performance"

        # Investigation path bonus: fetch_class_metrics before diagnosis for class_imbalance
        path_bonus = 0.0
        if task.get("ground_truth", {}).get("bug_type") == "class_imbalance":
            if "fetch_class_metrics" in self._called_tools:
                path_bonus = 0.1
                feedback += " | 🔍 Professional path bonus: verified class distribution"

        total = round(min(1.0, score + efficiency_bonus + trajectory_bonus + path_bonus), 3)
        reward = Reward(
            score=score,
            breakdown=breakdown,
            feedback=feedback,
            efficiency_bonus=efficiency_bonus,
            trajectory_bonus=trajectory_bonus,
            path_bonus=path_bonus,
            total=total,
        )

        self._action_history.append("diagnose")
        self._scores[difficulty] = total

        if action.action_type == "diagnose":
            diagnosis_preview = (action.diagnosis or "None")[:60]
            logger.info(
                f"Task {task['task_id']} diagnosed: "
                f"score={total:.2f}, "
                f"diagnosis='{diagnosis_preview}...', "
                f"fix_type={action.fix_type}"
            )

        done, info, next_obs = self._advance_or_end(reward, difficulty, task)

        return next_obs, reward, done, info

    # ─── Tool Handlers ─────────────────────────────────────────────────────────

    def _handle_tool_call(self, action: Action, task: Dict[str, Any]) -> Tuple[Dict, float]:
        data = task["data"]
        tool = action.action_type
        bug_type = task.get("ground_truth", {}).get("bug_type", "")
        relevant = RELEVANT_TOOLS_BY_BUG.get(bug_type) or RELEVANT_TOOLS.get(task["difficulty"], set())

        # ✅ FIX 1: Check repeat BEFORE appending
        if tool in self._called_tools:
            return {"result": "Already called this tool. No new information."}, -0.01

        # ✅ FIX 2: Append only on success
        self._called_tools.append(tool)

        intermediate_reward = 0.02 if tool in relevant else 0.0

        if tool == "fetch_logs":
            start = max(1, action.start_epoch or 1)
            end = min(len(data["logs"]), action.end_epoch or len(data["logs"]))
            if start > end:
                start, end = end, start
            result = data["logs"][start - 1 : end]
            return {"logs": result}, intermediate_reward

        elif tool == "fetch_config":
            keys = action.keys or list(data["config"].keys())
            result = {k: data["config"].get(k, "not found") for k in keys}
            return {"config": result}, intermediate_reward

        elif tool == "fetch_loss_curve":
            split = action.split
            loss_data = data.get("loss_curve", {})
            if split:
                if split in loss_data:
                    result = {split: loss_data[split]}
                elif split == "train" and "train" in loss_data:
                    result = {"train_loss": loss_data["train"]}
                elif split == "val":
                    val_keys = [k for k in loss_data.keys() if "val" in k.lower()]
                    if not val_keys:
                        return {"error": "No validation loss found"}, 0.0
                    result = {k: loss_data[k] for k in val_keys}
                else:
                    return {"error": f"Split '{split}' not found. Available keys: {list(loss_data.keys())}"}, 0.0
            else:
                result = loss_data
            return {"loss_curve": result}, intermediate_reward

        elif tool == "fetch_gpu_metrics":
            return {"gpu_metrics": data["gpu_metrics"]}, intermediate_reward

        elif tool == "fetch_class_metrics":
            class_id = action.class_id
            cm = data.get("class_metrics", {})
            if class_id is not None:
                if class_id in cm:
                    result = {str(class_id): cm[class_id]}
                elif str(class_id) in cm:
                    result = {str(class_id): cm[str(class_id)]}
                else:
                    return {"error": f"Class {class_id} not found. Available classes: {list(cm.keys())}"}, 0.0
            else:
                result = cm
            return {"class_metrics": result}, intermediate_reward

        return {"error": f"Unknown tool: {tool}"}, 0.0

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _make_obs(self, task: Dict, tool_result: Optional[Dict]) -> Observation:
        difficulty = task["difficulty"]
        loss_curve = tool_result.get("loss_curve", {"train": [], "val": []}) if tool_result else {"train": [], "val": []}
        class_metrics = tool_result.get("class_metrics", {}) if tool_result else {}
        logs = tool_result.get("logs", []) if tool_result else []
        config = tool_result.get("config", {}) if tool_result else {}
        gpu_metrics = tool_result.get("gpu_metrics", {"memory_mb": [], "util_pct": []}) if tool_result else {"memory_mb": [], "util_pct": []}
        return Observation(
            task_id=task["task_id"],
            difficulty=difficulty,
            description=task["description"],
            step_number=self._episode_steps,
            max_steps=self._episode_budget,
            steps_remaining=max(0, self._episode_budget - self._episode_steps),
            tool_result=tool_result,
            loss_curve=loss_curve,
            class_metrics=class_metrics,
            logs=logs,
            config=config,
            gpu_metrics=gpu_metrics,
            action_history=list(self._action_history),
            available_tools=AVAILABLE_TOOLS,
        )

    def _advance_or_end(self, reward: Reward, difficulty: str, current_task: Dict) -> Tuple[bool, Dict, Observation]:
        self._task_index += 1
        if self._task_index < len(self._tasks):
            logger.info(
                f"Task {current_task['task_id']} complete. "
                f"Moving to {self._tasks[self._task_index]['task_id']}"
            )
            next_obs = self._load_task(self._tasks[self._task_index])
            return False, {
                "task_complete": difficulty,
                "score": reward.total,
                "next_task": next_obs.task_id,
            }, next_obs
        # All 3 tasks done — return final obs from last task
        avg_score = sum(self._scores.values()) / max(len(self._scores), 1)
        logger.info(
            f"Episode complete. Scores: {self._scores}, "
            f"Average: {avg_score:.2f}"
        )
        final_obs = self._make_obs(
            current_task,
            tool_result={"graded": True, "score": reward.total, "episode_complete": True},
        )
        return True, {
            "episode_complete": True,
            "scores": self._scores,
            "average_score": round(avg_score, 3),
        }, final_obs

    # ─── State ─────────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        if self._current_task is None:
            return {"status": "not_started", "call": "POST /reset to begin"}
        difficulty = self._current_task["difficulty"]
        return {
            "current_task": self._current_task["task_id"],
            "difficulty": difficulty,
            "task_index": self._task_index,
            "task_step": self._task_step,
            "episode_steps": self._episode_steps,
            "steps_remaining": max(0, self._episode_budget - self._episode_steps),
            "tools_called": self._called_tools,
            "action_history": self._action_history,
            "scores_so_far": self._scores,
        }

    # ─── List Tasks ────────────────────────────────────────────────────────────

    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        Return task ID, difficulty, description, and grader info for all tasks.
        Works both before and after reset() — tasks are pre-generated at init.
        Grader is returned as a string name (JSON-safe, not a function reference).
        """
        return [
            {
                "task_id": t["task_id"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "has_grader": True,                      # ✅ validator-friendly boolean
                "grader": t["grader"].__name__,          # ✅ string name, JSON-serializable
                "bug_type": t["ground_truth"]["bug_type"],
            }
            for t in self._tasks
        ]