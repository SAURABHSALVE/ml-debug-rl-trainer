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

import random
from typing import Any, Dict, List, Optional, Tuple

from env.graders import grade
from env.models import Action, Observation, Reward
from env.tasks import (
    generate_lr_explosion_task,
    generate_overfitting_task,
    generate_poisoning_task,
)

# Tools that reveal data — ordered from most to least relevant per task
AVAILABLE_TOOLS = [
    "fetch_logs",
    "fetch_config",
    "fetch_loss_curve",
    "fetch_gpu_metrics",
    "fetch_class_metrics",
    "diagnose",
]

# Which tools are actually relevant per task difficulty
RELEVANT_TOOLS = {
    "easy": {"fetch_loss_curve", "fetch_config"},
    "medium": {"fetch_logs", "fetch_config"},
    "hard": {"fetch_class_metrics", "fetch_logs"},
}


class MLDebugEnv:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)

        # Episode state
        self._tasks: List[Dict[str, Any]] = []
        self._task_index: int = 0
        self._current_task: Optional[Dict[str, Any]] = None
        self._task_step: int = 0
        self._action_history: List[str] = []
        self._revealed_data: Dict[str, Any] = {}
        self._called_tools: List[str] = []

        # Scores per task
        self._scores: Dict[str, float] = {}

        # Max steps per task
        self._max_steps_map = {"easy": 5, "medium": 5, "hard": 7}

    # ─── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode. Returns Task 1 (easy) observation."""
        s = self._rng.randint(1, 9999)
        self._tasks = [
            generate_overfitting_task(seed=s),
            generate_lr_explosion_task(seed=s + 1),
            generate_poisoning_task(seed=s + 2),
        ]
        self._task_index = 0
        self._scores = {}
        return self._load_task(self._tasks[0])

    def _load_task(self, task: Dict[str, Any]) -> Observation:
        self._current_task = task
        self._task_step = 0
        self._action_history = []
        self._revealed_data = {}
        self._called_tools = []
        difficulty = task["difficulty"]
        max_steps = self._max_steps_map[difficulty]

        return Observation(
            task_id=task["task_id"],
            difficulty=difficulty,
            description=task["description"],
            step_number=0,
            max_steps=max_steps,
            steps_remaining=max_steps,
            tool_result=None,
            action_history=[],
            available_tools=AVAILABLE_TOOLS,
        )

    # ─── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._current_task is None:
            raise RuntimeError("Call reset() before step()")

        task = self._current_task
        difficulty = task["difficulty"]
        max_steps = self._max_steps_map[difficulty]

        # Out of budget — auto-fail
        if self._task_step >= max_steps and action.action_type != "diagnose":
            reward = Reward(
                score=0.0,
                breakdown={},
                feedback="Budget exhausted without diagnosis. Score = 0.",
                total=0.0,
            )
            done, info, next_obs = self._advance_or_end(reward, difficulty, task)
            return next_obs, reward, done, info

        self._task_step += 1

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
        score, breakdown, feedback = grade(difficulty, action_data, task["ground_truth"])

        # Efficiency bonus: got it right within half the budget
        efficiency_bonus = 0.0
        if score >= 0.8 and self._task_step <= max_steps // 2 + 1:
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

        total = round(min(1.0, score + efficiency_bonus + trajectory_bonus), 3)
        reward = Reward(
            score=score,
            breakdown=breakdown,
            feedback=feedback,
            efficiency_bonus=efficiency_bonus,
            trajectory_bonus=trajectory_bonus,
            total=total,
        )

        self._action_history.append("diagnose")
        self._scores[difficulty] = total
        done, info, next_obs = self._advance_or_end(reward, difficulty, task)

        return next_obs, reward, done, info

    # ─── Tool Handlers ─────────────────────────────────────────────────────────

    def _handle_tool_call(self, action: Action, task: Dict[str, Any]) -> Tuple[Dict, float]:
        data = task["data"]
        tool = action.action_type
        relevant = RELEVANT_TOOLS.get(task["difficulty"], set())

        # Penalty for repeating same tool
        if tool in self._called_tools:
            self._called_tools.append(tool)
            return {"result": "Already called this tool. No new information."}, -0.01

        self._called_tools.append(tool)
        intermediate_reward = 0.02 if tool in relevant else 0.0

        if tool == "fetch_logs":
            start = action.start_epoch or 1
            end = action.end_epoch or len(data["logs"])
            result = data["logs"][max(0, start - 1): min(len(data["logs"]), end)]
            return {"logs": result}, intermediate_reward

        elif tool == "fetch_config":
            keys = action.keys or list(data["config"].keys())
            result = {k: data["config"].get(k, "not found") for k in keys}
            return {"config": result}, intermediate_reward

        elif tool == "fetch_loss_curve":
            split = action.split or "both"
            if split == "train":
                result = {"train_loss": data["loss_curve"]["train"]}
            elif split == "val":
                result = {"val_loss": data["loss_curve"]["val"]}
            else:
                result = data["loss_curve"]
            return {"loss_curve": result}, intermediate_reward

        elif tool == "fetch_gpu_metrics":
            return {"gpu_metrics": data["gpu_metrics"]}, intermediate_reward

        elif tool == "fetch_class_metrics":
            class_id = action.class_id
            if class_id is not None:
                cm = data.get("class_metrics", {})
                result = {str(class_id): cm.get(class_id, cm.get(str(class_id), "not found"))}
            else:
                result = data.get("class_metrics", {})
            return {"class_metrics": result}, intermediate_reward

        return {"error": f"Unknown tool: {tool}"}, 0.0

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _make_obs(self, task: Dict, tool_result: Optional[Dict]) -> Observation:
        difficulty = task["difficulty"]
        max_steps = self._max_steps_map[difficulty]
        return Observation(
            task_id=task["task_id"],
            difficulty=difficulty,
            description=task["description"],
            step_number=self._task_step,
            max_steps=max_steps,
            steps_remaining=max(0, max_steps - self._task_step),
            tool_result=tool_result,
            action_history=list(self._action_history),
            available_tools=AVAILABLE_TOOLS,
        )

    def _advance_or_end(self, reward: Reward, difficulty: str, current_task: Dict) -> Tuple[bool, Dict, Observation]:
        self._task_index += 1
        if self._task_index < len(self._tasks):
            next_obs = self._load_task(self._tasks[self._task_index])
            return False, {
                "task_complete": difficulty,
                "score": reward.total,
                "next_task": next_obs.task_id,
            }, next_obs
        # All 3 tasks done — return final obs from last task
        final_obs = self._make_obs(
            current_task,
            tool_result={"graded": True, "score": reward.total, "episode_complete": True},
        )
        return True, {
            "episode_complete": True,
            "scores": self._scores,
            "average_score": round(
                sum(self._scores.values()) / max(len(self._scores), 1), 3
            ),
        }, final_obs

    # ─── State ─────────────────────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        if self._current_task is None:
            return {"status": "not_started", "call": "POST /reset to begin"}
        difficulty = self._current_task["difficulty"]
        max_steps = self._max_steps_map[difficulty]
        return {
            "current_task": self._current_task["task_id"],
            "difficulty": difficulty,
            "task_index": self._task_index,
            "task_step": self._task_step,
            "steps_remaining": max(0, max_steps - self._task_step),
            "tools_called": self._called_tools,
            "action_history": self._action_history,
            "scores_so_far": self._scores,
        }

    # ─── List Tasks ────────────────────────────────────────────────────────────

    def list_tasks(self) -> List[Dict[str, Any]]:
        return [
            {"task_id": t["task_id"], "difficulty": t["difficulty"], "description": t["description"]}
            for t in self._tasks
        ] if self._tasks else []