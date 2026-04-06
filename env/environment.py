import random
from typing import Tuple, Dict, Any, Optional, Set, List
from env.models import Observation, Action, Reward, ALL_ACTION_TYPES
from env.tasks import generate_tasks, Task
from env.tools import execute_action
from env.graders import GRADER_MAP
from env.reward import apply_trajectory_bonus, compute_episode_summary, compute_efficiency_bonus


class MLDebugEnv:
    def __init__(self, max_steps_per_task: int = 15, randomize_tasks: bool = False):
        self.max_steps_per_task = max_steps_per_task
        self.randomize_tasks = randomize_tasks
        self.tasks = generate_tasks()
        self._reset_state()

    def _reset_state(self):
        if self.randomize_tasks:
            self.tasks = generate_tasks(
                [random.randint(1, 1_000_000) for _ in range(3)]
            )
        self._task_index = 0
        self._current_task: Optional[Task] = self.tasks[0]
        self._task_step = 0
        self._total_steps = 0
        self._done = False
        self._scores: Dict[str, float] = {}
        self._actions_used: Set[str] = set()
        self._action_history: List[str] = []
        self._last_reward: Optional[Reward] = None
        self._cumulative_signal: float = 0.0

    def reset(self) -> Observation:
        self._reset_state()
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode done — call reset()")

        task = self._current_task
        self._task_step += 1
        self._total_steps += 1

        if action.action_type == "diagnose":
            return self._handle_diagnose(action, task)
        else:
            return self._handle_investigation(action, task)

    # ── Terminal action ───────────────────────────────────────────────────────

    def _handle_diagnose(
        self, action: Action, task: Task
    ) -> Tuple[Observation, Reward, bool, Dict]:
        grader = GRADER_MAP[task.difficulty]
        result = grader(action, task.ground_truth)
        raw_score = result["score"]

        efficiency_bonus = compute_efficiency_bonus(
            self._task_step, self.max_steps_per_task
        )

        if task.difficulty == "hard":
            final_score = apply_trajectory_bonus(
                easy_score=self._scores.get("easy", 0.0),
                medium_score=self._scores.get("medium", 0.0),
                hard_raw=raw_score,
            )
            if final_score > raw_score:
                result["reasoning"] += " | +0.05 trajectory bonus applied"
        else:
            final_score = raw_score

        self._scores[task.difficulty] = final_score

        reward = Reward(
            score=round(min(1.0, final_score + efficiency_bonus), 3),
            fix_quality=round(final_score, 3),
            efficiency_bonus=round(efficiency_bonus, 3),
            intermediate_signal=0.0,
            cumulative_episode_signal=round(self._cumulative_signal, 3),
            reasoning=result["reasoning"],
            steps_used=self._task_step,
        )
        self._last_reward = reward

        self._action_history.append(
            f"diagnose: {(action.diagnosis or '')[:50]}"
        )
        completed_action_history = [
            entry.split("[", 1)[0].split(":", 1)[0].strip()
            for entry in self._action_history
        ]

        # Advance to next task
        self._task_index += 1
        if self._task_index >= len(self.tasks):
            self._done = True
            self._current_task = None
        else:
            self._current_task = self.tasks[self._task_index]
            self._task_step = 0
            self._actions_used = set()
            self._action_history = []
            self._cumulative_signal = 0.0

        info: Dict[str, Any] = {
            "task_id": task.id,
            "difficulty": task.difficulty,
            "ground_truth_bug_type": task.ground_truth["bug_type"],
            "grader_breakdown": result,
            "actions_used": completed_action_history,
        }
        if self._done:
            info["episode_summary"] = compute_episode_summary(self._scores)

        return self._make_observation(), reward, self._done, info

    # ── Investigation action ──────────────────────────────────────────────────

    def _handle_investigation(
        self, action: Action, task: Task
    ) -> Tuple[Observation, Reward, bool, Dict]:
        result_str, signal = execute_action(action, task, self._actions_used)
        self._cumulative_signal = round(self._cumulative_signal + signal, 3)

        reward = Reward(
            score=round(signal, 3),
            fix_quality=0.0,
            efficiency_bonus=0.0,
            intermediate_signal=round(signal, 3),
            cumulative_episode_signal=self._cumulative_signal,
            reasoning=(
                f"{action.action_type} → signal={signal:.3f}"
                + (" [REDUNDANT — penalty]" if signal < 0 else "")
            ),
            steps_used=self._task_step,
        )
        self._last_reward = reward

        # Build concise action summary
        summary = action.action_type
        if action.epochs:
            summary += f"[{action.epochs}]"
        if action.keys:
            summary += f"[{','.join(action.keys)}]"
        if action.split:
            summary += f"[{action.split}]"
        if action.check:
            summary += f"[{action.check}]"
        if action.class_id is not None:
            summary += f"[class_{action.class_id}]"
        self._action_history = (self._action_history + [summary])[-5:]

        steps_remaining = self.max_steps_per_task - self._task_step
        info: Dict[str, Any] = {
            "task_id": task.id,
            "difficulty": task.difficulty,
            "action_result": result_str,
            "steps_remaining": steps_remaining,
        }

        # Force-complete task if step budget exhausted
        if steps_remaining <= 0:
            self._scores[task.difficulty] = 0.0
            self._task_index += 1
            if self._task_index >= len(self.tasks):
                self._done = True
                self._current_task = None
            else:
                self._current_task = self.tasks[self._task_index]
                self._task_step = 0
                self._actions_used = set()
                self._action_history = []
                self._cumulative_signal = 0.0
            info["timeout"] = True
            info["message"] = "Max steps reached — task scored 0.0"
            if self._done:
                info["episode_summary"] = compute_episode_summary(self._scores)
            return self._make_observation(action_result=result_str), reward, self._done, info

        return self._make_observation(action_result=result_str), reward, False, info

    # ── State / introspection ─────────────────────────────────────────────────

    def state(self) -> Dict[str, Any]:
        return {
            "task_index": self._task_index,
            "task_step": self._task_step,
            "total_steps": self._total_steps,
            "done": self._done,
            "current_task_id": self._current_task.id if self._current_task else None,
            "scores_so_far": self._scores,
            "cumulative_signal": self._cumulative_signal,
            "last_reward": self._last_reward.model_dump() if self._last_reward else None,
        }

    def list_tasks(self):
        return [
            {"id": t.id, "difficulty": t.difficulty, "description": t.description}
            for t in self.tasks
        ]

    def _make_observation(self, action_result: Optional[str] = None) -> Observation:
        if self._done or self._current_task is None:
            return Observation(
                task_id="done",
                difficulty="easy",
                description="Episode complete. All tasks finished.",
                step_number=self._total_steps,
                max_steps=self.max_steps_per_task,
                steps_remaining=0,
                available_actions=[],
                action_history=list(self._action_history),
            )
        t = self._current_task
        steps_remaining = max(0, self.max_steps_per_task - self._task_step)
        hint = None
        if self._task_step >= 8:
            hint = "Hint: focus on the metric with the most anomalous trend."

        is_task_entry = self._task_step == 0
        return Observation(
            task_id=t.id,
            difficulty=t.difficulty,
            description=t.description,
            step_number=self._task_step,
            max_steps=self.max_steps_per_task,
            steps_remaining=steps_remaining,
            action_result=action_result,
            action_history=list(self._action_history[-5:]),
            available_actions=ALL_ACTION_TYPES,
            hint=hint,
            training_logs=t.training_logs if is_task_entry else None,
            config_yaml=t.config_yaml if is_task_entry else None,
            loss_curve=t.loss_curve if is_task_entry else None,
            gpu_metrics=t.gpu_metrics if is_task_entry else None,
        )
