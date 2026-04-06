"""
Tabular RL agent for ML debugging strategy learning.

The agent learns Q-values over a compact state abstraction built from the
environment observation. This keeps the implementation lightweight and
submission-friendly while still being real reinforcement learning.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


NON_TERMINAL_ACTIONS = [
    "fetch_logs",
    "fetch_config",
    "fetch_loss_curve",
    "fetch_diagnostics",
    "fetch_class_data",
]

ALL_AGENT_ACTIONS = NON_TERMINAL_ACTIONS + ["diagnose"]


class StateEncoder:
    """Convert observations into a small, stable tabular state key."""

    def __init__(self):
        self.action_vocab = {name: idx for idx, name in enumerate(ALL_AGENT_ACTIONS)}

    def _as_dict(self, obs: Any) -> Dict[str, Any]:
        if obs is None:
            return {}
        if isinstance(obs, dict):
            return obs
        if hasattr(obs, "model_dump"):
            return obs.model_dump()
        raise TypeError(f"Unsupported observation type: {type(obs)!r}")

    def _history_to_flags(self, history: Sequence[str]) -> Dict[str, int]:
        seen = {str(item).split("[", 1)[0] for item in history}
        return {
            "logs": int("fetch_logs" in seen),
            "config": int("fetch_config" in seen),
            "curve": int("fetch_loss_curve" in seen),
            "diagnostics": int("fetch_diagnostics" in seen),
            "class_data": int("fetch_class_data" in seen),
        }

    def encode_observation(self, obs: Any) -> Tuple[Any, ...]:
        data = self._as_dict(obs)

        difficulty = data.get("difficulty") or data.get("task_difficulty") or "easy"
        step_number = int(data.get("step_number", data.get("step_count", 0)) or 0)
        max_steps = int(data.get("max_steps", 15) or 15)
        steps_remaining = int(data.get("steps_remaining", max(max_steps - step_number, 0)) or 0)
        history = data.get("action_history") or []
        hint_unlocked = int(bool(data.get("hint")))
        action_result = str(data.get("action_result") or "").lower()

        evidence_flags = self._history_to_flags(history)

        # Bucket the episode progress so Q-values generalize across nearby steps.
        progress_bucket = min(3, int((step_number / max(1, max_steps)) * 4))

        anomaly_flags = (
            int("nan" in action_result or "unstable" in action_result or "inf" in action_result),
            int("diverging" in action_result or "overfitting" in action_result),
            int("label" in action_result or "class_" in action_result or "corrupt" in action_result),
        )

        return (
            difficulty,
            progress_bucket,
            min(steps_remaining, max_steps),
            evidence_flags["logs"],
            evidence_flags["config"],
            evidence_flags["curve"],
            evidence_flags["diagnostics"],
            evidence_flags["class_data"],
            hint_unlocked,
            *anomaly_flags,
        )


class DQNAgent:
    """
    Lightweight Q-learning agent.

    The name is kept for backwards compatibility with the rest of the project,
    but the implementation is intentionally tabular for speed and reliability.
    """

    def __init__(
        self,
        learning_rate: float = 0.15,
        epsilon: float = 0.20,
        gamma: float = 0.95,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
    ):
        self.encoder = StateEncoder()
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(
            lambda: {action: 0.0 for action in ALL_AGENT_ACTIONS}
        )
        self.strategy_templates: Dict[str, List[str]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }
        self.best_episode_scores: Dict[str, float] = {
            "easy": 0.0,
            "medium": 0.0,
            "hard": 0.0,
        }
        self.difficulty_action_values: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {action: 0.0 for action in NON_TERMINAL_ACTIONS}
        )
        self.training_history: List[Dict[str, Any]] = []

    def _get_action_values(
        self,
        state_key: Tuple[Any, ...],
        available_actions: Sequence[str],
        task_difficulty: str,
    ) -> List[Tuple[str, float]]:
        state_values = self.q_table[state_key]
        difficulty_priors = self.difficulty_action_values[task_difficulty]

        values = []
        for action in available_actions:
            q_val = state_values.get(action, 0.0)
            prior = difficulty_priors.get(action, 0.0)
            values.append((action, q_val + 0.15 * prior))
        return values

    def select_action(
        self,
        observation: Any,
        available_actions: List[str],
        task_difficulty: str,
        use_greedy: bool = False,
    ) -> str:
        state_key = self.encoder.encode_observation(observation)

        if not available_actions:
            return "diagnose"

        if not use_greedy and np.random.random() < self.epsilon:
            return str(np.random.choice(available_actions))

        action_values = self._get_action_values(state_key, available_actions, task_difficulty)
        action_values.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return action_values[0][0]

    def _update_q_value(
        self,
        state_key: Tuple[Any, ...],
        action: str,
        reward: float,
        next_state_key: Optional[Tuple[Any, ...]],
        done: bool,
    ) -> None:
        current_q = self.q_table[state_key].get(action, 0.0)
        next_max = 0.0
        if not done and next_state_key is not None:
            next_max = max(self.q_table[next_state_key].values())

        target = reward if done else reward + self.gamma * next_max
        updated_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[state_key][action] = round(updated_q, 6)

    def learn_from_episode(
        self,
        task_difficulty: str,
        actions_used: Optional[List[str]] = None,
        final_reward: float = 0.0,
        episode_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Learn from a completed task.

        `episode_trace` should contain step-wise transitions with:
          state, action, reward, next_state, done
        """
        clean_actions = [a for a in (actions_used or []) if a != "diagnose"]

        if episode_trace:
            for transition in episode_trace:
                state_key = self.encoder.encode_observation(transition.get("state"))
                next_state = transition.get("next_state")
                next_state_key = (
                    self.encoder.encode_observation(next_state)
                    if next_state is not None and not transition.get("done", False)
                    else None
                )
                self._update_q_value(
                    state_key=state_key,
                    action=transition["action"],
                    reward=float(transition.get("reward", 0.0)),
                    next_state_key=next_state_key,
                    done=bool(transition.get("done", False)),
                )

        # Keep a difficulty-level prior so recommendations still work when a
        # user lands in a state the agent has not seen often enough.
        reward_signal = max(0.0, float(final_reward))
        for action in clean_actions:
            if action in self.difficulty_action_values[task_difficulty]:
                prior = self.difficulty_action_values[task_difficulty][action]
                self.difficulty_action_values[task_difficulty][action] = round(
                    prior + 0.1 * (reward_signal - prior),
                    6,
                )

        if clean_actions and final_reward >= self.best_episode_scores.get(task_difficulty, 0.0):
            self.best_episode_scores[task_difficulty] = final_reward
            self.strategy_templates[task_difficulty] = clean_actions

        self.training_history.append(
            {
                "difficulty": task_difficulty,
                "actions": list(actions_used or []),
                "reward": round(float(final_reward), 3),
            }
        )

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_recommended_actions(
        self,
        observation: Any,
        task_difficulty: str,
        limit: int = 5,
    ) -> List[str]:
        available = list(NON_TERMINAL_ACTIONS)
        state_key = self.encoder.encode_observation(observation)
        ranked = self._get_action_values(state_key, available, task_difficulty)

        if all(abs(score) < 1e-9 for _, score in ranked):
            fallback = self.strategy_templates.get(task_difficulty, [])
            if fallback:
                return fallback[:limit]

            priority_map = {
                "easy": ["fetch_loss_curve", "fetch_config", "fetch_diagnostics", "fetch_logs", "fetch_class_data"],
                "medium": ["fetch_logs", "fetch_config", "fetch_diagnostics", "fetch_loss_curve", "fetch_class_data"],
                "hard": ["fetch_diagnostics", "fetch_class_data", "fetch_logs", "fetch_loss_curve", "fetch_config"],
            }
            return priority_map.get(task_difficulty, available)[:limit]

        ranked.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return [action for action, _ in ranked[:limit]]

    def save(self, filepath: str) -> None:
        checkpoint = {
            "q_table": dict(self.q_table),
            "strategy_templates": self.strategy_templates,
            "training_history": self.training_history,
            "difficulty_action_values": dict(self.difficulty_action_values),
            "best_episode_scores": self.best_episode_scores,
            "epsilon": self.epsilon,
        }
        directory = os.path.dirname(filepath) or "."
        os.makedirs(directory, exist_ok=True)
        with open(filepath, "wb") as handle:
            pickle.dump(checkpoint, handle)

    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False

        with open(filepath, "rb") as handle:
            checkpoint = pickle.load(handle)

        self.q_table = defaultdict(
            lambda: {action: 0.0 for action in ALL_AGENT_ACTIONS},
            checkpoint.get("q_table", {}),
        )
        self.strategy_templates = checkpoint.get(
            "strategy_templates",
            {"easy": [], "medium": [], "hard": []},
        )
        self.training_history = checkpoint.get("training_history", [])
        self.best_episode_scores = checkpoint.get(
            "best_episode_scores",
            {"easy": 0.0, "medium": 0.0, "hard": 0.0},
        )

        loaded_action_values = checkpoint.get("difficulty_action_values", {})
        self.difficulty_action_values = defaultdict(
            lambda: {action: 0.0 for action in NON_TERMINAL_ACTIONS},
            loaded_action_values,
        )
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        return True

    def get_stats(self) -> Dict[str, Any]:
        if not self.training_history:
            return {
                "trained": False,
                "episodes": 0,
                "epsilon": round(self.epsilon, 4),
            }

        recent = self.training_history[-100:]
        rewards_by_difficulty: Dict[str, List[float]] = defaultdict(list)
        for entry in recent:
            rewards_by_difficulty[entry["difficulty"]].append(float(entry["reward"]))

        avg_rewards = {
            diff: round(float(np.mean(rewards)), 3) if rewards else 0.0
            for diff, rewards in rewards_by_difficulty.items()
        }

        return {
            "trained": True,
            "total_episodes": len(self.training_history),
            "average_rewards_by_difficulty": avg_rewards,
            "strategies_learned": bool(any(self.strategy_templates.values())),
            "best_episode_scores": self.best_episode_scores,
            "epsilon": round(self.epsilon, 4),
        }
