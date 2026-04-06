#!/usr/bin/env python3
"""
Train the tabular RL agent on randomized debugging episodes.

Usage:
    python train_agent.py --episodes 500 --checkpoint agent_checkpoint.pkl
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from env.environment import MLDebugEnv
from env.models import Action, Observation
from env.rl_agent import DQNAgent, NON_TERMINAL_ACTIONS


INVESTIGATION_DEFAULTS = {
    "fetch_logs": lambda difficulty: Action(
        action_type="fetch_logs",
        epochs="15-20" if difficulty == "easy" else "all",
    ),
    "fetch_config": lambda difficulty: Action(
        action_type="fetch_config",
        keys=["lr", "optimizer", "dropout", "weight_decay", "gradient_clip"],
    ),
    "fetch_loss_curve": lambda difficulty: Action(
        action_type="fetch_loss_curve",
        split="val" if difficulty == "easy" else "all",
    ),
    "fetch_diagnostics": lambda difficulty: Action(
        action_type="fetch_diagnostics",
        check={
            "easy": "overfitting",
            "medium": "gradients",
            "hard": "class_balance",
        }.get(difficulty, "trends"),
    ),
    "fetch_class_data": lambda difficulty: Action(
        action_type="fetch_class_data",
        class_id=0,
    ),
}


def build_diagnosis_action(difficulty: str) -> Action:
    templates = {
        "easy": Action(
            action_type="diagnose",
            diagnosis="The model is overfitting. Validation loss diverges while training loss keeps improving.",
            fix_type="config_change",
            fix_detail="Add dropout=0.3 and weight_decay=1e-4, then enable early stopping.",
            confidence=0.9,
        ),
        "medium": Action(
            action_type="diagnose",
            diagnosis="The learning rate is too high, which causes instability, gradient explosion, and NaN losses.",
            fix_type="config_change",
            fix_detail="Reduce lr to 0.001, add gradient_clip=1.0, and use warmup_steps=500.",
            confidence=0.9,
        ),
        "hard": Action(
            action_type="diagnose",
            diagnosis="This looks like silent label corruption affecting one class and dragging down its per-class accuracy.",
            fix_type="data_fix",
            fix_detail="Run a label audit, re-annotate corrupted samples, and filter the bad labels before retraining.",
            confidence=0.8,
        ),
    }
    return templates[difficulty]


def choose_action(
    agent: DQNAgent,
    obs: Observation,
    train_mode: bool,
) -> Action:
    difficulty = obs.difficulty
    steps_used = obs.step_number
    minimum_investigation = {"easy": 2, "medium": 2, "hard": 3}.get(difficulty, 2)

    should_diagnose = steps_used >= minimum_investigation or obs.steps_remaining <= 2
    if should_diagnose:
        return build_diagnosis_action(difficulty)

    available = [a for a in NON_TERMINAL_ACTIONS if a in obs.available_actions]
    action_name = agent.select_action(
        observation=obs,
        available_actions=available,
        task_difficulty=difficulty,
        use_greedy=not train_mode,
    )
    return INVESTIGATION_DEFAULTS[action_name](difficulty)


def run_episode(env: MLDebugEnv, agent: DQNAgent, train_mode: bool = True) -> Dict[str, float]:
    obs = env.reset()
    rewards_by_difficulty: Dict[str, float] = {}
    task_trace: List[Dict] = []
    task_actions: List[str] = []

    while True:
        current_difficulty = obs.difficulty
        action = choose_action(agent, obs, train_mode=train_mode)
        prev_obs = obs.model_dump()

        next_obs, reward, done, info = env.step(action)
        transition = {
            "state": prev_obs,
            "action": action.action_type,
            "reward": reward.score,
            "next_state": None if action.action_type == "diagnose" else next_obs.model_dump(),
            "done": action.action_type == "diagnose",
        }
        task_trace.append(transition)
        task_actions.append(action.action_type)

        if action.action_type == "diagnose":
            rewards_by_difficulty[current_difficulty] = reward.score
            if train_mode:
                agent.learn_from_episode(
                    task_difficulty=current_difficulty,
                    actions_used=list(task_actions),
                    final_reward=reward.score,
                    episode_trace=list(task_trace),
                )
            task_trace = []
            task_actions = []

        obs = next_obs
        if done:
            break

    return rewards_by_difficulty


def train_agent(num_episodes: int = 500, checkpoint_path: str = "agent_checkpoint.pkl") -> None:
    print(f"Starting RL agent training for {num_episodes} episodes...")
    print("=" * 60)

    env = MLDebugEnv(max_steps_per_task=15, randomize_tasks=True)
    agent = DQNAgent(learning_rate=0.15, epsilon=0.30, gamma=0.95)
    rewards_by_difficulty = {"easy": [], "medium": [], "hard": []}

    for episode in range(1, num_episodes + 1):
        episode_rewards = run_episode(env, agent, train_mode=True)
        for difficulty in rewards_by_difficulty:
            rewards_by_difficulty[difficulty].append(episode_rewards.get(difficulty, 0.0))

        if episode % 50 == 0:
            avg_easy = np.mean(rewards_by_difficulty["easy"][-50:])
            avg_med = np.mean(rewards_by_difficulty["medium"][-50:])
            avg_hard = np.mean(rewards_by_difficulty["hard"][-50:])
            print(
                f"Episode {episode:4d} | "
                f"Easy: {avg_easy:.3f} | "
                f"Medium: {avg_med:.3f} | "
                f"Hard: {avg_hard:.3f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        if episode % 100 == 0:
            agent.save(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

    agent.save(checkpoint_path)

    print("=" * 60)
    print("Training complete.")
    print(f"Saved agent to {checkpoint_path}")

    stats = agent.get_stats()
    print("\nFinal statistics:")
    print(f"  Total episodes trained: {stats.get('total_episodes', 0)}")
    print(f"  Epsilon: {stats.get('epsilon', 0.0)}")
    print(f"  Strategies learned: {stats.get('strategies_learned', False)}")

    for difficulty, avg_reward in stats.get("average_rewards_by_difficulty", {}).items():
        print(f"  Avg reward ({difficulty}): {avg_reward:.3f}")

    print("\nBest learned investigation sequences:")
    for difficulty, actions in agent.strategy_templates.items():
        if actions:
            print(f"  {difficulty.upper()}: {' -> '.join(actions)} -> diagnose")
        else:
            print(f"  {difficulty.upper()}: still exploring")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL agent for ML debugging strategies")
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="agent_checkpoint.pkl",
        help="Path to save the trained agent (default: agent_checkpoint.pkl)",
    )

    args = parser.parse_args()
    train_agent(num_episodes=args.episodes, checkpoint_path=args.checkpoint)
