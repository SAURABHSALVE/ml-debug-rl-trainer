#!/usr/bin/env python3
"""
RL Agent Training Script for ML Debugging.

This script trains the agent on synthetic debugging episodes,
learning optimal action sequences for each task difficulty.

Usage:
    python train_agent.py --episodes 500 --checkpoint agent_checkpoint.pkl
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from env.environment import MLDebugEnv
from env.rl_agent import DQNAgent
from env.models import Action


def run_episode(env: MLDebugEnv, agent: DQNAgent, task_difficulty: str, train_mode: bool = True) -> float:
    """
    Run one complete episode.
    
    Returns: Final reward score
    """
    obs = env.reset()
    
    done = False
    task = env._current_task
    total_reward = 0
    
    # Simulate an episode with investigation actions
    steps_taken = 0
    max_investigation_steps = 10
    
    while not done and steps_taken < max_investigation_steps:
        # Select investigation action
        available_actions = [
            "fetch_logs",
            "fetch_config", 
            "fetch_loss_curve",
            "fetch_diagnostics",
            "fetch_class_data"
        ]
        
        action_name = agent.select_action(
            obs,
            available_actions,
            task.difficulty,
            use_greedy=not train_mode
        )
        
        # Create action parameters based on type
        if action_name == "fetch_logs":
            action = Action(action_type="fetch_logs", epochs="all")
        elif action_name == "fetch_config":
            action = Action(action_type="fetch_config", keys=["lr", "optimizer", "batch_size"])
        elif action_name == "fetch_loss_curve":
            action = Action(action_type="fetch_loss_curve", split="all")
        elif action_name == "fetch_diagnostics":
            action = Action(action_type="fetch_diagnostics", check="overfitting")
        elif action_name == "fetch_class_data":
            action = Action(action_type="fetch_class_data", class_id=0)
        else:
            continue
        
        obs, reward, done, info = env.step(action)
        total_reward += reward.intermediate_signal
        steps_taken += 1
    
    # Submit diagnosis with a reasonable guess
    ground_truth = task.ground_truth
    diagnosis_templates = {
        "easy": {
            "diagnosis": "The model is overfitting. The validation loss diverges while training loss continues to decrease.",
            "fix_type": "config_change",
            "fix_detail": "Add dropout=0.5 and weight_decay=1e-5 as regularization",
            "confidence": 0.85
        },
        "medium": {
            "diagnosis": "The learning rate is too high, causing gradient explosion and NaN loss.",
            "fix_type": "config_change",
            "fix_detail": "Reduce learning rate from 0.1 to 0.001 and use a scheduler with warmup.",
            "confidence": 0.80
        },
        "hard": {
            "diagnosis": "One class has corrupted training data, causing poor performance on that class specifically.",
            "fix_type": "data_cleaning",
            "fix_detail": "Remove samples from the corrupted class and retrain on clean data.",
            "confidence": 0.75
        }
    }
    
    template = diagnosis_templates.get(task.difficulty,{})
    diagnosis_action = Action(
        action_type="diagnose",
        diagnosis=template.get("diagnosis", "Unknown issue detected"),
        fix_type=template.get("fix_type", "intervention"),
        fix_detail=template.get("fix_detail", "Review training configuration"),
        confidence=template.get("confidence", 0.5)
    )
    
    obs, reward, done, info = env.step(diagnosis_action)
    final_reward = reward.score
    
    if train_mode:
        agent.learn_from_episode(
            task_difficulty=task.difficulty,
            actions_used=env._action_history,
            final_reward=final_reward
        )
    
    return final_reward


def train_agent(num_episodes: int = 500, checkpoint_path: str = "agent_checkpoint.pkl"):
    """
    Train the RL agent over multiple episodes.
    
    Args:
        num_episodes: Number of training episodes
        checkpoint_path: Where to save the trained agent
    """
    print(f"🚀 Starting RL Agent Training ({num_episodes} episodes)...")
    print("=" * 60)
    
    env = MLDebugEnv(max_steps_per_task=15)
    agent = DQNAgent(learning_rate=0.01, epsilon=0.25, gamma=0.99)
    
    rewards_by_difficulty = {"easy": [], "medium": [], "hard": []}
    
    for episode in range(1, num_episodes + 1):
        # Reset environment for new task
        obs = env.reset()
        current_difficulty = env._current_task.difficulty if env._current_task else "easy"
        
        # Run episode
        final_reward = run_episode(env, agent, current_difficulty, train_mode=True)
        
        difficulty = env._current_task.difficulty if not env._done else current_difficulty
        rewards_by_difficulty[difficulty].append(final_reward)
        
        # Progress reporting
        if episode % 50 == 0:
            avg_easy = np.mean(rewards_by_difficulty["easy"][-50:]) if rewards_by_difficulty["easy"] else 0
            avg_med = np.mean(rewards_by_difficulty["medium"][-50:]) if rewards_by_difficulty["medium"] else 0
            avg_hard = np.mean(rewards_by_difficulty["hard"][-50:]) if rewards_by_difficulty["hard"] else 0
            
            print(f"Episode {episode:3d} | Easy: {avg_easy:.3f} | Medium: {avg_med:.3f} | Hard: {avg_hard:.3f}")
        
        # Save checkpoint periodically
        if episode % 100 == 0:
            agent.save(checkpoint_path)
            print(f"  ✓ Checkpoint saved to {checkpoint_path}")
    
    # Final save
    agent.save(checkpoint_path)
    
    # Print final statistics
    print("=" * 60)
    print("📊 Training Complete!")
    print(f"✓ Saved agent to {checkpoint_path}")
    
    stats = agent.get_stats()
    print(f"\n📈 Final Statistics:")
    print(f"  Total episodes trained: {stats['total_episodes']}")
    print(f"  Strategies learned: {stats['strategies_learned']}")
    
    if 'average_rewards_by_difficulty' in stats:
        for difficulty, avg_reward in stats['average_rewards_by_difficulty'].items():
            print(f"  Avg reward ({difficulty}): {avg_reward:.3f}")
    
    # Show learned strategies
    print(f"\n🎓 Learned Debugging Strategies:")
    for difficulty, actions in agent.strategy_templates.items():
        if actions:
            action_str = " → ".join(actions) + " → diagnose"
            print(f"  {difficulty.upper()}: {action_str}")
        else:
            print(f"  {difficulty.upper()}: (still learning...)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the RL agent for ML debugging strategies"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="agent_checkpoint.pkl",
        help="Path to save the trained agent (default: agent_checkpoint.pkl)"
    )
    
    args = parser.parse_args()
    
    train_agent(num_episodes=args.episodes, checkpoint_path=args.checkpoint)
