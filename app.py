from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from env.environment import MLDebugEnv
from env.models import Action, Observation
from env.rl_agent import DQNAgent
from typing import Dict, Any, List
import os

app = FastAPI(
    title="ML Experiment Debugger — OpenEnv",
    description=(
        "Multi-step RL environment where agents diagnose broken ML training runs. "
        "Agent calls fetch_* actions to gather evidence, then calls diagnose to submit the fix. "
        "3 tasks: overfitting (easy), LR explosion (medium), silent data poisoning (hard). "
        "Includes RL agent that learns optimal debugging strategies."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = MLDebugEnv(max_steps_per_task=15)
rl_agent = DQNAgent(learning_rate=0.01, epsilon=0.15, gamma=0.99)

# Load pre-trained agent if available
AGENT_CHECKPOINT = "agent_checkpoint.pkl"
rl_agent.load(AGENT_CHECKPOINT)


@app.post("/reset", response_model=Observation, summary="Reset episode — returns task 1 observation")
def reset():
    return env.reset()


@app.post("/step", summary="Submit one action")
def step(action: Action) -> Dict[str, Any]:
    """
    Submit one action:

    **Investigation** (non-terminal, returns action_result + small reward):
    - `{"action_type": "fetch_logs", "epochs": "15-20"}`
    - `{"action_type": "fetch_config", "keys": ["lr", "optimizer"]}`
    - `{"action_type": "fetch_loss_curve", "split": "val"}`
    - `{"action_type": "fetch_diagnostics", "check": "overfitting"}`
    - `{"action_type": "fetch_class_data", "class_id": 2}`

    **Terminal** (ends current task, returns graded reward):
    - `{"action_type": "diagnose", "diagnosis": "...", "fix_type": "config_change", "fix_detail": "...", "confidence": 0.9}`
    """
    try:
        obs, reward, done, info = env.step(action)
        
        # Learn from episode if terminal action
        if action.action_type == "diagnose":
            task_difficulty = obs.task_difficulty if hasattr(obs, 'task_difficulty') else "easy"
            rl_agent.learn_from_episode(
                task_difficulty=task_difficulty,
                actions_used=env._action_history,
                final_reward=reward.score
            )
            # Save learned strategies periodically
            if len(rl_agent.training_history) % 50 == 0:
                rl_agent.save(AGENT_CHECKPOINT)
        
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", summary="Get current episode state")
def state() -> Dict[str, Any]:
    return env.state()


@app.get("/tasks", summary="List all 3 tasks")
def list_tasks() -> List[Dict[str, Any]]:
    return env.list_tasks()


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.post("/recommend-actions", summary="Get AI-recommended next actions")
def recommend_actions(limit: int = 5) -> Dict[str, Any]:
    """
    Get recommended next actions based on learned debugging strategies.
    Helps user see what the RL agent learned works best.
    """
    if env._current_task is None:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")
    
    task_difficulty = env._current_task.difficulty
    steps_remaining = env.max_steps_per_task - env._task_step
    
    recommended = rl_agent.get_recommended_actions(task_difficulty, steps_remaining)
    
    return {
        "task_difficulty": task_difficulty,
        "steps_remaining": steps_remaining,
        "recommended_actions": recommended[:limit],
        "reasoning": f"Based on patterns learned from {len(rl_agent.training_history)} debugging episodes"
    }


@app.get("/agent-stats", summary="Get RL agent training statistics")
def agent_stats() -> Dict[str, Any]:
    """
    Return summarized statistics about the RL agent's learning progress.
    Shows what debugging strategies it has learned.
    """
    stats = rl_agent.get_stats()
    
    # Add strategy summaries
    strategies = {}
    for difficulty, actions in rl_agent.strategy_templates.items():
        strategies[difficulty] = {
            "action_sequence": actions,
            "count": len(actions)
        }
    
    return {
        "agent_stats": stats,
        "learned_strategies": strategies,
        "total_training_episodes": len(rl_agent.training_history),
        "message": "Agent learns from every episode. Better strategies = higher average rewards."
    }


@app.post("/suggest-diagnosis", summary="Get AI assistance on diagnosis")
def suggest_diagnosis() -> Dict[str, Any]:
    """
    AI advisor examines current evidence and suggests a diagnosis direction.
    Helps when you're stuck!
    """
    if env._current_task is None:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")
    
    task = env._current_task
    difficulty = task.difficulty
    ground_truth = task.ground_truth
    
    # Generate context-aware suggestion
    bug_type = ground_truth.get("bug_type", "unknown")
    
    suggestions = {
        "easy": {
            "bug_type": "overfitting",
            "hint": "Look at val vs train loss curves. If val loss is much worse, that's your answer.",
            "fix_categories": ["regularization", "data augmentation", "early stopping"]
        },
        "medium": {
            "bug_type": "learning_rate_explosion",
            "hint": "Check loss trend in logs. NaN or rapid divergence? Check learning rate in config.",
            "fix_categories": ["reduce_learning_rate", "use_scheduler", "gradient_clipping"]
        },
        "hard": {
            "bug_type": "silent_data_poisoning",
            "hint": "Some class has suspiciously bad accuracy. Fetch data for low-performing classes.",
            "fix_categories": ["remove_corrupted_samples", "clean_labels", "retrain_on_clean_data"]
        }
    }
    
    suggestion = suggestions.get(difficulty, {})
    
    return {
        "current_difficulty": difficulty,
        "expected_bug_type": suggestion.get("bug_type"),
        "investigation_hint": suggestion.get("hint"),
        "fix_suggestions": suggestion.get("fix_categories"),
        "steps_used": env._task_step,
        "steps_remaining": env.max_steps_per_task - env._task_step
    }
