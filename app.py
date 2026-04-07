"""
ML Experiment Debugger — OpenEnv Server
Clean FastAPI app exposing ONLY the OpenEnv-required endpoints.
No agent logic here. Agent lives in inference.py.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List

from env.environment import MLDebugEnv
from env.models import Action, Observation, Reward

app = FastAPI(
    title="ML Experiment Debugger — OpenEnv",
    description=(
        "Multi-step RL environment where agents diagnose broken ML training runs. "
        "Agent calls investigation tools (fetch_logs, fetch_config, fetch_loss_curve, "
        "fetch_gpu_metrics, fetch_class_metrics) to gather evidence step by step, "
        "then calls diagnose to submit the fix. "
        "3 tasks: overfitting (easy), LR explosion (medium), silent data poisoning (hard). "
        "Each investigation costs one step from a limited budget — wrong choices waste budget."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server
env = MLDebugEnv()


# ─── Required OpenEnv Endpoints ────────────────────────────────────────────────

@app.post("/reset", response_model=Observation, summary="Reset episode — returns Task 1 observation")
def reset() -> Observation:
    """
    Start a new episode. Returns the first task (easy) observation.
    Agent receives ONLY the task description — no data yet.
    Agent must call investigation tools to reveal data.
    """
    return env.reset()


@app.post("/step", summary="Submit one action — investigation tool call or diagnose")
def step(action: Action) -> Dict[str, Any]:
    """
    Submit one action.

    **Investigation actions** (non-terminal, reveal data, cost one step):
    - `{"action_type": "fetch_logs", "start_epoch": 1, "end_epoch": 10}`
    - `{"action_type": "fetch_config", "keys": ["lr", "optimizer"]}`
    - `{"action_type": "fetch_loss_curve", "split": "val"}`
    - `{"action_type": "fetch_gpu_metrics"}`
    - `{"action_type": "fetch_class_metrics", "class_id": 2}`

    **Terminal action** (ends current task, graded 0.0–1.0):
    - `{"action_type": "diagnose", "diagnosis": "...", "fix_type": "config_change", "fix_detail": "...", "confidence": 0.9}`

    fix_type must be one of: config_change | data_fix | architecture_change
    """
    try:
        obs, reward, done, info = env.step(action)
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
    """
    Returns current episode state including task index, steps used,
    tools called so far, and scores on completed tasks.
    """
    return env.state()


@app.get("/tasks", summary="List all 3 tasks in the current episode")
def list_tasks() -> List[Dict[str, Any]]:
    """Lists task IDs, difficulties, and descriptions for current episode."""
    return env.list_tasks()


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}