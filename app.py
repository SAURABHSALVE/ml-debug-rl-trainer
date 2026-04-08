"""
ML Experiment Debugger — FastAPI server (OpenEnv compatible)

Endpoints:
  POST /reset            → start new episode, returns first Observation
  POST /step             → submit one action, returns {observation, reward, done, info}
  GET  /state            → current episode state
  GET  /tasks            → list all 3 tasks in the current episode
  GET  /health           → service health check
  GET  /docs             → Swagger UI (auto-generated)
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ml_env.environment import MLDebugEnv
from ml_env.models import Action, Observation, Reward

# ─── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("ml_debug.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ml_debug_env")

# ─── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Experiment Debugger — OpenEnv",
    description=(
        "Multi-step RL environment for diagnosing broken ML training runs.\n\n"
        "**Episode flow:**\n"
        "1. `POST /reset` — start episode, get first task observation\n"
        "2. `POST /step` — submit investigation tool calls, collect clues\n"
        "3. `POST /step` with `action_type=diagnose` — submit root cause + fix, receive score\n"
        "4. Repeat for all 3 tasks per episode\n\n"
        "**Reward:** 0.0–1.0 per task (diagnosis correctness + fix quality + efficiency bonus)\n"
        "**Task pool:** 6 scenarios across 3 difficulty levels; 3 selected randomly per episode."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (stateful, one per server process)
env = MLDebugEnv()

# ─── Error Handlers ────────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(status_code=422, content={"error": "Validation error", "detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error(f"Runtime error: {exc}")
    return JSONResponse(status_code=400, content={"error": "Runtime error", "detail": str(exc)})


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": "See server logs"})


# ─── Core Endpoints ────────────────────────────────────────────────────────────

@app.post(
    "/reset",
    response_model=Observation,
    summary="Reset episode — start a new episode with 3 randomly selected tasks",
    tags=["core"],
)
def reset() -> Observation:
    """
    Start a new episode. Picks **1 easy + 1 medium + 1 hard** task from the 6-task pool.

    Returns the first task's **Observation** — the agent sees only the task description
    and must call investigation tools to gather evidence before diagnosing.
    """
    try:
        obs = env.reset()
        logger.info(
            f"Episode reset | tasks: {[t['task_id'] for t in env._tasks]}"
        )
        return obs
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post(
    "/step",
    summary="Submit one action — investigation tool call or terminal diagnose",
    tags=["core"],
)
def step(action: Action) -> Dict[str, Any]:
    """
    Submit one action and advance the environment.

    **Investigation tools** (cost 1 step, reveal information):
    - `fetch_logs` — get training log lines (`start_epoch`, `end_epoch`)
    - `fetch_config` — get hyperparameter config (`keys: [\"lr\", \"optimizer\", ...]`)
    - `fetch_loss_curve` — get loss curves (`split: \"train\" | \"val\"`)
    - `fetch_gpu_metrics` — get GPU memory + utilization
    - `fetch_class_metrics` — get per-class accuracy (`class_id: 0-9`)

    **Terminal action** (ends current task, triggers grading):
    - `diagnose` — submit root cause + fix (`diagnosis`, `fix_type`, `fix_detail`, `confidence`)

    Returns `{observation, reward, done, info}`.
    `done=true` means the entire episode is complete (all 3 tasks finished).
    """
    if env._current_task is None:
        raise HTTPException(
            status_code=400,
            detail="No active task. Call POST /reset to start an episode.",
        )

    task_id = env._current_task["task_id"]
    logger.info(f"Step | task={task_id} step={env._task_step} action={action.action_type}")

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Step error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Step execution failed")

    if action.action_type == "diagnose":
        logger.info(
            f"Diagnosis | task={task_id} score={reward.total:.3f} feedback={reward.feedback[:80]}"
        )

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


# ─── Introspection Endpoints ───────────────────────────────────────────────────

@app.get(
    "/state",
    summary="Get current episode state",
    tags=["introspection"],
)
def state() -> Dict[str, Any]:
    """
    Returns the current episode's full state:
    - Active task ID and difficulty
    - Steps used and remaining
    - Tools called so far
    - Scores for completed tasks
    """
    return env.state()


@app.get(
    "/tasks",
    summary="List all 3 tasks in the current episode",
    tags=["introspection"],
)
def list_tasks() -> Dict[str, Any]:
    """
    Returns the task ID, difficulty, and description for all 3 tasks in the current episode.
    Call `POST /reset` first to populate the task list.
    """
    tasks = env.list_tasks()
    return {
        "episode_active": env._current_task is not None,
        "task_count": len(tasks),
        "tasks": tasks,
    }


@app.get(
    "/health",
    summary="Service health check",
    tags=["meta"],
)
def health() -> Dict[str, str]:
    """Returns `ok` if the server is running correctly."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "episode_active": str(env._current_task is not None),
    }


# ─── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("ML Experiment Debugger v2.0.0 starting up — OpenEnv compatible")


@app.on_event("shutdown")
async def shutdown():
    logger.info("ML Experiment Debugger shutting down")