"""
FastAPI app with comprehensive error handling and validation.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from env.environment import MLDebugEnv
from env.models import Action, Observation, Reward

# ─── Logging ───────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_debug.log"),
        logging.StreamHandler(),
    ]
)

# ─── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Experiment Debugger — OpenEnv",
    description="Multi-step RL environment for diagnosing broken ML training runs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = MLDebugEnv()


# ─── Error Handlers ────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)},
    )


@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    logger.error(f"TimeoutError: {exc}")
    return JSONResponse(
        status_code=504,
        content={"error": "Request timeout", "detail": "LLM grading took too long"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "See server logs"},
    )


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation, summary="Reset episode")
def reset() -> Observation:
    """Start a new episode with randomly selected tasks."""
    try:
        logger.info(f"Episode reset requested")
        obs = env.reset()
        logger.info(f"Episode started with tasks: {env.list_tasks()}")
        return obs
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset: {str(e)}")


@app.post("/step", summary="Submit investigation tool call or diagnose")
def step(action: Action) -> Dict[str, Any]:
    """
    Submit one action (investigation tool or diagnose).
    
    **Investigation tools** (cost 1 step each):
    - fetch_logs: Retrieve training logs
    - fetch_config: Retrieve hyperparameter config
    - fetch_loss_curve: Retrieve train/val loss curves
    - fetch_gpu_metrics: Retrieve GPU/memory metrics
    - fetch_class_metrics: Retrieve per-class accuracy metrics
    
    **Terminal action**:
    - diagnose: Submit diagnosis and fix (ends current task, grades 0.0-1.0)
    """
    try:
        # ✅ Validate preconditions
        if env._current_task is None:
            raise HTTPException(
                status_code=400,
                detail="No active task. Call POST /reset to start an episode."
            )
        
        # ✅ Validate action
        valid_actions = [
            "fetch_logs", "fetch_config", "fetch_loss_curve",
            "fetch_gpu_metrics", "fetch_class_metrics", "diagnose"
        ]
        if action.action_type not in valid_actions:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action: {action.action_type}. Valid: {valid_actions}"
            )
        
        # ✅ Log action
        task_id = env._current_task["task_id"]
        logger.info(
            f"Task {task_id} step {env._task_step}: action={action.action_type}"
        )
        
        # ✅ Execute step
        obs, reward, done, info = env.step(action)
        
        # ✅ Log result
        if action.action_type == "diagnose":
            logger.info(
                f"Task {task_id} graded: score={reward.total}, feedback={reward.feedback[:80]}"
            )
        
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except TimeoutError as e:
        logger.error(f"Timeout during step: {e}")
        raise HTTPException(status_code=504, detail="LLM grading timeout")
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /step: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/state", summary="Get current episode state")
def state() -> Dict[str, Any]:
    """Returns current episode state (task index, steps, tools called, scores)."""
    try:
        return env.state()
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail="Failed to get state")


@app.get("/tasks", summary="List all 3 tasks in current episode")
def list_tasks() -> Dict[str, Any]:
    """Returns task IDs, difficulties, and descriptions."""
    try:
        tasks = env.list_tasks()
        return {
            "episode_active": env._current_task is not None,
            "task_index": env._task_index,
            "tasks": tasks,
        }
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to list tasks")


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    """Service health check."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


# ─── APIRouter for Frontend ──────────────────────────────────────────────────

api_router = APIRouter(prefix="/api")

@api_router.post("/reset", response_model=Observation, summary="Reset episode")
def api_reset():
    return reset()

@api_router.post("/step", summary="Submit investigation tool call or diagnose")
def api_step(action: Action):
    return step(action)

@api_router.get("/state", summary="Get current episode state")
def api_state():
    return state()

@api_router.get("/tasks", summary="List all 3 tasks in current episode")
def api_list_tasks():
    return list_tasks()

@api_router.get("/health", summary="Health check")
def api_health():
    return health()

app.include_router(api_router)


# ─── Startup/Shutdown ───────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    logger.info("ML Experiment Debugger starting up")


@app.on_event("shutdown")
def shutdown():
    logger.info("ML Experiment Debugger shutting down")


# Mount the frontend UI (must be at the end to not shadow API routes)
app.mount("/", StaticFiles(directory="ui/dist", html=True), name="ui")