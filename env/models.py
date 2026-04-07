from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ─── Action ────────────────────────────────────────────────────────────────────

class Action(BaseModel):
    action_type: str = Field(
        ...,
        description=(
            "One of: fetch_logs | fetch_config | fetch_loss_curve | "
            "fetch_gpu_metrics | fetch_class_metrics | diagnose"
        ),
    )
    # Investigation params (used depending on action_type)
    start_epoch: Optional[int] = Field(None, description="For fetch_logs")
    end_epoch: Optional[int] = Field(None, description="For fetch_logs")
    keys: Optional[List[str]] = Field(None, description="For fetch_config e.g. ['lr','optimizer']")
    split: Optional[str] = Field(None, description="For fetch_loss_curve: 'train' or 'val'")
    class_id: Optional[int] = Field(None, description="For fetch_class_metrics")

    # Diagnose params (terminal action)
    diagnosis: Optional[str] = Field(None, description="Root cause explanation")
    fix_type: Optional[str] = Field(
        None,
        description="One of: config_change | data_fix | architecture_change",
    )
    fix_detail: Optional[str] = Field(None, description="Specific actionable fix")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


# ─── Observation ───────────────────────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    difficulty: str
    description: str
    step_number: int
    max_steps: int
    steps_remaining: int
    tool_result: Optional[Dict[str, Any]] = None
    action_history: List[str] = []
    available_tools: List[str] = [
        "fetch_logs",
        "fetch_config",
        "fetch_loss_curve",
        "fetch_gpu_metrics",
        "fetch_class_metrics",
        "diagnose",
    ]


# ─── Reward ────────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)  # allows small penalty for redundant calls
    breakdown: Dict[str, float] = {}
    feedback: str = ""
    efficiency_bonus: float = 0.0
    trajectory_bonus: float = 0.0
    total: float = 0.0