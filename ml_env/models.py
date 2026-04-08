"""
Pydantic models for ML Experiment Debugger.
All inputs/outputs are type-safe and validated.
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, validator


class Action(BaseModel):
    """Agent action: either investigate or diagnose."""
    action_type: str
    
    # Investigation tool parameters (optional)
    keys: Optional[List[str]] = None
    class_id: Optional[int] = None
    split: Optional[str] = None
    start_epoch: Optional[int] = None
    end_epoch: Optional[int] = None
    
    # Diagnose parameters (optional)
    diagnosis: Optional[str] = None
    fix_type: Optional[str] = None
    fix_detail: Optional[str] = None
    confidence: Optional[float] = None

    @validator("action_type")
    def validate_action_type(cls, v: str) -> str:
        valid = [
            "fetch_logs", "fetch_config", "fetch_loss_curve",
            "fetch_gpu_metrics", "fetch_class_metrics", "diagnose"
        ]
        if v not in valid:
            raise ValueError(f"action_type must be one of {valid}")
        return v

    @validator("class_id")
    def validate_class_id(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (0 <= v < 10):
            raise ValueError("class_id must be in [0, 10)")
        return v

    @validator("confidence")
    def validate_confidence(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        return v


class Reward(BaseModel):
    """Reward structure from a step."""
    score: float = Field(..., ge=-1.0, le=1.0)
    breakdown: Dict[str, float]
    feedback: str
    total: float = Field(..., ge=-1.0, le=1.0)
    efficiency_bonus: float = Field(default=0.0, ge=0.0, le=0.05)
    trajectory_bonus: float = Field(default=0.0, ge=0.0, le=0.05)
    path_bonus: float = Field(default=0.0, ge=0.0, le=0.1)


class Observation(BaseModel):
    """Observation returned to agent after each step."""
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    step_number: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    steps_remaining: int = Field(..., ge=0)
    tool_result: Optional[Dict[str, Any]] = None
    loss_curve: Optional[Dict[str, Any]] = None
    class_metrics: Optional[Dict[str, Any]] = None
    logs: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    gpu_metrics: Optional[Dict[str, Any]] = None
    action_history: List[str] = Field(default_factory=list)
    available_tools: List[str] = Field(default_factory=list)