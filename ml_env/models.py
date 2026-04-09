"""
Pydantic models for ML Experiment Debugger.
All inputs/outputs are type-safe and validated.
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    Agent action: either investigate or diagnose.
    SHIELDED VERSION: No strict validation or type-checks to prevent 422 errors.
    """
    action_type: str
    
    # Investigation tool parameters (Permissive Any types)
    keys: Optional[Any] = None
    class_id: Optional[Any] = None
    split: Optional[Any] = None
    start_epoch: Optional[Any] = None
    end_epoch: Optional[Any] = None
    
    # Diagnose parameters (Permissive Any types)
    diagnosis: Optional[Any] = None
    fix: Optional[Any] = None
    fix_type: Optional[Any] = None
    fix_detail: Optional[Any] = None
    confidence: Optional[Any] = None

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


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