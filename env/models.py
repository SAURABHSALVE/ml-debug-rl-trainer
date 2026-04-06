from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ActionType(str):
    fetch_logs        = "fetch_logs"
    fetch_config      = "fetch_config"
    fetch_loss_curve  = "fetch_loss_curve"
    fetch_diagnostics = "fetch_diagnostics"
    fetch_class_data  = "fetch_class_data"
    diagnose          = "diagnose"


ALL_ACTION_TYPES = [
    "fetch_logs",
    "fetch_config",
    "fetch_loss_curve",
    "fetch_diagnostics",
    "fetch_class_data",
    "diagnose",
]


class Action(BaseModel):
    """
    Unified action model — action_type is the discriminator.

    Investigation actions (non-terminal):
      fetch_logs        → epochs: "1-10" | "15-20" | "all"
      fetch_config      → keys: ["lr", "optimizer", ...]  (omit = return all)
      fetch_loss_curve  → split: "train" | "val" | "all"
      fetch_diagnostics → check: "overfitting" | "gradients" | "trends" | "class_balance"
      fetch_class_data  → class_id: 0-4

    Terminal action (ends current task):
      diagnose          → diagnosis, fix_type, fix_detail, confidence
    """
    action_type: str

    # fetch_logs
    epochs: Optional[str] = None          # e.g. "15-20", "all", "1-10"

    # fetch_config
    keys: Optional[List[str]] = None      # e.g. ["lr", "optimizer"]

    # fetch_loss_curve
    split: Optional[str] = None           # "train" | "val" | "all"

    # fetch_diagnostics
    check: Optional[str] = None           # "overfitting" | "gradients" | "trends" | "class_balance"

    # fetch_class_data
    class_id: Optional[int] = None

    # diagnose (terminal)
    diagnosis:  Optional[str]   = None
    fix_type:   Optional[str]   = None    # config_change | data_fix | architecture_change
    fix_detail: Optional[str]   = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class Observation(BaseModel):
    """
    What the agent sees after each step.
    training_logs / config_yaml / loss_curve / gpu_metrics are populated only on
    task entry (reset / first step of each task) so the UI can display context.
    After that the agent uses fetch_* actions to query specific data.
    """
    task_id:         str
    difficulty:      str
    description:     str
    step_number:     int
    max_steps:       int         = 15
    steps_remaining: int
    action_result:   Optional[str]       = None   # result of the last action
    action_history:  List[str]           = []     # rolling last-5 action summaries
    available_actions: List[str]         = []
    hint:            Optional[str]       = None   # unlocked after step 8
    # Full task context — populated on task entry for UI display
    training_logs:   Optional[List[str]] = None
    config_yaml:     Optional[str]       = None
    loss_curve:      Optional[Dict]      = None
    gpu_metrics:     Optional[Dict]      = None


class Reward(BaseModel):
    """
    Returned on every step.

    Investigation actions:
      score = intermediate_signal in [-0.01, 0.03]
        +0.01–0.03 for relevant queries
        -0.01 for redundant/invalid queries (RL penalty)

    diagnose (terminal per task):
      score = fix_quality + efficiency_bonus  (capped at 1.0)
      fix_quality      = grader score 0.0–1.0
      efficiency_bonus = 0.0–0.05 based on steps used
    """
    score:                     float = Field(ge=-0.1, le=1.0)
    fix_quality:               float = 0.0
    efficiency_bonus:          float = 0.0
    intermediate_signal:       float = 0.0
    cumulative_episode_signal: float = 0.0
    reasoning:                 str   = ""
    steps_used:                int   = 0
