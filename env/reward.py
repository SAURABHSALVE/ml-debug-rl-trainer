"""
Trajectory-level reward shaping and efficiency bonus.

Trajectory bonus (applied to hard task final score):
  If easy_score > 0.7 AND medium_score > 0.6  →  +0.05 on hard score

Efficiency bonus (applied per submit_fix):
  Rewards agents that solve tasks with fewer tool calls.
  steps_used counts steps including the submit_fix call itself.
"""


def compute_efficiency_bonus(steps_used: int, max_steps: int) -> float:
    """Return a small bonus for efficient (fewer-step) solutions."""
    ratio = steps_used / max(1, max_steps)
    if ratio <= 0.33:
        return 0.05
    elif ratio <= 0.50:
        return 0.03
    elif ratio <= 0.67:
        return 0.01
    return 0.0


def apply_trajectory_bonus(
    easy_score: float,
    medium_score: float,
    hard_raw: float,
) -> float:
    """Return the final hard-task score with optional consistency bonus."""
    bonus = 0.05 if (easy_score > 0.7 and medium_score > 0.6) else 0.0
    return round(min(1.0, hard_raw + bonus), 3)


def compute_episode_summary(scores: dict) -> dict:
    """
    Given {'easy': float, 'medium': float, 'hard': float},
    return final scores and average.
    """
    easy = scores.get("easy", 0.0)
    medium = scores.get("medium", 0.0)
    hard = scores.get("hard", 0.0)
    avg = round((easy + medium + hard) / 3, 3)
    return {
        "easy_score": easy,
        "medium_score": medium,
        "hard_score": hard,
        "average_score": avg,
    }
