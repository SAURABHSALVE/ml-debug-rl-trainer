import sys
import os

# Ensure the root directory and ml_env are discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_env.graders import grade_task

def grade(action_data: dict, ground_truth: dict):
    """
    Final, standardized, root-level grader entry point for the validator.
    Takes exactly 2 arguments (action_data, ground_truth).
    """
    try:
        # Route to the main logic in ml_env/graders.py
        return grade_task(action_data, ground_truth)
    except Exception as e:
        print(f"ERROR: Grader execution failed: {e}")
        # Return a fallback score to prevent the entire validation from crashing
        return 0.1, {"error": str(e)}, f"Internal grader error: {e}"
