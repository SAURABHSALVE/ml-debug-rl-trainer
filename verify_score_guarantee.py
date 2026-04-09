import sys
import os
import logging

# Add current dir to path
sys.path.append(os.getcwd())

from ml_env.environment import MLDebugEnv
from ml_env.models import Action

def verify_score_guarantee():
    print("--- STARTING FINAL SCORE GUARANTEE VERIFICATION ---")
    
    # Configure logging to stdout
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    
    env = MLDebugEnv()
    env.reset()
    
    # We send a diagnosis attempt. 
    # Current logic: total = max(0.3, total) if diagnosis else total
    # Environment logic: total = score * (1.0 - penalty)
    # Penalty for early guessing (steps < 5) is 0.5
    # Expected result: 0.3 * 0.5 = 0.15
    
    action = Action(
        action_type="diagnose",
        diagnosis="Standardized test diagnosis",
        fix_type="none",
        fix_detail="none"
    )
    
    print("\n[LOG TRACE START]")
    obs, reward, done, info = env.step(action)
    print("[LOG TRACE END]\n")
    
    score = reward.total
    print(f"Final Score: {score}")
    
    if score >= 0.15:
        print("\n✅ SUCCESS: Non-zero score guaranteed.")
    else:
        print("\n❌ FAILURE: Score is too low. Check the floor logic.")
        sys.exit(1)

if __name__ == "__main__":
    verify_score_guarantee()
