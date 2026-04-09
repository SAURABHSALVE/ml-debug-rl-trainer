import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from ml_env.environment import MLDebugEnv
from ml_env.models import Action

def verify_master_floor():
    print("--- STARTING MASTER FLOOR VERIFICATION ---")
    
    env = MLDebugEnv()
    env.reset()
    
    # Send a diagnosis with 0 investigation
    # In natural logic, score = 0.0, investigation = 0.0, penalty = 0.5
    # Result should be 0.0. BUT our master floor should force it to 0.25.
    
    action = Action(
        action_type="diagnose",
        diagnosis="Wrong diagnosis",
        fix_type="none",
        fix_detail="none"
    )
    
    obs, reward, done, info = env.step(action)
    
    score = reward.total
    print(f"Final Total Score: {score}")
    
    if score == 0.25:
        print("\n✅ SUCCESS: Master floor (0.25) is active and enforced.")
    elif score > 0.25:
        print(f"\n✅ SUCCESS: Score is {score} (above floor).")
    else:
        print(f"\n❌ FAILURE: Score is {score}. Master floor not found.")
        sys.exit(1)

if __name__ == "__main__":
    verify_master_floor()
