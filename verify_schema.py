import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from ml_env.models import Action

def verify_schema():
    print("--- STARTING SCHEMA VERIFICATION ---")
    
    test_payload = {
        "action_type": "diagnose",
        "diagnosis": "The labels are poisoned.",
        "fix_type": "data_fix",
        "fix_detail": "Clean class 5 labels",
        "confidence": 0.9
    }
    
    try:
        action = Action(**test_payload)
        print("✅ SUCCESS: Action model validated with fix_type/fix_detail.")
        print(f"Action object: {action}")
        
        # Verify fields are accessible
        assert action.fix_type == "data_fix"
        assert action.fix_detail == "Clean class 5 labels"
        print("✅ SUCCESS: Fields correctly mapped.")
        
    except Exception as e:
        print(f"❌ FAILURE: Schema validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_schema()
