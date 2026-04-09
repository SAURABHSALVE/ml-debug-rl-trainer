import sys
import os
from typing import Any, Dict

# Add current dir to path
sys.path.append(os.getcwd())

from ml_env.models import Action

def test_422_fix():
    print("--- STARTING 422 ERROR RESOLUTION TEST ---")
    
    # Payload 1: Contains 'fix' instead of 'fix_detail'
    payload_1 = {
        "action_type": "diagnose",
        "diagnosis": "The model is underflowing.",
        "fix": "Use a GradScaler",
        "confidence": 0.5
    }
    
    # Payload 2: Contains UNKNOWN extra fields (common source of 422)
    payload_2 = {
        "action_type": "diagnose",
        "diagnosis": "Data leakage",
        "fix_type": "data_fix",
        "fix_detail": "remove target column",
        "extra_validator_field": "some_value",
        "step_count": 10
    }
    
    try:
        a1 = Action(**payload_1)
        print("✅ SUCCESS: Accepted payload with 'fix' field.")
        assert a1.fix == "Use a GradScaler"
        
        a2 = Action(**payload_2)
        print("✅ SUCCESS: Accepted payload with UNKNOWN extra fields.")
        # pydantic with extra="allow" will keep them in __dict__ or .extra
        assert hasattr(a2, "extra_validator_field")
        
    except Exception as e:
        print(f"❌ FAILURE: Action model still too rigid: {e}")
        sys.exit(1)

    print("\n✅ Verification PASSED: 422 errors should be resolved.")

if __name__ == "__main__":
    test_422_fix()
