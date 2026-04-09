import sys
import os
import yaml

# Add current dir to path
sys.path.append(os.getcwd())

from ml_env.environment import MLDebugEnv

def verify_sync():
    print("--- STARTING MANIFEST SYNC VERIFICATION ---")
    
    # 1. Load the manifest
    with open("openenv.yaml", "r") as f:
        manifest = yaml.safe_load(f)
    manifest_ids = set([t["id"] for t in manifest["tasks"]])
    print(f"Manifest IDs: {manifest_ids}")
    
    # 2. Check the Environment Task List
    env = MLDebugEnv()
    tasks = env.list_tasks()
    env_ids = set([t["task_id"] for t in tasks])
    print(f"Environment IDs: {env_ids}")
    
    # 3. Check for mismatches
    missing = manifest_ids - env_ids
    unexpected = env_ids - manifest_ids
    
    if not missing and not unexpected:
        print("\n✅ SUCCESS: Manifest and Environment are 100% synchronized.")
    else:
        if missing:
            print(f"\n❌ FAILURE: Missing tasks in Env: {missing}")
        if unexpected:
            print(f"\n❌ FAILURE: Unexpected tasks in Env: {unexpected}")
        sys.exit(1)

    # 4. Check for Grader References
    for t in tasks:
        print(f"Checking {t['task_id']}...")
        if not t.get("has_grader"):
            print(f"❌ FAILURE: Task {t['task_id']} is missing has_grader=True")
            sys.exit(1)
        if not t.get("grader"):
            print(f"❌ FAILURE: Task {t['task_id']} is missing grader name")
            sys.exit(1)
    
    print("\n✅ Verification PASSED: Grader discovery is operational.")

if __name__ == "__main__":
    verify_sync()
