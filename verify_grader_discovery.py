import sys
import os
import yaml

# Add current dir to path
sys.path.append(os.getcwd())

from ml_env.environment import MLDebugEnv

def verify_grader_discovery():
    print("--- STARTING GRADER DISCOVERY VERIFICATION ---")
    
    # 1. Load the manifest
    with open("openenv.yaml", "r") as f:
        manifest = yaml.safe_load(f)
    
    manifest_graders = {t["id"]: t["grader"] for t in manifest["tasks"]}
    print(f"Manifest Graders: {manifest_graders}")
    
    # 2. Check the Environment Task List
    env = MLDebugEnv()
    tasks = env.list_tasks()
    
    match_count = 0
    for t in tasks:
        task_id = t["task_id"]
        env_grader = t.get("grader")
        manifest_grader = manifest_graders.get(task_id)
        
        print(f"Task: {task_id:25} | Env: {env_grader:15} | Manifest: {manifest_grader:15}")
        
        if env_grader == manifest_grader and t.get("has_grader") is True:
            print("  ✅ MATCH")
            match_count += 1
        else:
            print("  ❌ MISMATCH or Missing has_grader")
    
    if match_count == 6:
        print("\n✅ SUCCESS: All 6 tasks have synchronized grader metadata.")
    else:
        print(f"\n❌ FAILURE: Only {match_count}/6 tasks matched.")
        sys.exit(1)

if __name__ == "__main__":
    verify_grader_discovery()
