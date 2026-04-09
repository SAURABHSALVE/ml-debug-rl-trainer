import os
import yaml
import sys
sys.path.append(os.getcwd())
from ml_env.environment import MLDebugEnv

with open("openenv.yaml", "r") as f:
    manifest = yaml.safe_load(f)
manifest_ids = sorted([t["id"] for t in manifest["tasks"]])

env = MLDebugEnv()
env_ids = sorted([t["task_id"] for t in env.list_tasks()])

print(f"MANIFEST: {manifest_ids}")
print(f"ENV:      {env_ids}")

if manifest_ids == env_ids:
    print("MATCH!")
else:
    print("MISMATCH!")
