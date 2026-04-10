import json
import urllib.request

ENV_BASE_URL = "http://localhost:7860"

def _post(path, body=None):
    url = f"{ENV_BASE_URL}{path}"
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

print("--- RESET ---")
res = _post("/reset")
print(json.dumps(res, indent=2))

print("\n--- DIAGNOSE ---")
action = {
    "action_type": "diagnose",
    "diagnosis": "test",
    "fix_type": "data_fix",
    "fix_detail": "test",
    "confidence": 0.9
}
res = _post("/step", action)
print(json.dumps(res, indent=2))
