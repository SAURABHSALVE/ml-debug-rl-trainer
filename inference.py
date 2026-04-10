"""
Baseline inference script for ML Experiment Debugger.
Uses OpenAI-compatible client to run an LLM agent through all 3 tasks per episode.

Usage:
  export API_BASE_URL=https://router.huggingface.co/v1
  export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
  export HF_TOKEN=your_token_here
  python inference.py

  # With LLM grading enabled:
  export USE_LLM_GRADING=true
  export GRADER_MODEL=meta-llama/Llama-3.3-70B-Instruct
  python inference.py
"""

import json
import os
import time
import urllib.request
import ssl
import signal

from openai import OpenAI

# ─── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
# Switch to 8B for extreme speed to beat the 30min timeout
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# GLOBAL START TIME for 30-min kill safety
EPISODE_START_TIME = time.time()
TIMEOUT_LIMIT      = 25 * 60  # 25 minutes buffer

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ─── HTTP Helpers ───────────────────────────────────────────────────────────────

def _request(path: str, method: str = "GET", body: dict = None) -> dict:
    url = f"{ENV_BASE_URL}{path}"
    data = json.dumps(body or {}).encode() if body else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError, ssl.SSLError) as e:
            if attempt == max_retries - 1:
                print(f"\n  [ERROR] Network Failure on {method} {path}: {str(e)}")
                raise e
            wait = 1.0 * (attempt + 1)
            print(f"\n  [RETRY] Network Hiccup ({str(e)}). Retrying in {wait}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait)
    return {}

def _post(path: str, body: dict = None) -> dict:
    return _request(path, method="POST", body=body)

def _get(path: str) -> dict:
    return _request(path, method="GET")


# ─── System Prompt (DECISIVE MODE) ───────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert ML debugging agent. Your job is to diagnose and fix machine learning issues with high precision.

## STEP BUDGET (MOST IMPORTANT RULE):
- You have MAX 3-4 steps per task. Budget them carefully.
- NEVER call the same tool twice.
- Preferred tool order: fetch_config → fetch_logs → diagnose
- Only call fetch_loss_curve if config+logs are not enough.
- If you are unsure, DIAGNOSE IMMEDIATELY — a guess is better than wasting steps.
- On your FINAL step, you MUST output a diagnose action no matter what.

## CRITICAL RULES:
1. NEVER default to "overfitting" or "regularization" without clear evidence.
2. Match symptoms carefully to the bug guide below.
3. One tool call per step — do not chain multiple actions.
4. Always output pure JSON with zero extra text.

## BUG IDENTIFICATION GUIDE:

### NaN INITIALIZATION (Easy)
- Signs: Loss is NaN from step 0, gradients are NaN immediately, no training progress at all
- Fix: Fix weight initialization (use Xavier/He init), clip gradients, normalize inputs to [-1,1] or [0,1]
- fix_type: "NaN Initialization Fix"

### DATA LEAKAGE (Easy)
- Signs: Suspiciously high train AND val accuracy from early epochs, near-perfect metrics too quickly
- Fix: Remove target-derived features, fix train/val split to prevent future data leaking into past, audit preprocessing
- fix_type: "Data Leakage Prevention"

### CLASS IMBALANCE (Medium)
- Signs: High overall accuracy but poor minority class recall, skewed class distribution in logs
- Fix: Set class_weight='balanced' AND apply SMOTE oversampling, adjust decision threshold to 0.3-0.4
- fix_type: "Class Reweighting and Resampling"

### FP16 UNDERFLOW (Medium)
- Signs: Loss suddenly drops to 0 or NaN mid-training, gradients vanish, model uses mixed precision / fp16
- Fix: Enable dynamic loss scaling (GradScaler), switch BatchNorm layers to float32, set min loss scale=1
- fix_type: "FP16 Loss Scaling Fix"

### SILENT DATA POISONING (Hard)
- Signs: Training loss low but val loss erratic, random errors on easy samples, label distribution mismatch
- Fix: Run Cleanlab label noise detection, remove corrupted samples, retrain on cleaned dataset
- fix_type: "Label Noise Detection and Removal"

### CATASTROPHIC FORGETTING (Hard)
- Signs: New task accuracy rises while old task accuracy collapses, sequential task training
- Fix: Apply EWC (Elastic Weight Consolidation), add rehearsal replay buffer with 5-10% old task samples
- fix_type: "Elastic Weight Consolidation + Replay"

## STRICT OUTPUT FORMAT:
For tool calls, respond ONLY with:
{"action": "fetch_config"}
{"action": "fetch_logs"}
{"action": "fetch_loss_curve"}

For diagnosis, respond ONLY with:
{
  "action": "diagnose",
  "diagnosis": "<root cause with specific evidence>",
  "fix_type": "<exact fix_type from guide>",
  "fix_detail": "<concrete steps with specific parameters>",
  "confidence": <0.85 to 0.99>
}

## RULES FOR GEMINI:
- Output raw JSON ONLY. No markdown. No backticks. No explanation text.
- Never write ``` or ```json around your response.
- Never add any text before or after the JSON object.
- One JSON object per response, nothing else.
"""


# ─── Agent ─────────────────────────────────────────────────────────────────────

def get_agent_action(task_description: str, history: list, obs: dict) -> dict:
    """Ask the LLM for the next action, with extreme speed optimization."""
    
    elapsed = time.time() - EPISODE_START_TIME
    if elapsed > TIMEOUT_LIMIT:
        print(f"\n  🕒 Watchdog: Forcing emergency diagnosis.")
        return {
            "action_type": "diagnose",
            "diagnosis":   "Emergency exit.",
            "fix_type":    "config_change",
            "fix_detail":  "lower learning rate",
            "confidence":  0.3,
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # EXTREME optimization: Keep ONLY the last turn
    for h in history[-1:]:
        messages.append({"role": "user",      "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})

    tool_result = obs.get("tool_result")
    tool_result_str = json.dumps(tool_result)

    step_num = obs.get('step_number', 0) + 1
    
    # FORCED DECISION: On step 5, we MUST diagnose.
    if step_num >= 5:
         print(f"  [FORCE] Last Step Trigger: Forcing diagnosis.")
         user_msg = f"Task: {task_description}\nStep: {step_num}/5\nResult: {tool_result_str}\nFINAL STEP: YOU MUST DIAGNOSE NOW."
    else:
         user_msg = f"Task: {task_description}\nStep: {step_num}/5\nResult: {tool_result_str}\nNext? (JSON)"
    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=256, # Minimum necessary
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        
        # 🛡️ Iron-Clad Cleaning: Extract first {...} block
        if "{" in raw and "}" in raw:
            raw = raw[raw.find("{"):raw.rfind("}")+1]
        
        action_data = json.loads(raw)
        
        # 🛡️ Action Guard: Ensure action_type exists
        if not action_data or not isinstance(action_data, dict):
            return {"action_type": "fetch_logs"}
            
        if "action_type" not in action_data:
            # 🔄 Universal Translator: Support 'action' key from new prompt
            if "action" in action_data: 
                action_data["action_type"] = action_data["action"]
            # If they gave a diagnosis without action_type, assume they meant diagnose
            elif "diagnosis" in action_data: 
                action_data["action_type"] = "diagnose"
            else: 
                action_data["action_type"] = "fetch_logs"
            
        return action_data

    except Exception as e:
        print(f"\n  ⚠️ Iron-Clad Fallback ({e}). Defaulting to fetch_logs.")
        return {"action_type": "fetch_logs"}


# ─── Episode Loop ──────────────────────────────────────────────────────────────

def run_episode() -> dict:
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME} | Watchdog: 25m")
    print(f"{'='*60}\n")

    result = _post("/reset")
    obs = result["observation"]
    
    all_scores:    dict = {}
    episode_done:  bool = False

    while not episode_done:
        task_desc = obs["description"]
        task_id   = obs["task_id"]
        history   = []

        print(f"\n=== {task_id} ({obs['difficulty']}) ===")

        while True:
            action = get_agent_action(task_desc, history, obs)
            print(f"  [{time.time()-EPISODE_START_TIME:.0f}s] action={action.get('action_type')}", end="", flush=True)

            result  = _post("/step", action)
            reward  = result["reward"]
            obs     = result["observation"]
            done    = result["done"] # Task level done
            info    = result["info"]

            print(f" | score={reward['total']:.2f}")

            history.append({
                "user":      f"last_result={json.dumps(obs.get('tool_result', {}))}",
                "assistant": json.dumps(action),
            })

            if done:
                if action.get("action_type") == "diagnose":
                    all_scores[obs["difficulty"]] = reward["total"]
                if info.get("episode_done"):
                    episode_done = True
                break

    return all_scores


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    start  = time.time()
    scores = run_episode()
    elapsed = time.time() - start

    print(f"\nFinal Scores:")
    for diff, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {diff:8s}: [{bar}] {score:.3f}")
    avg = sum(scores.values()) / max(len(scores), 1)
    print(f"  {'average':8s}: {avg:.3f}")
    print(f"  runtime:  {elapsed:.1f}s")

    assert elapsed < 1800, f"Inference took {elapsed:.0f}s — exceeds 30-min limit"
    print("\n✅ All checks passed. Ready to submit.")


if __name__ == "__main__":
    main()