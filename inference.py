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

## CRITICAL RULES:
1. ALWAYS use available tools (fetch_config, fetch_logs, fetch_loss_curve) BEFORE diagnosing.
2. NEVER default to "overfitting" or "regularization" without evidence — these are lazy defaults that will score poorly.
3. Each bug type has specific, unique signatures. Match them carefully.
4. You have EXACTLY 5 STEPS per task. On Step 5, you MUST choose the 'diagnose' tool and make your best guess.

## BUG IDENTIFICATION GUIDE:

### DATA LEAKAGE
- Signs: Suspiciously high train AND val accuracy early, near-perfect metrics
- Fix: Remove features derived from target, fix train/val split order, audit preprocessing pipeline
- fix_type: "Data Leakage Prevention"

### CLASS IMBALANCE  
- Signs: High accuracy but poor recall on minority class, skewed confusion matrix, unequal class counts in logs
- Fix: Use BOTH class_weight='balanced' AND oversampling (SMOTE) or undersampling. Adjust decision threshold.
- fix_type: "Class Reweighting and Resampling"

### SILENT DATA POISONING / LABEL CORRUPTION
- Signs: Validation loss is erratic or suspiciously high, training loss is low, random-looking errors on easy examples, mismatch between expected and actual label distributions
- Fix: Audit label quality, run label noise detection (e.g. Cleanlab), remove or relabel corrupted samples
- fix_type: "Label Noise Detection and Removal"

### FP16 UNDERFLOW
- Signs: Loss suddenly goes to 0 or NaN, gradients vanish, training with mixed precision
- Fix: Use loss scaling, switch critical layers to float32, enable dynamic loss scaling

### NaN INITIALIZATION
- Signs: Loss is NaN from step 0, gradients are NaN immediately
- Fix: Check weight initialization, clip gradients, verify input normalization

### CATASTROPHIC FORGETTING
- Signs: New task accuracy improves while old task accuracy collapses
- Fix: Elastic Weight Consolidation (EWC), rehearsal/replay buffer, progressive neural networks

## OUTPUT FORMAT (JSON):
{
  "diagnosis": "<specific root cause with evidence from logs/config/curves>",
  "fix_type": "<exact fix category from guide above>",
  "fix_detail": "<step-by-step concrete fix with specific parameters>",
  "confidence": <0.0-1.0>
}

## IMPORTANT NOTES:
- Respond with valid JSON only for diagnosis actions
- Use all available tool calls before concluding
- Be specific — vague answers score 0.25, precise answers score 0.8+
- Respond in JSON ONLY."""


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
    except Exception as e:
        print(f"\n  ⚠️ LLM Failed: {e}. Falling back...")
        return {
            "action_type": "diagnose",
            "diagnosis":   "LLM timeout.",
            "fix_type":    "config_change",
            "fix_detail":  "reduce learning rate",
            "confidence":  0.1,
        }

    try:
        return json.loads(raw)
    except:
        # Simple extraction for robustness
        if "{" in raw and "}" in raw:
            try:
                return json.loads(raw[raw.find("{"):raw.rfind("}")+1])
            except: pass
        
        return {
            "action_type": "diagnose",
            "diagnosis":   "JSON parsing failed under time pressure.",
            "fix_type":    "config_change",
            "fix_detail":  "fix initialization or data labels",
            "confidence":  0.1,
        }


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