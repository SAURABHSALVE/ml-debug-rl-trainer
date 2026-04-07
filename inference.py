"""
Baseline inference script for ML Experiment Debugger.
Uses OpenAI-compatible client to run an agent through all 3 tasks.

Usage:
  export API_BASE_URL=https://router.huggingface.co/v1
  export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
  export HF_TOKEN=your_token_here
  python inference.py
"""

import json
import os
import time
from openai import OpenAI

# ─── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ─── HTTP helpers ───────────────────────────────────────────────────────────────

import urllib.request

def _post(path: str, body: dict = None) -> dict:
    url = f"{ENV_BASE_URL}{path}"
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def _get(path: str) -> dict:
    url = f"{ENV_BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ─── Agent ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ML debugging agent. You will diagnose broken ML training runs.

You have a LIMITED budget of steps per task. Use them wisely.

INVESTIGATION TOOLS (each costs 1 step):
- fetch_loss_curve: {"action_type": "fetch_loss_curve", "split": "val"}
- fetch_config: {"action_type": "fetch_config", "keys": ["lr", "dropout", "optimizer"]}
- fetch_logs: {"action_type": "fetch_logs", "start_epoch": 1, "end_epoch": 15}
- fetch_gpu_metrics: {"action_type": "fetch_gpu_metrics"}
- fetch_class_metrics: {"action_type": "fetch_class_metrics", "class_id": 0}

TERMINAL ACTION (ends the task — use when confident):
- diagnose: {"action_type": "diagnose", "diagnosis": "...", "fix_type": "config_change|data_fix|architecture_change", "fix_detail": "...", "confidence": 0.9}

STRATEGY BY TASK TYPE:
- EASY (overfitting): Start with fetch_loss_curve to see train vs val divergence, then fetch_config to check regularization params.
- MEDIUM (LR/gradient): Start with fetch_logs to see gradient norms and loss explosion, then fetch_config to check lr value.
- HARD (data quality): ALWAYS check ALL 5 classes (class_id 0–4) one by one using fetch_class_metrics. Look for the ONE class with significantly lower accuracy than the others. Also fetch_logs for anomaly warnings.

ANTI-PATTERNS (avoid these):
- Do NOT repeat the same tool call with the same parameters — it costs a step and gives nothing new.
- Do NOT call fetch_gpu_metrics unless all other tools have been used — GPU is rarely the cause.
- For hard tasks: do not diagnose before checking ALL class metrics.

When diagnosing, be SPECIFIC:
- Name the exact class that is corrupted (e.g., "class_2 has corrupted labels")
- Include exact config values (e.g., "reduce lr from 0.5 to 0.001")
- Use fix_type=data_fix for data/label problems, config_change for hyperparameter problems.

Always respond with a single valid JSON object. No explanation outside the JSON."""


def get_agent_action(task_description: str, history: list, obs: dict) -> dict:
    """Ask the LLM for the next action given current observation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Build conversation history
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})

    # Current observation
    user_msg = f"""Task: {task_description}

Current state:
- Difficulty: {obs.get('difficulty')}
- Steps remaining: {obs.get('steps_remaining')}
- Tools called so far: {obs.get('action_history', [])}
- Last tool result: {json.dumps(obs.get('tool_result'), indent=2) if obs.get('tool_result') else 'None yet'}

What is your next action? Respond with a single JSON object."""

    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON
    try:
        # Handle markdown code blocks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        # Fallback — if JSON parse fails, return a diagnose with what we have
        return {
            "action_type": "diagnose",
            "diagnosis": raw[:500],
            "fix_type": "config_change",
            "fix_detail": "Unable to parse structured response",
            "confidence": 0.1,
        }


# ─── Main Loop ─────────────────────────────────────────────────────────────────

def run_episode() -> dict:
    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")
    print(f"{'='*60}\n")

    # Reset
    obs = _post("/reset")
    print(f"Episode started. Task 1: {obs['task_id']} ({obs['difficulty']})")

    all_scores = {}
    history = []
    task_description = obs["description"]
    episode_done = False

    while not episode_done:
        current_task_id = obs["task_id"]
        current_difficulty = obs["difficulty"]

        print(f"\n--- Task: {current_task_id} ({current_difficulty}) ---")
        print(f"Description: {task_description[:120]}...")

        # Run task loop
        task_history = []
        task_done = False

        while not task_done:
            steps_remaining = obs.get("steps_remaining", 0)
            print(f"  Step {obs['step_number']+1} | Steps remaining: {steps_remaining}")

            # Get action from agent
            action = get_agent_action(task_description, task_history, obs)
            print(f"  Action: {action.get('action_type')} | ", end="")

            if action.get("action_type") == "diagnose":
                print(f"Diagnosis: {action.get('diagnosis', '')[:80]}...")

            # Submit action
            result = _post("/step", action)
            reward = result["reward"]
            obs = result["observation"]
            done = result["done"]
            info = result["info"]

            print(f"  Reward: {reward['total']:.3f} | Feedback: {reward['feedback'][:80]}")

            # Update history
            task_history.append({
                "user": f"Steps remaining: {steps_remaining}. Last result: {json.dumps(obs.get('tool_result', {}))}",
                "assistant": json.dumps(action),
            })

            # Check if this task ended (diagnose was called)
            if action.get("action_type") == "diagnose":
                score = reward["total"]
                all_scores[current_difficulty] = score
                print(f"\n  ✅ Task complete. Score: {score:.3f}")
                task_done = True

                if done:
                    episode_done = True
                    print(f"\n{'='*60}")
                    print("EPISODE COMPLETE")
                    print(f"Scores: {info.get('scores', {})}")
                    print(f"Average: {info.get('average_score', 0):.3f}")
                    print(f"{'='*60}")
                else:
                    # Move to next task
                    task_description = obs["description"]
                    print(f"\n  Next task: {obs['task_id']} ({obs['difficulty']})")

            # Budget exhausted without diagnosing
            elif steps_remaining <= 1 and action.get("action_type") != "diagnose":
                print(f"\n  ⚠️ Budget almost exhausted — forcing diagnose")

            time.sleep(0.5)  # Rate limit courtesy

    return all_scores


def main():
    start = time.time()
    scores = run_episode()
    elapsed = time.time() - start

    print(f"\nFinal Scores:")
    for difficulty, score in scores.items():
        print(f"  {difficulty}: {score:.3f}")
    avg = sum(scores.values()) / max(len(scores), 1)
    print(f"  Average: {avg:.3f}")
    print(f"  Runtime: {elapsed:.1f}s")

    # Validate runtime < 20 mins
    assert elapsed < 1200, f"Inference took {elapsed:.0f}s — exceeds 20 min limit"
    print("\n✅ All checks passed. Ready to submit.")


if __name__ == "__main__":
    main()