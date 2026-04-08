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

from openai import OpenAI

# ─── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ─── HTTP Helpers ───────────────────────────────────────────────────────────────

def _post(path: str, body: dict = None) -> dict:
    url  = f"{ENV_BASE_URL}{path}"
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{ENV_BASE_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert ML debugging agent. Your goal is to diagnose broken ML training runs.

You have a LIMITED budget of steps per task. Each investigation tool costs 1 step.

INVESTIGATION TOOLS (each costs 1 step — pick wisely):
  {"action_type": "fetch_logs",        "start_epoch": 1, "end_epoch": 10}
  {"action_type": "fetch_config",      "keys": ["lr", "dropout", "optimizer"]}
  {"action_type": "fetch_loss_curve",  "split": "val"}
  {"action_type": "fetch_gpu_metrics"}
  {"action_type": "fetch_class_metrics", "class_id": 0}

TERMINAL ACTION (call when you are confident — ends the task):
  {
    "action_type": "diagnose",
    "diagnosis":   "Root cause explanation...",
    "fix_type":    "config_change | data_fix | architecture_change",
    "fix_detail":  "Exact fix with values...",
    "confidence":  0.9
  }

STRATEGY BY DIFFICULTY:
  EASY   → fast: fetch_loss_curve (spot val divergence or NaN), fetch_config (check init_std/dropout), then diagnose.
  MEDIUM → fetch_logs (check loss + grad_norm), fetch_config (check lr), then diagnose with exact LR value.
  HARD (data_poisoning)  → fetch_class_metrics for ALL 5 classes (0..4), find the one with low accuracy.
  HARD (forgetting)      → fetch_logs to find original_task_acc collapse, fetch_config for freeze_backbone + lr.

RULES:
  ✅ Be SPECIFIC: name exact class, exact config key and value.
  ✅ For data issues, use fix_type=data_fix.
  ✅ For hyperparameter issues, use fix_type=config_change.
  ✅ For architecture/continual-learning issues, use fix_type=architecture_change.
  ❌ Do NOT repeat a tool call — it costs a step and returns nothing new.
  ❌ Do NOT call fetch_gpu_metrics unless all other signals are exhausted.

Always respond with a SINGLE valid JSON object. No explanation outside the JSON."""


# ─── Agent ─────────────────────────────────────────────────────────────────────

def get_agent_action(task_description: str, history: list, obs: dict) -> dict:
    """Ask the LLM for the next action given the current observation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for h in history:
        messages.append({"role": "user",      "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})

    tool_result = obs.get("tool_result")
    tool_result_str = json.dumps(tool_result, indent=2) if tool_result else "None yet (first step)"

    user_msg = (
        f"TASK: {task_description}\n\n"
        f"Difficulty:       {obs.get('difficulty')}\n"
        f"Step:             {obs.get('step_number', 0) + 1} / {obs.get('max_steps', '?')}\n"
        f"Steps remaining:  {obs.get('steps_remaining')}\n"
        f"Tools called:     {obs.get('action_history', [])}\n\n"
        f"Last tool result:\n{tool_result_str}\n\n"
        f"What is your next action? Respond with a single JSON object."
    )
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()

    # ── JSON extraction ───────────────────────────────────────────────────────
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    for fence in ("```json", "```"):
        if fence in raw:
            raw = raw.split(fence)[1].split("```")[0].strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

    # Last resort: return a low-confidence diagnose so the episode continues
    return {
        "action_type": "diagnose",
        "diagnosis":   raw[:500],
        "fix_type":    "config_change",
        "fix_detail":  "Unable to parse structured response",
        "confidence":  0.1,
    }


# ─── Episode Loop ──────────────────────────────────────────────────────────────

def run_episode() -> dict:
    print(f"\n{'='*60}")
    print(f"Model:       {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")
    print(f"{'='*60}\n")

    obs = _post("/reset")
    print(f"Episode started. Task 1: {obs['task_id']} ({obs['difficulty']})")

    all_scores:    dict = {}
    episode_done:  bool = False

    while not episode_done:
        current_task_id   = obs["task_id"]
        current_diff      = obs["difficulty"]
        task_description  = obs["description"]
        task_history:     list = []

        print(f"\n─── Task: {current_task_id} ({current_diff}) ───")
        print(f"    {task_description[:120]}...")

        while True:
            steps_remaining = obs.get("steps_remaining", 0)
            print(f"  Step {obs['step_number'] + 1:2d} | remaining={steps_remaining} | ", end="")

            action = get_agent_action(task_description, task_history, obs)
            print(f"action={action.get('action_type')}", end="")

            if action.get("action_type") == "diagnose":
                print(f" | diagnosis={action.get('diagnosis', '')[:60]}...")
            else:
                print()

            result  = _post("/step", action)
            reward  = result["reward"]
            obs     = result["observation"]
            done    = result["done"]
            info    = result["info"]

            # `reward.total` is the scored value (investigation steps have low intermediate rewards)
            print(f"           reward.total={reward['total']:.3f} | {reward['feedback'][:80]}")

            task_history.append({
                "user":      f"steps_remaining={steps_remaining} | last_result={json.dumps(obs.get('tool_result', {}))}",
                "assistant": json.dumps(action),
            })

            if action.get("action_type") == "diagnose":
                all_scores[current_diff] = reward["total"]
                print(f"\n  ✅ Task complete. Score: {reward['total']:.3f}")

                if done:
                    episode_done = True
                    print(f"\n{'='*60}")
                    print("EPISODE COMPLETE")
                    print(f"Scores:  {info.get('scores', {})}")
                    print(f"Average: {info.get('average_score', 0):.3f}")
                    print(f"{'='*60}")
                else:
                    print(f"  Next task: {obs['task_id']} ({obs['difficulty']})")
                break

            time.sleep(0.4)  # courtesy rate-limit

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

    assert elapsed < 1200, f"Inference took {elapsed:.0f}s — exceeds 20-min limit"
    print("\n✅ All checks passed. Ready to submit.")


if __name__ == "__main__":
    main()