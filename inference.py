"""
ML Debug Environment — Multi-step Inference Script
Agent iteratively calls fetch_* actions to gather evidence,
then calls diagnose to complete each task.
"""
import os
import json
from openai import OpenAI
from env.environment import MLDebugEnv
from env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM = """\
You are an expert ML engineer. A model is underperforming — investigate why and prescribe a fix.

## Available actions (respond with strict JSON, no markdown)

Investigation (call these first to gather evidence):
  {"action_type": "fetch_logs",        "epochs": "1-10"}
  {"action_type": "fetch_logs",        "epochs": "15-20"}
  {"action_type": "fetch_config",      "keys": ["lr", "optimizer"]}
  {"action_type": "fetch_loss_curve",  "split": "val"}
  {"action_type": "fetch_loss_curve",  "split": "train"}
  {"action_type": "fetch_diagnostics", "check": "overfitting"}
  {"action_type": "fetch_diagnostics", "check": "gradients"}
  {"action_type": "fetch_diagnostics", "check": "class_balance"}
  {"action_type": "fetch_class_data",  "class_id": 2}

Terminal (call this when confident — ends current task):
  {
    "action_type": "diagnose",
    "diagnosis":   "Root cause explanation",
    "fix_type":    "config_change | data_fix | architecture_change",
    "fix_detail":  "Specific actionable fix with concrete values",
    "confidence":  0.0-1.0
  }

Strategy: use 3–6 fetch_* calls to build evidence, then diagnose.
Do NOT repeat the same call twice (penalty applied).
"""


def _format_obs(obs, action_result: str | None = None) -> str:
    parts = [
        f"## Task: {obs.task_id}  (difficulty: {obs.difficulty})",
        obs.description,
        f"Steps remaining: {obs.steps_remaining}/{obs.max_steps}",
    ]
    if obs.action_history:
        parts.append(f"Actions so far: {', '.join(obs.action_history)}")
    if action_result:
        parts.append(f"\n## Last action result:\n{action_result}")
    if obs.hint:
        parts.append(f"\n💡 Hint: {obs.hint}")
    parts.append("\nRespond with a single JSON action.")
    return "\n".join(parts)


def _parse_action(raw: str) -> Action:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        return Action(**parsed)
    except Exception:
        return Action(
            action_type="diagnose",
            diagnosis="Unable to parse LLM response",
            fix_type="config_change",
            fix_detail="N/A",
            confidence=0.0,
        )


def run_inference():
    env = MLDebugEnv(max_steps_per_task=15)
    obs = env.reset()
    results = []
    done = False

    while not done:
        messages = [{"role": "system", "content": SYSTEM}]
        action_result = None
        task_done = False

        while not task_done:
            messages.append({"role": "user", "content": _format_obs(obs, action_result)})

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )
            assistant_text = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": assistant_text})

            action = _parse_action(assistant_text)
            obs, reward, done, info = env.step(action)

            if action.action_type == "diagnose" or info.get("timeout"):
                task_done = True
                results.append({
                    "task_id":        info["task_id"],
                    "difficulty":     info["difficulty"],
                    "fix_quality":    reward.fix_quality,
                    "efficiency_bonus": reward.efficiency_bonus,
                    "final_score":    reward.score,
                    "steps_used":     reward.steps_used,
                    "reasoning":      reward.reasoning,
                })
                print(
                    f"[{info['task_id']}] "
                    f"quality={reward.fix_quality:.3f} "
                    f"+bonus={reward.efficiency_bonus:.3f} "
                    f"→ {reward.score:.3f}  "
                    f"[{reward.steps_used} steps] | {reward.reasoning}"
                )
            else:
                action_result = info.get("action_result", "")

    print("\n=== FINAL SCORES ===")
    for r in results:
        print(f"  {r['task_id']} ({r['difficulty']}): {r['final_score']:.3f}  [{r['steps_used']} steps]")
    avg = sum(r["final_score"] for r in results) / len(results)
    print(f"  AVERAGE: {avg:.3f}")
    return results


if __name__ == "__main__":
    run_inference()
