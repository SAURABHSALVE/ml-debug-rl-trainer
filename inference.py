import os
import json
import asyncio
import textwrap
from typing import List, Optional
from openai import OpenAI

# Your environment imports (adjust based on your actual file structure)
from env.environment import MLDebugEnv
from env.models import Action

# Mandatory Hackathon Variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"

TASK_NAME = os.getenv("MY_ENV_TASK", "ml-debugging")
BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "openenv-ml-debug")
MAX_STEPS = 5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert AI Engineer debugging a broken ML training pipeline.
    You must investigate the issue using fetch_* actions, then submit a diagnosis.
    
    Investigation Actions (Gather evidence):
      {"action_type": "fetch_logs", "epochs": "all"}
      {"action_type": "fetch_config", "keys": ["lr", "optimizer"]}
      {"action_type": "fetch_loss_curve", "split": "val"}
      {"action_type": "fetch_diagnostics", "check": "overfitting"}
      {"action_type": "fetch_class_data", "class_id": 0}

    Terminal Action (Ends the episode):
      {
        "action_type": "diagnose",
        "diagnosis": "Root cause explanation",
        "fix_type": "config_change",
        "fix_detail": "Specific fix values",
        "confidence": 0.9
      }
      
    Respond ONLY with a valid JSON object matching one of the actions above. No markdown.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Collapse JSON into a single line for the logger
    action_single_line = action.replace('\n', '').replace('\r', '').replace('  ', '')
    print(f"[STEP] step={step} action={action_single_line} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs_description: str, action_history: List[str]) -> str:
    history_str = "\n".join(action_history) if action_history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation: {obs_description}
        Previous Actions:
        {history_str}
        
        Provide your next action as a strict JSON object.
        """
    ).strip()

def get_llm_action(client: OpenAI, step: int, obs_description: str, history: List[str]) -> Action:
    user_prompt = build_user_prompt(step, obs_description, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        # Clean up markdown if the LLM ignores instructions
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        parsed_json = json.loads(raw_text)
        return Action(**parsed_json), raw_text
        
    except Exception as exc:
        # Fallback action to prevent script crash if LLM hallucinates format
        fallback = Action(action_type="fetch_logs", epochs="all")
        return fallback, f'{{"action_type": "fetch_logs", "error": "{str(exc)}"}}'


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MLDebugEnv(max_steps_per_task=MAX_STEPS)

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # We will test all 3 difficulties (Easy, Medium, Hard)
    total_score = 0.0
    
    for task_idx in range(3):
        obs = env.reset() # OpenEnv standard reset
        
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        task_score = 0.0
        success = False
        error_msg = None

        for step in range(1, MAX_STEPS + 1):
            action_obj, raw_action_str = get_llm_action(client, step, obs.description, history)
            
            try:
                # OpenEnv standard step
                obs, reward_obj, done, info = env.step(action_obj)
                reward_val = reward_obj.score if hasattr(reward_obj, 'score') else float(reward_obj)
            except Exception as e:
                reward_val = 0.0
                done = True
                error_msg = str(e)
            
            rewards.append(reward_val)
            steps_taken = step
            history.append(f"Step {step}: {raw_action_str}")
            
            log_step(step=step, action=raw_action_str, reward=reward_val, done=done, error=error_msg)
            
            if done:
                task_score = reward_val 
                break

        # Assuming a score > 0.5 is a "success" based on your grading criteria
        success = task_score >= 0.5
        total_score += task_score
        
        log_end(success=success, steps=steps_taken, score=task_score, rewards=rewards)

if __name__ == "__main__":
    main()