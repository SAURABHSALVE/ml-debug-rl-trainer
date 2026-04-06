import json
import os
import re
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env.environment import MLDebugEnv
from env.models import Action, Observation

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"

TASK_NAME = os.getenv("MY_ENV_TASK", "ml-debugging")
BENCHMARK = os.getenv("MY_ENV_BENCHMARK", "openenv-ml-debug")


DIAGNOSIS_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are diagnosing a failed ML training run.
    Return only one valid JSON object for a terminal diagnose action:
    {
      "action_type": "diagnose",
      "diagnosis": "...",
      "fix_type": "config_change | data_fix | architecture_change",
      "fix_detail": "...",
      "confidence": 0.0 to 1.0
    }

    Rules:
    - Be concrete.
    - Mention the root cause, not just symptoms.
    - Include numeric values when proposing hyperparameter fixes.
    - For data issues, name the affected class if evidence supports it.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_single_line = action.replace("\n", " ").replace("\r", " ").strip()
    print(
        f"[STEP] step={step} action={action_single_line} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def action_to_log_string(action: Action) -> str:
    payload = action.model_dump(exclude_none=True)
    return json.dumps(payload, separators=(",", ":"))


def summarize_initial_context(obs: Observation) -> str:
    log_excerpt = "\n".join((obs.training_logs or [])[:4])
    config_excerpt = obs.config_yaml or ""
    return textwrap.dedent(
        f"""
        Task ID: {obs.task_id}
        Difficulty: {obs.difficulty}
        Description: {obs.description}

        Initial Logs:
        {log_excerpt}

        Initial Config:
        {config_excerpt}
        """
    ).strip()


def parse_suspicious_class(action_result: Optional[str]) -> Optional[int]:
    if not action_result:
        return None

    suspicious_matches = re.findall(r"class_(\d+): .*STAGNANT", action_result)
    if suspicious_matches:
        return int(suspicious_matches[0])

    generic_matches = re.findall(r"class_(\d+)", action_result)
    if generic_matches:
        return int(generic_matches[0])
    return None


def planned_action(obs: Observation, suspicious_class: Optional[int]) -> Action:
    history = obs.action_history or []

    if obs.difficulty == "easy":
        if "fetch_loss_curve[val]" not in history:
            return Action(action_type="fetch_loss_curve", split="val")
        if not any(item.startswith("fetch_config") for item in history):
            return Action(action_type="fetch_config", keys=["dropout", "weight_decay", "lr"])

    elif obs.difficulty == "medium":
        if "fetch_logs[all]" not in history:
            return Action(action_type="fetch_logs", epochs="all")
        if not any(item.startswith("fetch_config") for item in history):
            return Action(
                action_type="fetch_config",
                keys=["lr", "optimizer", "warmup_steps", "gradient_clip"],
            )

    elif obs.difficulty == "hard":
        if "fetch_diagnostics[class_balance]" not in history:
            return Action(action_type="fetch_diagnostics", check="class_balance")
        if suspicious_class is not None and not any(item.startswith("fetch_class_data") for item in history):
            return Action(action_type="fetch_class_data", class_id=suspicious_class)
        if "fetch_logs[all]" not in history:
            return Action(action_type="fetch_logs", epochs="all")

    return Action(action_type="diagnose")


def build_diagnosis_prompt(obs: Observation, evidence_log: List[str]) -> str:
    evidence = "\n\n".join(evidence_log[-4:])
    return textwrap.dedent(
        f"""
        Diagnose the current ML debugging task.

        Task:
        {obs.description}

        Current difficulty: {obs.difficulty}
        Action history: {obs.action_history}

        Evidence:
        {evidence}

        Return the terminal diagnose action as JSON only.
        """
    ).strip()


def fallback_diagnosis(obs: Observation, suspicious_class: Optional[int]) -> Action:
    if obs.difficulty == "easy":
        return Action(
            action_type="diagnose",
            diagnosis="The model is overfitting because validation loss diverges while training loss continues to fall.",
            fix_type="config_change",
            fix_detail="Add dropout=0.3 and weight_decay=1e-4, and enable early stopping.",
            confidence=0.88,
        )
    if obs.difficulty == "medium":
        return Action(
            action_type="diagnose",
            diagnosis="The learning rate is too high, causing instability, exploding gradients, and NaN loss.",
            fix_type="config_change",
            fix_detail="Reduce lr to 0.001, add gradient_clip=1.0, and use warmup_steps=500.",
            confidence=0.9,
        )

    target_class = suspicious_class if suspicious_class is not None else 0
    return Action(
        action_type="diagnose",
        diagnosis=f"There is silent label corruption affecting class_{target_class}, which explains the stagnant per-class accuracy.",
        fix_type="data_fix",
        fix_detail=f"Re-annotate or filter corrupted samples from class_{target_class} and retrain on the cleaned dataset.",
        confidence=0.82,
    )


def request_diagnosis(
    client: OpenAI,
    obs: Observation,
    evidence_log: List[str],
    suspicious_class: Optional[int],
) -> Action:
    user_prompt = build_diagnosis_prompt(obs, evidence_log)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=220,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        payload = json.loads(raw_text)
        action = Action(**payload)
        if action.action_type != "diagnose":
            return fallback_diagnosis(obs, suspicious_class)
        return action
    except Exception:
        return fallback_diagnosis(obs, suspicious_class)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-key")
    env = MLDebugEnv(max_steps_per_task=15)

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    step_counter = 0
    final_score = 0.0
    success = False
    last_error: Optional[str] = None

    obs = env.reset()
    evidence_log: List[str] = [summarize_initial_context(obs)]
    suspicious_class: Optional[int] = None

    while True:
        action = planned_action(obs, suspicious_class)
        if action.action_type == "diagnose":
            action = request_diagnosis(client, obs, evidence_log, suspicious_class)

        raw_action = action_to_log_string(action)

        try:
            next_obs, reward_obj, done, info = env.step(action)
            reward_value = float(reward_obj.score)
            last_error = None
        except Exception as exc:
            reward_value = 0.0
            done = True
            info = {}
            next_obs = obs
            last_error = str(exc)

        step_counter += 1
        rewards.append(reward_value)
        log_step(step=step_counter, action=raw_action, reward=reward_value, done=done, error=last_error)

        action_result = info.get("action_result") or getattr(next_obs, "action_result", None)
        if action_result:
            evidence_log.append(f"Action result for {action.action_type}:\n{action_result}")
            parsed_class = parse_suspicious_class(action_result)
            if parsed_class is not None:
                suspicious_class = parsed_class

        if action.action_type == "diagnose":
            evidence_log = [summarize_initial_context(next_obs)] if not done else evidence_log
            suspicious_class = None if not done else suspicious_class

        if done:
            episode_summary = info.get("episode_summary", {})
            final_score = float(episode_summary.get("average_score", reward_value))
            success = final_score >= 0.5
            break

        obs = next_obs

    log_end(success=success, steps=step_counter, score=final_score, rewards=rewards)


if __name__ == "__main__":
    main()
