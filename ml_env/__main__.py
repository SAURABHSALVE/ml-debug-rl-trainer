import argparse
import json
from typing import Any, Dict, List

from ml_env.environment import MLDebugEnv
from ml_env.models import Action


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, str) or isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if hasattr(obj, "model_dump"):
        return _safe_json(obj.model_dump())
    return str(obj)


def run_demo(max_tools: int = 3) -> Dict[str, Any]:
    env = MLDebugEnv()
    observation = env.reset()
    result: Dict[str, Any] = {
        "initial_observation": observation.model_dump(),
        "actions": [],
        "state": env.state(),
    }

    actions: List[Action] = [
        Action(action_type="fetch_config", keys=["lr", "optimizer", "dropout", "class_weights"]),
        Action(action_type="fetch_loss_curve", split="val"),
        Action(action_type="fetch_class_metrics", class_id=0),
    ]

    for action in actions[:max_tools]:
        obs, reward, done, info = env.step(action)
        result["actions"].append({
            "action": action.model_dump(),
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        })
        if done:
            break

    result["final_state"] = env.state()
    return _safe_json(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ml-debug",
        description="Run a sample ML Experiment Debugger episode from the command line.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a short demonstration episode and print JSON output.",
    )
    parser.add_argument(
        "--tools",
        type=int,
        default=3,
        help="Number of investigation tools to run during the demo.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print clean JSON output instead of a text summary.",
    )
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    demo = run_demo(max_tools=args.tools)
    if args.json_output:
        print(json.dumps(demo, indent=2))
        return

    print("ML Experiment Debugger CLI Demo")
    print("--------------------------------")
    print("Initial observation:")
    print(json.dumps(demo["initial_observation"], indent=2))
    print("\nTool actions:")
    for idx, step in enumerate(demo["actions"], start=1):
        print(f"\nAction {idx}: {step['action']['action_type']}")
        print(json.dumps(step["observation"], indent=2))
        print(json.dumps(step["reward"], indent=2))
    print("\nFinal state:")
    print(json.dumps(demo["final_state"], indent=2))


if __name__ == "__main__":
    main()
