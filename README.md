# ML Experiment Debugger — OpenEnv Environment

An RL environment where AI agents diagnose broken ML training runs through **multi-step investigation**.

Built for the **Meta × PyTorch × HuggingFace OpenEnv Hackathon** | Team: AIverse

---

## Why This Environment Exists

Every ML engineer has stared at a loss curve wondering what went wrong.
This environment formalizes that diagnostic process into a structured,
gradable RL task where agents must **choose what to investigate** — not just read the answer.

Unlike environments that give agents all data upfront, this environment forces the agent to:
- Start with **only a task description**
- **Choose which tools to call** (each costs a step from a limited budget)
- Navigate **misleading signals** (GPU metrics look fine when the real bug is in labels)
- **Diagnose the root cause** and prescribe a specific fix

No existing OpenEnv environment covers ML training diagnostics. This fills a direct gap
for the HuggingFace/Meta engineering community — the same engineers who ship model training
infrastructure every day.

---

## Environment Overview

| Property | Value |
|---|---|
| Action Space | Structured (investigation tool calls + terminal diagnose) |
| Observation Space | Structured (task description, tool results, step counter) |
| Tasks | 3 (easy → medium → hard) |
| Reward Type | Partial credit, float in [0.0, 1.0] |
| Max Steps | 5 (easy/medium), 7 (hard) |
| Efficiency Bonus | +0.05 if correct diagnosis in ≤ half budget |
| Trajectory Bonus | +0.05 on hard if easy>0.7 AND medium>0.6 |
| Deployment | HuggingFace Spaces (port 7860) |

---

## Action Space

**Investigation tools** (non-terminal — reveal data, cost 1 step each):
```json
{"action_type": "fetch_loss_curve", "split": "val"}
{"action_type": "fetch_config", "keys": ["lr", "optimizer", "dropout"]}
{"action_type": "fetch_logs", "start_epoch": 1, "end_epoch": 10}
{"action_type": "fetch_gpu_metrics"}
{"action_type": "fetch_class_metrics", "class_id": 2}
```

**Terminal action** (ends current task, graded 0.0–1.0):
```json
{
  "action_type": "diagnose",
  "diagnosis": "The model is overfitting — val loss diverges after epoch 10",
  "fix_type": "config_change",
  "fix_detail": "Add dropout=0.3 and weight_decay=1e-4",
  "confidence": 0.9
}
```
`fix_type` must be one of: `config_change | data_fix | architecture_change`

---

## Observation Space

```json
{
  "task_id": "overfitting_1234",
  "difficulty": "easy",
  "description": "A ResNet-50 was trained...",
  "step_number": 2,
  "max_steps": 5,
  "steps_remaining": 3,
  "tool_result": {"val_loss": [2.0, 1.8, 1.6, 1.7, 1.9, ...]},
  "action_history": ["fetch_loss_curve"],
  "available_tools": ["fetch_logs", "fetch_config", "fetch_loss_curve", "fetch_gpu_metrics", "fetch_class_metrics", "diagnose"]
}
```

---

## Tasks

### Task 1 — Overfitting Detection (Easy)

**Scenario:** ResNet-50 trained for 20 epochs on a small image dataset with zero regularization.
Train accuracy reaches 99% while val accuracy peaks at 81% then drops to 61%.

**Agent must:** Identify overfitting and propose a valid regularization fix.

**Grader:**
- 0.5 pts — Correctly identifies overfitting
- 0.5 pts — Proposes valid fix (dropout, weight_decay, augmentation, early_stopping)

**Baseline scores:**
- GPT-4o: ~0.85 | Llama-3.3-70B: ~0.65 | Llama-3-8B: ~0.40

---

### Task 2 — Learning Rate Explosion (Medium)

**Scenario:** Transformer trained with SGD at LR=0.5. Loss stable for 5 epochs then explodes,
gradient norms visible in logs growing >100 by epoch 8, finally producing NaN.

**Agent must:** Identify LR as root cause AND propose a numeric value in [1e-5, 1e-2].

**Grader:**
- 0.5 pts — Names LR as too high / mentions gradient explosion
- 0.5 pts — Proposes valid LR range (partial credit for direction only)

**Baseline scores:**
- GPT-4o: ~0.75 | Llama-3.3-70B: ~0.55 | Llama-3-8B: ~0.30

---

### Task 3 — Silent Data Poisoning (Hard)

**Scenario:** EfficientNet-B0 on 5-class manufacturing defect dataset. Overall accuracy=84%
looks fine. But 15-25% of one class has corrupted labels. Per-class metrics show that
class stagnating at ~35% while others reach 90%+. A subtle warning appears in logs at epoch 13.

**Agent must:** Identify label corruption AND name the specific affected class.

**Grader:**
- 0.3 pts — Identifies label/data corruption as bug type
- 0.2 pts — Names the correct poisoned class
- 0.5 pts — Proposes correct data_fix (re-annotate, clean labels)

**Baseline scores:**
- GPT-4o: ~0.55 | Llama-3.3-70B: ~0.35 | Llama-3-8B: ~0.10

---

## Reward Function

```
# Per task
score = diagnosis_score + fix_score          # both components sum to 1.0

# Efficiency bonus (if correct and used few steps)
if score >= 0.8 and steps_used <= max_steps // 2:
    score = min(1.0, score + 0.05)

# Trajectory bonus (hard task only)
if difficulty == "hard" and easy_score > 0.7 and medium_score > 0.6:
    score = min(1.0, score + 0.05)
```

Partial credit is always given — agents are rewarded for:
- Correct symptom identification even without full diagnosis
- Correct fix direction even without exact values
- Efficient investigation (fewer tool calls = higher efficiency bonus)

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode, returns Task 1 observation |
| POST | `/step` | Submit investigation tool call or diagnose action |
| GET | `/state` | Current episode state, scores, tools called |
| GET | `/tasks` | List all 3 tasks with descriptions |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Setup

### Local development
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run tests
```bash
pytest tests/ -v
```

### Docker
```bash
docker build -t ml-debug-env .
docker run -p 7860:7860 ml-debug-env
```

### Baseline inference
```bash
export HF_TOKEN=your_hf_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

---

## Project Structure

```
├── app.py               # FastAPI server (reset/step/state/tasks/health)
├── inference.py         # Baseline agent (OpenAI client)
├── openenv.yaml         # OpenEnv manifest
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py        # Pydantic models: Action, Observation, Reward
│   ├── tasks.py         # Task generators (3 tasks, seeded randomness)
│   ├── graders.py       # Graders with partial credit
│   └── environment.py   # MLDebugEnv: reset/step/state logic
└── tests/
    └── test_env.py      # Full pytest suite
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | required |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |