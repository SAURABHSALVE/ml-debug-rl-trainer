# ML Experiment Debugger — OpenEnv Environment

An RL environment where AI agents diagnose broken machine learning training runs.
Agents receive training logs, loss curves, config YAMLs, and GPU metrics — then
must identify root causes and prescribe concrete fixes.

**Built for OpenEnv Hackathon | Team: AIverse**

---

## Why This Environment Exists

Every ML engineer has stared at a loss curve wondering what went wrong.
This environment formalizes that diagnostic process into a structured,
gradable task that frontier models can be trained and evaluated on.

No existing OpenEnv environment covers ML training diagnostics.
This fills a direct gap for the HuggingFace/Meta engineering community —
the same engineers who ship model training infrastructure every day.

---

## Environment Overview

| Property | Value |
|---|---|
| Action Space | Structured (diagnosis, fix_type, fix_detail, confidence) |
| Observation Space | Structured (logs, loss curves, config YAML, GPU metrics) |
| Tasks | 3 (Easy → Medium → Hard) |
| Reward Type | Partial credit, float in [0.0, 1.0] |
| Max Steps | 1 per task (3 total per episode) |
| Trajectory Bonus | +0.05 on hard if easy > 0.7 AND medium > 0.6 |
| Deployment | Hugging Face Spaces (port 7860) |

---

## Action Space

```json
{
  "diagnosis": "string — root cause explanation",
  "fix_type":  "config_change | data_fix | architecture_change",
  "fix_detail": "string — specific actionable fix (include values where possible)",
  "confidence": "float in [0.0, 1.0]"
}
```

## Observation Space

```json
{
  "task_id":       "string",
  "difficulty":    "easy | medium | hard",
  "description":   "string — task prompt for the agent",
  "training_logs": ["list of log lines"],
  "config_yaml":   "string — model/training config",
  "loss_curve":    {"train": [...], "val": [...]},
  "gpu_metrics":   {"memory_mb": [...], "util_pct": [...]},
  "step_number":   "int",
  "max_steps":     5
}
```

---

## Tasks

### Task 1 — Overfitting Detection (Easy)

**What happens:** A ResNet-50 is trained for 20 epochs on a small image classification
dataset with zero regularization (dropout=0, weight_decay=0). Train loss drops cleanly;
val loss diverges after epoch 10.

**Agent must:** Identify overfitting and propose a valid regularization fix.

**Grader signals:**
- 0.5 pts — Correctly names "overfitting"
- 0.5 pts — Proposes valid fix: dropout, weight_decay, data augmentation, or early stopping

**Expected baseline scores:**
- GPT-4o: ~0.85
- Llama-3.3-70B: ~0.65
- Llama-3-8B: ~0.40

---

### Task 2 — Learning Rate Explosion (Medium)

**What happens:** A Transformer is trained with SGD at LR ∈ {0.1, 0.5, 1.0}.
Loss is stable for 5 epochs, then explodes, then produces NaN.
Gradient norms are visible in logs (growing > 10 by epoch 8).

**Agent must:** Identify LR as the root cause AND propose a numeric value in [1e-5, 1e-2].

**Grader signals:**
- 0.5 pts — Names LR as too high / mentions instability
- 0.5 pts — Proposes value in valid range (partial credit for direction: 0.35 pts if too small)

**Expected baseline scores:**
- GPT-4o: ~0.75
- Llama-3.3-70B: ~0.55
- Llama-3-8B: ~0.30

---

### Task 3 — Silent Data Poisoning (Hard)

**What happens:** An EfficientNet-B0 trains on a 5-class manufacturing defect dataset
where 15–25% of one class has corrupted labels. Global loss metrics look fine.
Per-class accuracy for the poisoned class stagnates at ~0.35 while others reach 0.90+.
A subtle warning about `label_consistency_check` appears in logs at epoch 13.

**Agent must:** Identify label corruption AND name the specific affected class.

**Grader signals:**
- 0.3 pts — Identifies label/annotation corruption as bug type
- 0.2 pts — Names the correct poisoned class
- 0.5 pts — Proposes data_fix with re-annotation or filtering

**Trajectory Bonus:** If the agent scored >0.7 on Task 1 and >0.6 on Task 2,
a +0.05 consistency bonus is added to this task's score (capped at 1.0).

**Expected baseline scores:**
- GPT-4o: ~0.55
- Llama-3.3-70B: ~0.35
- Llama-3-8B: ~0.10

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Reset episode, returns first observation |
| POST | `/step` | Submit action, returns next obs + reward |
| GET | `/state` | Current episode state + scores |
| GET | `/tasks` | List all 3 tasks with metadata |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive Swagger UI |

---

## Reward Function

Each task rewards partial progress:

```
score = diagnosis_score + fix_score       # both in [0.0, 0.5]

# Trajectory bonus (hard task only):
if easy_score > 0.7 AND medium_score > 0.6:
    hard_score = min(1.0, hard_raw + 0.05)
```

Rewards are never sparse — partial credit is given for:
- Noticing the right symptom without naming the cause
- Correct fix type without correct specific values
- Correct direction but wrong magnitude

---

## Setup & Usage

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

### Run baseline inference

```bash
export HF_TOKEN=your_hf_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python inference.py
```

---

## Baseline Scores (Llama-3.3-70B-Instruct)

| Task | Score |
|------|-------|
| overfitting_42 (easy) | 0.650 |
| lr_explosion_99 (medium) | 0.550 |
| data_poisoning_7 (hard) | 0.350 |
| **Average** | **0.517** |

---

## Project Structure

```
meta/
├── app.py               # FastAPI server (reset/step/state/tasks endpoints)
├── inference.py         # Baseline inference script (OpenAI client)
├── openenv.yaml         # OpenEnv manifest
├── Dockerfile           # Container definition
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py      # Package exports
│   ├── models.py        # Pydantic models: Observation, Action, Reward
│   ├── tasks.py         # Task generators (3 tasks × seeded randomness)
│   ├── graders.py       # Programmatic graders for each difficulty
│   ├── environment.py   # MLDebugEnv: reset/step/state logic
│   └── reward.py        # Trajectory bonus computation
└── tests/
    └── test_env.py      # Full pytest suite (graders + env + API)
```

---

## Environment Variables for Inference

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | Hugging Face / API key | — (required) |
