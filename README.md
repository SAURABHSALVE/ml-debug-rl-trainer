# ML Experiment Debugger — OpenEnv Environment

> *Every ML engineer has stared at a loss curve wondering what went wrong. We built an RL environment that teaches AI agents to debug like a senior engineer.*

Built for the **Meta × PyTorch × HuggingFace OpenEnv Hackathon** | Team: AIverse

---

## Why This Exists

ML training failures cost real money. A model silently overfits, a learning rate explodes overnight, a hospital's X-ray classifier reaches 93% accuracy — and then clinicians realize it never catches disease in minority classes. These bugs are subtle, expensive, and everywhere.

This environment formalizes the **ML debugging process** into a structured, gradable RL task:

- The agent starts with **only a task description**
- It must **choose which tools to call** from a budget — each call costs one step
- It must navigate **misleading signals** (93% accuracy masking a class imbalance disaster)
- It must diagnose the **root cause** and prescribe a specific, actionable fix
- It is graded on both correctness **and** investigation efficiency

No existing OpenEnv environment covers ML training diagnostics. This fills a direct gap for the HuggingFace/Meta engineering community — the same engineers who ship model training infrastructure every day.

---

## Environment Overview

| Property | Value |
|---|---|
| Action Space | Structured (investigation tool calls + terminal diagnose) |
| Observation Space | Structured (task description, tool results, step counter) |
| **Task Pool** | **5 distinct scenarios across 3 difficulty levels** |
| Tasks per Episode | 3 (1 easy + 1 medium + 1 hard, randomly sampled from pool) |
| Reward Type | Partial credit, float in [0.0, 1.0] |
| Max Steps | 5 (easy), 6 (medium), 8 (hard) |
| Grading | Keyword grader + optional LLM grader (blended 60/40) |
| Efficiency Bonus | +0.05 if correct diagnosis in ≤ half budget |
| Trajectory Bonus | +0.05 on hard if easy>0.7 AND medium>0.6 |
| Deployment | HuggingFace Spaces (port 7860) |

---

## Task Catalogue (5 Scenarios)

### Task 1 — Overfitting Detection `[easy]`
**Scenario:** ResNet-50 trained 20 epochs on a small dataset. Train accuracy reaches 99% while val accuracy peaks at 81% then drops to 61%. Zero regularization in config.

**Grader:** 0.5 pts for identifying overfitting · 0.5 pts for valid regularization fix (dropout/weight_decay/augmentation)

**Baseline scores:** GPT-4o: ~0.85 | Llama-3.3-70B: ~0.85 | Llama-3-8B: ~0.45

---

### Task 2 — Learning Rate Explosion `[medium]`
**Scenario:** Transformer trained with SGD at LR=0.5. Loss stable for 5 epochs then explodes, gradient norms grow >100 by epoch 8, produces NaN.

**Grader:** 0.5 pts for naming LR/gradient explosion · 0.5 pts for proposing valid LR in [1e-5, 1e-2]

**Baseline scores:** GPT-4o: ~0.75 | Llama-3.3-70B: ~0.80 | Llama-3-8B: ~0.35

---

### Task 3 — Class Imbalance `[medium]`
**Scenario:** MobileNetV2 on a 4-class medical X-ray dataset. Overall accuracy=93% looks excellent. But one class has 9500 samples while others have ~150-200. The model never predicts minority classes.

**Grader:** 0.5 pts for identifying class imbalance · 0.5 pts for proposing weighted loss / oversampling / SMOTE

**Baseline scores:** GPT-4o: ~0.80 | Llama-3.3-70B: ~0.70 | Llama-3-8B: ~0.30

---

### Task 4 — Silent Data Poisoning `[hard]`
**Scenario:** EfficientNet-B0 on a 5-class manufacturing defect dataset. Overall accuracy=84% looks fine. But 15-25% of one class has corrupted labels. That class stagnates at ~35% while others reach 90%+. A generic anomaly warning appears in logs at epoch 13 — but doesn't name the class.

**Grader:** 0.3 pts for identifying label/data corruption · 0.2 pts for naming the correct class · 0.5 pts for data_fix proposal

**Baseline scores:** GPT-4o: ~0.55 | Llama-3.3-70B: ~0.55 | Llama-3-8B: ~0.10

---

### Task 5 — Catastrophic Forgetting `[hard]`
**Scenario:** ResNet-50 pretrained on ImageNet is fine-tuned on satellite imagery. New task accuracy reaches 91%. But after deployment, the original ImageNet classification capability drops from 92% to under 15%. `freeze_backbone=False` and `lr=0.01` destroys pretrained representations.

**Grader:** 0.5 pts for identifying catastrophic forgetting · 0.5 pts for proposing EWC / backbone freeze / replay buffer

**Baseline scores:** GPT-4o: ~0.65 | Llama-3.3-70B: ~0.55 | Llama-3-8B: ~0.15

---

## Action Space

**Investigation tools** (non-terminal — reveal data, cost 1 step each):
```json
{"action_type": "fetch_loss_curve", "split": "val"}
{"action_type": "fetch_config", "keys": ["lr", "optimizer", "dropout"]}
{"action_type": "fetch_logs", "start_epoch": 1, "end_epoch": 15}
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
  "task_id": "class_imbalance_4507",
  "difficulty": "medium",
  "description": "A MobileNetV2 was trained on a 4-class medical X-ray dataset...",
  "step_number": 2,
  "max_steps": 6,
  "steps_remaining": 4,
  "tool_result": {"class_metrics": {"0": {"accuracy": 0.98, "support": 9500}, "1": {"accuracy": 0.02, "support": 167}}},
  "action_history": ["fetch_loss_curve", "fetch_class_metrics"],
  "available_tools": ["fetch_logs", "fetch_config", "fetch_loss_curve", "fetch_gpu_metrics", "fetch_class_metrics", "diagnose"]
}
```

---

## Reward Function

```
# Per task
score = diagnosis_score + fix_score          # sum to 1.0

# Efficiency bonus (few steps, strong answer)
if score >= 0.8 and steps_used <= max_steps // 2:
    score = min(1.0, score + 0.05)

# Trajectory bonus (hard task only)
if difficulty == "hard" and easy_score > 0.7 and medium_score > 0.6:
    score = min(1.0, score + 0.05)

# LLM blending (when USE_LLM_GRADING=true)
final_score = 0.6 * llm_score + 0.4 * keyword_score
```

---

## Grading

Two grading modes — both available, keyword is default for reproducibility:

| Mode | How | Activated By |
|---|---|---|
| **Keyword** (default) | Regex-free string matching with co-occurrence checks | Always on |
| **LLM** (optional) | Llama-3.3-70B scores 0.0–1.0, blended 60/40 with keyword | `USE_LLM_GRADING=true` + `HF_TOKEN` |

LLM grading example prompt:
```
BUG TYPE: class_imbalance
ROOT CAUSE: Severe class imbalance — 95% of data is class 0

AGENT DIAGNOSIS: The model only predicts the dominant class...
→ Score: 0.9 (identified imbalance + proposed weighted loss)
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode, randomly picks 3 from 5 task pool |
| POST | `/step` | Submit investigation tool call or diagnose action |
| GET | `/state` | Current episode state, scores, tools called |
| GET | `/tasks` | List the 3 tasks in the current episode |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Setup

### Local development
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run tests (35 tests)
```bash
pytest tests/ -v
```

### Docker
```bash
docker build -t ml-debug-env .
docker run -p 7860:7860 ml-debug-env
```

### Baseline inference (keyword grading)
```bash
export HF_TOKEN=your_hf_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Baseline inference (LLM grading enabled)
```bash
export HF_TOKEN=your_hf_token
export USE_LLM_GRADING=true
export GRADER_MODEL=meta-llama/Llama-3.3-70B-Instruct
python inference.py
```

---

## Project Structure

```
├── app.py               # FastAPI server (reset/step/state/tasks/health)
├── inference.py         # Baseline agent (OpenAI client, task-specific strategy)
├── openenv.yaml         # OpenEnv manifest
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py        # Pydantic models: Action, Observation, Reward
│   ├── tasks.py         # 5 task generators (seeded randomness)
│   ├── graders.py       # Keyword graders + LLM grader with fallback
│   └── environment.py   # MLDebugEnv: reset/step/state logic, 5-task pool
└── tests/
    └── test_env.py      # 35-test pytest suite
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Agent model identifier | `meta-llama/Llama-3.3-70B-Instruct` |
| `GRADER_MODEL` | LLM grader model | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | required |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |
| `USE_LLM_GRADING` | Enable LLM grading (60/40 blend) | `false` |