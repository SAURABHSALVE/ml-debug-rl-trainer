---
title: ML Experiment Debugger — OpenEnv
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
short_description: RL env for AI agents to debug ML training runs
tags:
  - reinforcement-learning
  - ml-engineering
  - openenv
  - tool-use
  - debugging
  - huggingface
  - pytorch
  - hackathon
license: apache-2.0
app_port: 7860
---

# ML Experiment Debugger — OpenEnv Environment

> *Every ML engineer has stared at a broken training run wondering what went wrong.
> We built an RL environment that teaches AI agents to debug like a senior engineer.*

Built for the **Meta × PyTorch × HuggingFace OpenEnv Hackathon** | Team: **AIverse**

🚀 **Live Demo:** [https://12-vinit-ml-experiment-debugger.hf.space](https://12-vinit-ml-experiment-debugger.hf.space)  
📖 **API Docs (Swagger):** [https://12-vinit-ml-experiment-debugger.hf.space/docs](https://12-vinit-ml-experiment-debugger.hf.space/docs)  
🤗 **HuggingFace Space:** [https://huggingface.co/spaces/12-vinit/ml-experiment-debugger](https://huggingface.co/spaces/12-vinit/ml-experiment-debugger)

---

## 🎯 Why This Exists

ML training failures cost real money and real lives:
- A model silently overfits and the product team only finds out after launch
- A hospital's X-ray classifier reaches 93% accuracy — but never catches disease in minority classes
- A fine-tuned model destroys its pretrained capabilities overnight (catastrophic forgetting)

These bugs are subtle, expensive to catch, and everywhere. This environment **formalizes the ML debugging process** into a structured, gradable RL task:

- Agent starts with **only a natural-language task description** (no data)
- Must **choose which tools to call** from a limited step budget — each call costs one step
- Must navigate **misleading signals** (93% accuracy masking class imbalance)
- Must identify the **root cause** and prescribe a **specific, actionable fix**
- Is graded on **diagnosis correctness** + **fix quality** + **investigation efficiency**

No existing OpenEnv environment covers ML training diagnostics. This fills a direct gap for the HuggingFace / Meta engineering community — the same engineers who ship model training infrastructure every day.

---

## 🧩 Environment Overview

| Property            | Value |
|---|---|
| **Action Space**    | Structured (5 investigation tool calls + 1 terminal diagnose) |
| **Observation**     | Structured (task description, tool result, step counter, action history) |
| **Task Pool**       | **6 unique scenarios across 3 difficulty levels** |
| **Tasks/Episode**   | 3 (1 easy + 1 medium + 1 hard, randomly sampled from pool) |
| **Reward**          | Float in [0.0, 1.0] per task — partial credit |
| **Max Steps**       | 5 (easy), 6 (medium), 8 (hard) |
| **Grading**         | Keyword grader (always) + optional LLM grader (blended 60/40) |
| **Efficiency Bonus**| +0.05 if correct diagnosis within half the step budget |
| **Trajectory Bonus**| +0.05 on hard task if easy > 0.7 AND medium > 0.6 |
| **Deployment**      | HuggingFace Spaces (port 7860), Docker-ready |

---

## 📋 Task Catalogue (6 Scenarios)

### Task 1 — Overfitting Detection `[easy]`
**Scenario:** ResNet-50 trained for 20 epochs on CIFAR-10 subset. Training accuracy reaches 99% while val accuracy peaks at 81% then drops to 61%. Config shows `dropout=0`, `weight_decay=0`, `data_augmentation=False`.

**Correct diagnosis:** overfitting | **Grader:** 0.5 pts for identifying overfitting · 0.5 pts for valid regularization fix

**Baseline scores:** GPT-4o: ~0.85 | Llama-3.3-70B: ~0.85 | Llama-3-8B: ~0.45

---

### Task 2 — NaN from Bad Initialization `[easy]`
**Scenario:** Custom BERT-small trained on news classification. Loss is NaN from epoch 1, GPU utilization near 0%, training stopped after 8 epochs. Config contains `init_std=10.0`.

**Correct diagnosis:** bad_initialization | **Grader:** 0.5 pts for naming bad init · 0.5 pts for fix (std → 0.02 / Xavier / Kaiming)

**Baseline scores:** GPT-4o: ~0.80 | Llama-3.3-70B: ~0.75 | Llama-3-8B: ~0.35

---

### Task 3 — Learning Rate Explosion `[medium]`
**Scenario:** Transformer model trained with SGD at LR=0.5. Loss stable for 5 epochs then explodes; gradient norms grow past 1000 by epoch 10, training produces NaN.

**Correct diagnosis:** learning_rate_explosion | **Grader:** 0.5 pts for naming LR/gradient explosion · 0.5 pts for proposing valid LR in [1e-5, 1e-2]

**Baseline scores:** GPT-4o: ~0.75 | Llama-3.3-70B: ~0.80 | Llama-3-8B: ~0.35

---

### Task 4 — Class Imbalance `[medium]`
**Scenario:** MobileNetV2 on a 4-class medical X-ray dataset. Overall accuracy=93% looks excellent. Class 0 has 9,500 samples; classes 1-3 have 135–198 samples. Model never predicts minority classes (recall ~2%).

**Correct diagnosis:** class_imbalance | **Grader:** 0.5 pts for identifying class imbalance · 0.5 pts for proposing weighted loss / oversampling / SMOTE

**Baseline scores:** GPT-4o: ~0.80 | Llama-3.3-70B: ~0.70 | Llama-3-8B: ~0.30

---

### Task 5 — Silent Data Poisoning `[hard]`
**Scenario:** EfficientNet-B0 on a 5-class manufacturing defect dataset. Overall accuracy=84% looks fine. But 15-25% of one class has corrupted labels — that class stagnates at 28-42% accuracy while others reach 87-95%. A generic anomaly warning appears in logs at epoch 14.

**Correct diagnosis:** silent_data_poisoning | **Grader:** 0.3 pts for data/label corruption · 0.2 pts for naming the correct class · 0.5 pts for data_fix proposal

**Baseline scores:** GPT-4o: ~0.55 | Llama-3.3-70B: ~0.55 | Llama-3-8B: ~0.10

---

### Task 6 — Catastrophic Forgetting `[hard]`
**Scenario:** ResNet-50 pretrained on ImageNet is fine-tuned on satellite imagery binary classification. New task accuracy reaches 91%. But original ImageNet accuracy drops from 92% to under 15%. Config shows `freeze_backbone=False`, `lr=0.01`, `ewc_lambda=None`.

**Correct diagnosis:** catastrophic_forgetting | **Grader:** 0.5 pts for identifying catastrophic forgetting · 0.5 pts for proposing EWC / backbone freeze / replay / lower LR

**Baseline scores:** GPT-4o: ~0.65 | Llama-3.3-70B: ~0.55 | Llama-3-8B: ~0.15

---

## 🛠️ Action Space

**Investigation tools** (non-terminal — reveal data, cost 1 step each):
```json
{"action_type": "fetch_logs",         "start_epoch": 1, "end_epoch": 10}
{"action_type": "fetch_config",       "keys": ["lr", "optimizer", "dropout"]}
{"action_type": "fetch_loss_curve",   "split": "val"}
{"action_type": "fetch_gpu_metrics"}
{"action_type": "fetch_class_metrics","class_id": 2}
```

**Terminal action** (ends current task — triggers grading):
```json
{
  "action_type": "diagnose",
  "diagnosis":   "The model is overfitting — val loss diverges after epoch 10",
  "fix_type":    "config_change",
  "fix_detail":  "Add dropout=0.3 and weight_decay=1e-4; enable early stopping",
  "confidence":  0.9
}
```
`fix_type` must be one of: `config_change | data_fix | architecture_change`

---

## 👁️ Observation Space

```json
{
  "task_id":         "class_imbalance_55",
  "difficulty":      "medium",
  "description":     "A MobileNetV2 was trained on a 4-class medical X-ray dataset...",
  "step_number":     2,
  "max_steps":       6,
  "steps_remaining": 4,
  "tool_result":     {
    "class_metrics": {
      "0": {"accuracy": 0.982, "f1": 0.981, "support": 9500, "recall": 0.983},
      "1": {"accuracy": 0.024, "f1": 0.018, "support": 167,  "recall": 0.012}
    }
  },
  "action_history":  ["fetch_loss_curve", "fetch_class_metrics"],
  "available_tools": ["fetch_logs", "fetch_config", "fetch_loss_curve",
                      "fetch_gpu_metrics", "fetch_class_metrics", "diagnose"]
}
```

---

## 🏆 Reward Function

```python
# Per-task grading
score = diagnosis_score + fix_score        # sums to 1.0 max

# Efficiency bonus — rewards short investigation sequences
if score >= 0.8 and steps_used <= max_steps // 2 + 1:
    score = min(1.0, score + 0.05)

# Trajectory bonus — hard task gets extra credit if agent was consistent
if difficulty == "hard" and easy_score > 0.7 and medium_score > 0.6:
    score = min(1.0, score + 0.05)

# LLM blending (when USE_LLM_GRADING=true)
final_score = 0.6 * llm_score + 0.4 * keyword_score
```

---

## 🤖 Grading

| Mode | How | When Active |
|---|---|---|
| **Keyword** (default) | Deterministic string-matching with co-occurrence checks | Always on |
| **LLM** (optional) | Llama-3.3-70B-Instruct scores 0.0–1.0 as judge | `USE_LLM_GRADING=true` + `HF_TOKEN` set |

LLM grading uses robust multi-strategy JSON extraction with fallback to plain-text score parsing.

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Start new episode — returns first Observation |
| POST | `/step` | Submit tool call or diagnose action |
| GET | `/state` | Current episode state, scores, tools called |
| GET | `/tasks` | List the 3 tasks in the current episode |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI with full schema |

---

## 🚀 Setup & Running

### Local Development
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Run Tests (full suite)
```bash
pytest tests/ -v
```

### Docker
```bash
docker build -t ml-debug-env .
docker run -p 7860:7860 ml-debug-env
```

### Baseline Inference — Keyword Grading
```bash
export HF_TOKEN=your_hf_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Baseline Inference — LLM Grading Enabled
```bash
export HF_TOKEN=your_hf_token
export USE_LLM_GRADING=true
export GRADER_MODEL=meta-llama/Llama-3.3-70B-Instruct
python inference.py
```

---

## 📁 Project Structure

```
├── app.py               # FastAPI server — /reset /step /state /tasks /health
├── inference.py         # Baseline LLM agent (OpenAI-client compatible)
├── openenv.yaml         # OpenEnv manifest (v2.0.0)
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py        # Pydantic models: Action, Observation, Reward
│   ├── tasks.py         # 6 task generators (seeded, reproducible)
│   ├── graders.py       # Keyword + LLM graders with multi-strategy JSON parsing
│   └── environment.py   # MLDebugEnv: reset / step / state logic
└── tests/
    └── test_env.py      # Full test suite
```

---

## 🔧 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Agent model | `meta-llama/Llama-3.3-70B-Instruct` |
| `GRADER_MODEL` | LLM grader model | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | required for LLM calls |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |
| `USE_LLM_GRADING` | Enable LLM grading (60/40 blend) | `false` |

---

## 🔮 Extending the Environment

### Add a New Task Type
1. Write a generator in `env/tasks.py` following the existing pattern
2. Add a grader in `env/graders.py` with the rubric
3. Register it in `BUG_TYPE_GRADERS` dict and `TASK_POOL` in `env/environment.py`
4. Update `openenv.yaml` and `README.md`

### Change Reward Structure
Edit `env/environment.py` — the efficiency and trajectory bonus thresholds are clearly commented.

---

*Built with ❤️ by Team AIverse for the Meta × PyTorch × HuggingFace OpenEnv Hackathon*