---
title: ML Experiment Debugger — OpenEnv
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
short_description: "Multi-step RL env (OpenEnv spec) — AI agents diagnose 6 real ML bugs: data leakage, FP16 underflow, class imbalance, catastrophic forgetting & more"
tags:
  - reinforcement-learning
  - openenv
  - ml-engineering
  - tool-use
  - debugging
  - huggingface
  - pytorch
  - hackathon
  - fastapi
  - rl-environment
  - meta-hackathon
  - llm-grading
  - reward-shaping
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

ML training failures cost real money and time:
- A model silently overfits and the product team only finds out after launch.
- A hospital's X-ray classifier reaches 93% accuracy — but never catches disease in minority classes.
- A fine-tuned model destroys its pretrained capabilities overnight (catastrophic forgetting).

This environment **formalizes the ML debugging process** into a structured, gradable RL task:
- Agent starts with **only a natural-language task description**.
- Must **choose which tools to call** from a limited step budget.
- Must navigate **misleading signals** (e.g., high accuracy masking imbalance).
- Must identify the **root cause** and prescribe an **actionable fix**.
- Is graded on **diagnosis correctness** + **fix quality** + **investigation efficiency**.

---

## 🧩 Environment Overview

| Property            | Value |
|---|---|
| **Action Space**    | Structured (5 investigation tool calls + 1 terminal diagnose) |
| **Observation**     | Structured (task description, tool result, step counter, history) |
| **Task Pool**       | **6 unique scenarios across 3 difficulty levels** |
| **Tasks/Episode**   | 3 (1 easy + 1 medium + 1 hard, randomly sampled) |
| **Reward**          | Float in [0.0, 1.0] per task — partial credit available |
| **Max Steps**       | 5 (easy), 6 (medium), 8 (hard) |
| **Grading**         | Keyword grader (always) + optional LLM grader (blended 60/40) |
| **Efficiency Bonus**| +0.05 if correct diagnosis within half the step budget |
| **Trajectory Bonus**| +0.05 on hard task if easy > 0.7 AND medium > 0.6 |
| **Deployment**      | HuggingFace Spaces (port 7860), Docker-ready |

---

## 📋 Task Catalogue (6 Scenarios)

### Task 1 — Data Leakage `[easy]`
**Scenario:** XGBoost trained on tabular customer data. Accuracy is 100% on both train and val. Investigation reveals a feature `last_purchase_date` is identical to the target `churn_date` in many samples.

### Task 2 — NaN from Bad Initialization `[easy]`
**Scenario:** BERT-small custom model. Training loss is NaN from epoch 1. GPU utilization is near 0%. Config contains an extreme `init_std=10.0`.

### Task 3 — FP16 Underflow `[medium]`
**Scenario:** Llama-3-8B LoRA fine-tuning in fp16. Gradients are mostly 0, loss is stagnant. Investigation reveals no gradient scaler was used, causing underflow to zero.

### Task 4 — Class Imbalance `[medium]`
**Scenario:** MobileNetV2 on medical X-rays. Overall accuracy is 93%. However, one class has 9,500 samples while others have <200. The model never predicts minority classes (recall ~2%).

### Task 5 — Silent Data Poisoning `[hard]`
**Scenario:** EfficientNet-B0 on manufacturing defect data. 15-25% of one class has corrupted labels. That class stagnates at ~30% accuracy while others reach 90%+.

### Task 6 — Catastrophic Forgetting `[hard]`
**Scenario:** Fine-tuned ResNet-50. New task accuracy 91%, but original ImageNet capability dropped from 92% to 15% because `freeze_backbone=False` and `lr=0.01` were used.

---

## 🛠️ Action Space

**Investigation tools** (cost 1 step each):
```json
{"action_type": "fetch_logs",         "start_epoch": 1, "end_epoch": 10}
{"action_type": "fetch_config",       "keys": ["lr", "optimizer", "dropout"]}
{"action_type": "fetch_loss_curve",   "split": "val"}
{"action_type": "fetch_gpu_metrics"}
{"action_type": "fetch_class_metrics","class_id": 2}
```

**Terminal action** (triggers grading):
```json
{
  "action_type": "diagnose",
  "diagnosis":   "The model is overfitting — val loss diverges after epoch 10",
  "fix_type":    "config_change",
  "fix_detail":  "Add dropout=0.3 and weight_decay=1e-4",
  "confidence":  0.9
}
```

---

## 👁️ Observation Space

```json
{
  "task_id":         "class_imbalance_55",
  "difficulty":      "medium",
  "description":     "A MobileNetV2 was trained on a medical X-ray dataset...",
  "step_number":     2,
  "max_steps":       6,
  "tool_result":     {"class_metrics": {"0": {"support": 9500}, "1": {"support": 167}}},
  "action_history":  ["fetch_loss_curve", "fetch_class_metrics"]
}
```

---

## 🤖 Grading System

| Mode | How | When Active |
|---|---|---|
| **Keyword** (default) | Deterministic string-matching | Always on |
| **LLM** (optional) | Llama-3.3-70B-Instruct as judge | `USE_LLM_GRADING=true` |

*Blended score: 60% LLM + 40% Keyword for nuanced evaluation.*

---

## 🚀 Setup

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

---

*Built with ❤️ by Team AIverse for the Meta × PyTorch × HuggingFace OpenEnv Hackathon*