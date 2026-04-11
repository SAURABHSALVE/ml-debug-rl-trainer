---
title: ML Experiment Debugger — OpenEnv
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
short_description: "RL env — AI agents diagnose 6 real ML bugs (OpenEnv spec)"
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

# 🔬 ML Experiment Debugger — OpenEnv

> *"Every ML engineer has stared at a broken training run wondering why. We built an environment that teaches AI agents to investigate like a Senior MLE."*

Built for the **Meta × PyTorch × HuggingFace OpenEnv Hackathon** | Team: **AIverse**

🚀 **Live Demo:** [https://12-vinit-ml-experiment-debugger.hf.space](https://12-vinit-ml-experiment-debugger.hf.space)  
📖 **API Docs (Swagger):** [https://12-vinit-ml-experiment-debugger.hf.space/docs](https://12-vinit-ml-experiment-debugger.hf.space/docs)  
🤗 **HuggingFace Space:** [https://huggingface.co/spaces/12-vinit/ml-experiment-debugger](https://huggingface.co/spaces/12-vinit/ml-experiment-debugger)

---

## 🎯 The Vision
Modern LLM agents are good at coding, but struggle with **system-level diagnosis**. The ML Experiment Debugger is a professional-grade Reinforcement Learning environment where agents must diagnose 6 distinct, realistic ML failure modes. 

Unlike "toy" environments, this benchmark forces agents to manage a **global step budget**, verify hypotheses through **tool usage**, and avoid **redundant investigation**.

---

## 🧩 Technical Specifications

| Feature | Specification |
|:--- |:--- |
| **Action Space** | Structured investigation (5 tools) + 1 Terminal `diagnose` |
| **Observation** | High-fidelity data: Logs, Loss Curves, Class Metrics, Configs |
| **Step Budget** | **16 Global Steps** (Episodic) |
| **Difficulty Brackets** | **Easy, Medium, Hard** (1 of each per episode) |
| **Reward Function** | Multi-tiered (Investigation + Diagnosis + Bonuses) |
| **Compliance** | **100% OpenEnv Spec** (Pydantic models + YAML schema) |
| **Grading** | Blended (60% LLM Judge + 40% Keyword Match) |

---

## ♿ Accessibility for different users

### For ML engineers
- Run a reproducible demo with `python -m ml_env --demo`.
- After package installation, use the CLI command `ml-debug --demo`.
- Outputs are JSON-native and expose consistent metrics, logs, observations, and tool history.
- This makes integration easy for pipelines, evaluation scripts, and tooling frameworks.

### For data scientists
- Use the notebook at `notebooks/ML_Experiment_Debugger_Demo.ipynb`.
- The notebook contains sample runs, observation inspection, and visualization guidance.
- It also includes a short guide on how to add a new task type to the environment.

### For project reviewers / non-technical users
- The API docs at `/docs` provide clickable run buttons for all endpoints.
- The root path `/` redirects to `/docs` for immediate interactive exploration.
- Scenario descriptions and task summaries are written in plain language.

### For hackathon judges
- Launch the app with one command: `python -m uvicorn server.app:app --host 0.0.0.0 --port 7860`.
- The project is designed to be easy to run, verify, and review quickly.
- The README includes a short pitch and a clear checklist for submission readiness.

---

## 📓 How to add a new task

To add a new task:
1. Create a new task generator in `ml_env/tasks.py`.
2. Include a realistic description, training logs, config, and ground truth metadata.
3. Add the task generator function to the `TASK_POOL` in `ml_env/environment.py`.
4. Implement or reuse a grader in `ml_env/graders.py` to score diagnosis and fix details.
5. Verify the task by running the CLI demo and the `/docs` API flow.

---

## 📋 The 6-Bug Catalogue

The environment randomly samples 3 tasks (one from each bracket) to test generalizability across diverse ML domains.

### 🟢 Easy Bracket
1.  **Data Leakage (Tabular/XGBoost)**: A feature `last_purchase_date` is a direct proxy for the target. Accuracy is 100% — too good to be true.
2.  **NaN Initialization (NLP/BERT)**: Extreme initialization scale (`std=10.0`) crashes gradients instantly.

### 🟡 Medium Bracket
3.  **FP16 Underflow (LLM/Llama-3)**: LoRA fine-tuning without a gradient scaler. Gradients underflow to zero; loss stays stagnant.
4.  **Class Imbalance (CV/MobileNet)**: High accuracy (93%) masks a catastrophic failure to predict minority classes (9500 vs 200 samples).

### 🔴 Hard Bracket
5.  **Silent Data Poisoning (CV/EfficientNet)**: Sub-tle label corruption (15-25%) in a specific manufacturing class causes it to never cross 30% accuracy.
6.  **Catastrophic Forgetting (CV/ResNet)**: Fine-tuning a pretrained model with an aggressive LR and unfrozen backbone destroys original capability.

---

## 🏆 Sophisticated Reward Modeling
The reward system incentivizes professional investigative behavior, not just lucky guessing.

- **Investigation Score (0.5 max)**: Score scaled by the ratio of relevant tools identified vs. the total tools required for that bug type.
- **Diagnosis Score (0.5 max)**: LLM or Keyword assessment of the correctness of the root cause and fix.
- **⚡ Efficiency Bonus (+0.05)**: Awarded for a high-accuracy solution in ≤ 3 steps per task.
- **🏆 Trajectory Bonus (+0.05)**: Awarded on the Hard task only if the agent maintained high performance (>0.7) on earlier Easy and Medium tasks.
- **🔍 Professional Path Bonus (+0.1)**: Awarded for specific expert sequences (e.g., checking class distribution *before* diagnosing imbalance).
- **🚫 Spam Penalty (-0.01)**: Deducted for every redundant tool call already made.

---

## ⚙️ Fast Setup

### Local Run
```bash
pip install -r requirements.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### CLI demo
```bash
python -m ml_env --demo
```
After package installation, the CLI command is also available as:
```bash
ml-debug --demo
```

### Docker
```bash
docker build -t ml-debug-env .
docker run -p 7860:7860 ml-debug-env
```

---

## 🛠️ OpenEnv API Interface
The environment exposes standard endpoints:
- `POST /reset`: Start 3-task episode.
- `POST /step`: call investigation tools or `diagnose`.
- `GET /state`: Deep introspection into episode progress.
- `GET /tasks`: Details on current tasks.
- `GET /health`: Service health and readiness.

The root path `/` redirects to `/docs`, where reviewers can click through the API with live request buttons.

---

## ✅ Hackathon readiness checklist
- [x] One-command local launch documented
- [x] CLI demo available via `python -m ml_env --demo`
- [x] Consistent JSON output for metrics, logs, and observations
- [x] Interactive `/docs` API demo for non-technical reviewers
- [x] Notebook example added for data scientists
- [x] Clear “how to add a new task” guide included
- [x] Project summary and judge pitch included
- [x] Submission-ready README with concise, review-friendly sections

---
*Built with ❤️ for the Meta OpenEnv Hackathon 2026*