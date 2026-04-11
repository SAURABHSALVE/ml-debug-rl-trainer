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

A selection-ready, production-oriented benchmark environment where agents investigate broken ML runs with tool-based reasoning, structured diagnostics, and graded fixes.

Built for the **Meta × PyTorch × HuggingFace OpenEnv Hackathon** | Team: **AIverse**

---

## What this project delivers

- A clean OpenEnv-compatible RL environment for diagnosing ML failures.
- Six realistic failure modalities across easy, medium, and hard brackets.
- A reproducible CLI demo with clean JSON output.
- A notebook demo for data-scientist exploration and visualization.
- A FastAPI app with `/docs` interactive buttons for non-technical reviewers.
- One-command launch instructions and a judge-ready verification checklist.

---

## Quick start

### Run the demo CLI
```bash
python -m ml_env --demo
```

### Run CLI with JSON output
```bash
python -m ml_env --demo --json
```

### Start the server
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```
Open: `http://localhost:7860/docs`

---

## New feature summary

- **ML engineers** get a reproducible CLI path and `ml-debug` package entrypoint.
- **Data scientists** get a ready-to-run notebook in `notebooks/ML_Experiment_Debugger_Demo.ipynb`.
- **Reviewers** get a clickable API demo via `/docs` and plain-language scenario descriptions.
- **Judges** get one-command app launch and a submission-ready checklist.

---

## Usage by audience

### ML engineers
- Use `python -m ml_env --demo` for a deterministic sample episode.
- Use `python -m ml_env --demo --json` to inspect output programmatically.
- Integrate the environment into pipelines using JSON outputs and structured `observation` / `reward` data.

### Data scientists
- Open `notebooks/ML_Experiment_Debugger_Demo.ipynb`.
- Run sample steps, plot training curves, and explore observation data.
- Follow the included guide for adding a new task to the benchmark.

### Reviewers / non-technical users
- Start the server and use `/docs` to exercise the environment with buttons.
- The root path `/` forwards to the interactive documentation.
- All endpoints return JSON, making their behavior easy to verify.

### Judges
- Launch with one command.
- Use the checklist below to confirm the feature paths.
- The README is structured for fast evaluation.

---

## Core features

- **3-task episodic flow**: 1 easy, 1 medium, 1 hard task per episode.
- **5 investigation tools** plus a terminal `diagnose` action.
- **Global step budget of 16** across the episode.
- **Structured outputs**: observations, rewards, task state, and tool history.
- **Grading blend** of keyword matching and optional LLM scoring.
- **Easy extensibility** for adding new tasks and graders.

---

## Task catalogue

- **Easy**: Data Leakage, NaN Initialization
- **Medium**: FP16 Underflow, Class Imbalance
- **Hard**: Silent Data Poisoning, Catastrophic Forgetting

---

## How to add a new task

1. Add a new generator function in `ml_env/tasks.py`.
2. Add it to `TASK_POOL` in `ml_env/environment.py`.
3. Add or extend grading logic in `ml_env/graders.py`.
4. Validate using `python -m ml_env --demo` and the `/docs` UI.

---

## Judge validation commands

```bash
python -m ml_env --demo --json
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```
Then open:
```bash
http://localhost:7860/docs
```

---

## Hackathon readiness checklist

- [x] One-command launch documented
- [x] CLI demo available and reproducible
- [x] JSON output path available for integration
- [x] Notebook demo included for data scientists
- [x] `/docs` interactive API available for reviewers
- [x] Task extension guide included
- [x] Submission-ready README with concise audience-specific sections

---

*Built with ❤️ for the Meta OpenEnv Hackathon 2026*