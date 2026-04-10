"""
Full test suite for ML Experiment Debugger.
Tests: graders, environment logic, API endpoints.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from server.app import app
from ml_env.environment import MLDebugEnv
from ml_env.graders import grade_data_leakage, grade_fp16_underflow, grade_data_poisoning, grade_class_imbalance, grade_forgetting, grade_nan_init
from ml_env.models import Action
from ml_env.tasks import (
    generate_data_leakage_task, generate_fp16_underflow_task, generate_poisoning_task,
    generate_class_imbalance_task, generate_forgetting_task, generate_nan_init_task,
)

client = TestClient(app)


# ─── Grader Tests ──────────────────────────────────────────────────────────────

class TestDataLeakageGrader:
    def setup_method(self):
        self.gt = generate_data_leakage_task(42)["ground_truth"]

    def test_perfect_score(self):
        action = {
            "diagnosis": "The target variable latest_churn_flag is leaked into the training features",
            "fix_type": "config_change",
            "fix_detail": "Remove latest_churn_flag from features_used list",
            "confidence": 0.95,
        }
        score, breakdown, feedback = grade_data_leakage(action, self.gt)
        assert score >= 0.8, f"Expected >=0.8, got {score}"
        assert breakdown["diagnosis"] == 0.5

    def test_partial_score_wrong_fix(self):
        action = {
            "diagnosis": "There is a massive data leak causing perfect precision",
            "fix_type": "wrong_type",
            "fix_detail": "Change something",
            "confidence": 0.5,
        }
        score, _, _ = grade_data_leakage(action, self.gt)
        assert 0.3 <= score < 0.8

    def test_zero_score(self):
        action = {
            "diagnosis": "The model is overfitting",
            "fix_type": "architecture_change",
            "fix_detail": "Add dropout",
            "confidence": 0.3,
        }
        score, _, _ = grade_data_leakage(action, self.gt)
        assert score < 0.4


class TestFP16UnderflowGrader:
    def setup_method(self):
        self.gt = generate_fp16_underflow_task(99)["ground_truth"]

    def test_perfect_score(self):
        action = {
            "diagnosis": "Using fp16 precision without gradient scaling causes underflow to exactly 0",
            "fix_type": "config_change",
            "fix_detail": "Enable grad_scaler=True or switch precision to bf16",
            "confidence": 0.9,
        }
        score, _, _ = grade_fp16_underflow(action, self.gt)
        assert score >= 0.8

    def test_val_in_range(self):
        action = {
            "diagnosis": "The gradients are zero because of fp16 underflow",
            "fix_type": "config_change",
            "fix_detail": "Add a scaler to prevent zero gradients",
            "confidence": 0.8,
        }
        score, breakdown, _ = grade_fp16_underflow(action, self.gt)
        assert breakdown["diagnosis"] == 0.5
        assert score >= 0.7

    def test_wrong_diagnosis(self):
        action = {
            "diagnosis": "The model has too many parameters — reduce layers",
            "fix_type": "architecture_change",
            "fix_detail": "Use a smaller model",
            "confidence": 0.4,
        }
        score, _, _ = grade_fp16_underflow(action, self.gt)
        assert score < 0.3


class TestPoisoningGrader:
    def setup_method(self):
        self.task = generate_poisoning_task(7)
        self.gt = self.task["ground_truth"]
        self.poisoned = self.gt["poisoned_class"]

    def test_perfect_score(self):
        action = {
            "diagnosis": f"Silent data poisoning — 20% of labels in class_{self.poisoned} are corrupted",
            "fix_type": "data_fix",
            "fix_detail": f"Audit and re-annotate class_{self.poisoned} training samples",
            "confidence": 0.85,
        }
        score, breakdown, _ = grade_data_poisoning(action, self.gt)
        assert score >= 0.9
        assert breakdown["bug_type"] == 0.3
        assert breakdown["class_identified"] == 0.2

    def test_finds_bug_not_class(self):
        action = {
            "diagnosis": "Some labels are corrupted causing poor performance",
            "fix_type": "data_fix",
            "fix_detail": "Clean the training labels",
            "confidence": 0.6,
        }
        score, breakdown, _ = grade_data_poisoning(action, self.gt)
        assert breakdown["bug_type"] == 0.3
        assert breakdown["class_identified"] == 0.0

    def test_zero_score(self):
        action = {
            "diagnosis": "Learning rate is too high",
            "fix_type": "config_change",
            "fix_detail": "Reduce lr",
            "confidence": 0.5,
        }
        score, _, _ = grade_data_poisoning(action, self.gt)
        assert score < 0.2


# ─── Environment Tests ─────────────────────────────────────────────────────────

class TestEnvironment:
    def test_reset_returns_task1(self):
        env = MLDebugEnv()
        obs = env.reset()
        assert obs.difficulty == "easy"
        assert obs.step_number == 0
        assert obs.tool_result is None
        assert len(obs.action_history) == 0

    def test_fetch_loss_curve(self):
        env = MLDebugEnv()
        env.reset()
        action = Action(action_type="fetch_loss_curve", split="val")
        obs, reward, done, _ = env.step(action)
        assert obs.tool_result is not None
        assert "loss_curve" in obs.tool_result or "val_loss" in obs.tool_result
        assert done is False
        assert obs.steps_remaining == 15  # 16 - 1

    def test_redundant_tool_penalized(self):
        env = MLDebugEnv()
        env.reset()
        action = Action(action_type="fetch_loss_curve", split="val")
        env.step(action)
        _, reward2, _, _ = env.step(action)  # repeat
        assert reward2.score < 0

    def test_diagnose_advances_task(self):
        env = MLDebugEnv()
        env.reset()
        action = Action(
            action_type="diagnose",
            diagnosis="The target leaked into the features",
            fix_type="config_change",
            fix_detail="Remove leakage column",
            confidence=0.9,
        )
        # Avoid guessing penalty during tests
        env._called_tools = ["fetch_config"]
        obs, reward, done, info = env.step(action)
        assert done is False  # Not done — 2 more tasks
        assert obs.difficulty == "medium"  # Moved to task 2

    def test_full_episode(self):
        env = MLDebugEnv()
        env.reset()
        # Mock tool calls to avoid Guessing Penalty in tests
        env._called_tools = ["fetch_config", "fetch_loss_curve"]

        diagnose_easy = Action(
            action_type="diagnose",
            diagnosis="data leakage detected",
            fix_type="config_change",
            fix_detail="remove target column",
            confidence=0.9,
        )
        # Task 1 — easy
        _, _, done, _ = env.step(diagnose_easy)
        assert not done

        env._called_tools = ["fetch_logs", "fetch_config"]
        # Task 2 — medium
        _, _, done, _ = env.step(Action(
            action_type="diagnose",
            diagnosis="fp16 underflow to zero",
            fix_type="config_change",
            fix_detail="enable grad_scaler",
            confidence=0.9,
        ))
        assert not done

        # Task 3 — hard (budget=7, use 3 tool calls + 1 diagnose = 4 steps)
        env.step(Action(action_type="fetch_class_metrics", class_id=0))
        env.step(Action(action_type="fetch_class_metrics", class_id=1))
        env.step(Action(action_type="fetch_logs", start_epoch=10, end_epoch=15))

        _, _, done, info = env.step(Action(
            action_type="diagnose",
            diagnosis="silent data poisoning — class_0 labels are corrupted",
            fix_type="data_fix",
            fix_detail="re-annotate and clean class_0 training samples",
            confidence=0.8,
        ))
        assert done is True
        assert "average_score" in info


# ─── API Endpoint Tests ─────────────────────────────────────────────────────────

class TestAPI:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_reset(self):
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()["observation"]
        assert data["difficulty"] == "easy"
        assert data["step_number"] == 0
        assert data["tool_result"] is None

    def test_step_investigation(self):
        client.post("/reset")
        resp = client.post("/step", json={"action_type": "fetch_loss_curve", "split": "val"})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert data["done"] is False

    def test_step_diagnose(self):
        """Tests that diagnose action returns valid reward structure.
        Uses direct env access to guarantee easy task."""
        from ml_env.environment import MLDebugEnv as _Env
        from ml_env.models import Action as _Action
        env = _Env(seed=1)
        env.reset()
        env._called_tools = ["fetch_config"] # Avoid guessing penalty
        obs, reward, done, info = env.step(_Action(
            action_type="diagnose",
            diagnosis="data leakage detected",
            fix_type="config_change",
            fix_detail="remove latest_churn_flag",
            confidence=0.9,
        ))
        assert reward.total >= 0.5
        assert "Guessing Penalty" not in reward.feedback

    def test_state(self):
        client.post("/reset")
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "current_task" in data
        assert "steps_remaining" in data

    def test_tasks(self):
        client.post("/reset")
        resp = client.get("/tasks")
        assert resp.status_code == 200
        tasks = resp.json()["tasks"]
        # Episode always has exactly 3 tasks (1 easy, 1 medium, 1 hard)
        assert len(tasks) == 3
        difficulties = [t["difficulty"] for t in tasks]
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_no_suggest_diagnosis_endpoint(self):
        """Verify the answer-leaking endpoint does NOT exist."""
        resp = client.post("/suggest-diagnosis")
        assert resp.status_code == 405 or resp.status_code == 404

    def test_no_agent_stats_endpoint(self):
        """Verify DQN agent endpoints do NOT exist."""
        resp = client.get("/agent-stats")
        assert resp.status_code == 404

    def test_reward_partial_credit(self):
        """Verify partial credit is given for incomplete but relevant answers."""
        from ml_env.environment import MLDebugEnv as _Env
        from ml_env.models import Action as _Action
        env = _Env(seed=1)
        env.reset()
        env._called_tools = ["fetch_loss_curve"] # Avoid guessing penalty masking partial credit
        _, reward, _, _ = env.step(_Action(
            action_type="diagnose",
            diagnosis="There is data leakage",
            fix_type="wrong_type",
            fix_detail="unknown",
            confidence=0.5,
        ))
        score = reward.total
        assert 0.0 < score < 1.0, f"Should give partial credit, got {score}"


# ─── New Grader Tests ───────────────────────────────────────────────────────────

class TestClassImbalanceGrader:
    def setup_method(self):
        self.gt = generate_class_imbalance_task(55)["ground_truth"]

    def test_perfect_score(self):
        action = {
            "diagnosis": "Severe class imbalance — 95% of data is the majority class, model predicts it always",
            "fix_type": "config_change",
            "fix_detail": "Add class weights and use weighted loss or oversample minority classes with SMOTE",
            "confidence": 0.9,
        }
        score, breakdown, feedback = grade_class_imbalance(action, self.gt)
        assert score >= 0.8, f"Expected >=0.8, got {score}"
        assert breakdown["diagnosis"] == 0.5

    def test_partial_score_diagnosis_only(self):
        action = {
            "diagnosis": "The dataset has a dominant class causing skewed predictions",
            "fix_type": "wrong_type",
            "fix_detail": "Try something",
            "confidence": 0.4,
        }
        score, breakdown, _ = grade_class_imbalance(action, self.gt)
        assert 0.3 <= score < 0.8

    def test_zero_score(self):
        action = {
            "diagnosis": "The learning rate is too high",
            "fix_type": "config_change",
            "fix_detail": "Reduce lr",
            "confidence": 0.3,
        }
        score, _, _ = grade_class_imbalance(action, self.gt)
        assert score <= 0.2


class TestForgettingGrader:
    def setup_method(self):
        self.gt = generate_forgetting_task(33)["ground_truth"]

    def test_perfect_score(self):
        action = {
            "diagnosis": "Catastrophic forgetting — fine-tuning without EWC overwrites original task weights completely",
            "fix_type": "config_change",
            "fix_detail": "Freeze the backbone layers and use elastic weight consolidation (EWC) or a replay buffer",
            "confidence": 0.9,
        }
        score, breakdown, feedback = grade_forgetting(action, self.gt)
        assert score >= 0.8, f"Expected >=0.8, got {score}"
        assert breakdown["diagnosis"] == 0.5

    def test_partial_score_diagnosis_only(self):
        action = {
            "diagnosis": "The model is forgetting the original pretrained task representation",
            "fix_type": "wrong_type",
            "fix_detail": "Do something about it",
            "confidence": 0.5,
        }
        score, breakdown, _ = grade_forgetting(action, self.gt)
        assert 0.3 <= score < 0.8

    def test_zero_score(self):
        action = {
            "diagnosis": "The validation loss is too high — overfitting",
            "fix_type": "config_change",
            "fix_detail": "Add dropout",
            "confidence": 0.3,
        }
        score, _, _ = grade_forgetting(action, self.gt)
        assert score <= 0.2


class TestTaskPool:
    def test_episode_always_has_3_tasks(self):
        """Verify every reset gives exactly 3 tasks (easy, medium, hard)."""
        for seed in [1, 42, 999, 12345]:
            env = MLDebugEnv(seed=seed)
            obs = env.reset()
            tasks = env.list_tasks()
            assert len(tasks) == 3, f"Seed {seed}: expected 3 tasks, got {len(tasks)}"
            difficulties = {t["difficulty"] for t in tasks}
            assert difficulties == {"easy", "medium", "hard"}, f"Seed {seed}: wrong difficulties {difficulties}"

    def test_medium_task_pool_varies(self):
        """Verify different seeds can produce different medium tasks from the pool."""
        seen_medium_task_ids = set()
        for seed in range(50):
            env = MLDebugEnv(seed=seed)
            env.reset()
            tasks = env.list_tasks()
            medium_task = next(t for t in tasks if t["difficulty"] == "medium")
            seen_medium_task_ids.add(medium_task["task_id"].split("_")[0] + "_" + medium_task["task_id"].split("_")[1])
        # Should see both types of medium tasks across 50 episodes
        assert len(seen_medium_task_ids) >= 2, f"Expected variety in medium tasks, only saw: {seen_medium_task_ids}"

    def test_easy_task_pool_varies(self):
        """Verify the easy pool now has 2 variants — data leakage and nan_init."""
        seen_easy_task_types = set()
        for seed in range(50):
            env = MLDebugEnv(seed=seed)
            env.reset()
            tasks = env.list_tasks()
            easy_task = next(t for t in tasks if t["difficulty"] == "easy")
            # task_id format: data_leakage_NNNN or nan_init_NNNN
            task_type = easy_task["task_id"].rsplit("_", 1)[0]
            if task_type == "data": task_type = "data_leakage" # handle data_leakage_seed format
            seen_easy_task_types.add(task_type)
        assert len(seen_easy_task_types) >= 2, f"Easy pool should have 2 variants, only saw: {seen_easy_task_types}"


# ─── NaN Init Grader Tests ────────────────────────────────────────────────

class TestNaNInitGrader:
    def setup_method(self):
        self.gt = generate_nan_init_task(77)["ground_truth"]

    def test_perfect_score(self):
        action = {
            "diagnosis": "Bad weight initialization — init_std=10.0 is 500x too large, causing NaN from first forward pass",
            "fix_type": "config_change",
            "fix_detail": "Set init_std=0.02 to match standard BERT initialization (Xavier/normal init)",
            "confidence": 0.95,
        }
        score, breakdown, _ = grade_nan_init(action, self.gt)
        assert score >= 0.8, f"Expected >=0.8, got {score}"
        assert breakdown["diagnosis"] == 0.5

    def test_partial_score_diagnosis_only(self):
        action = {
            "diagnosis": "The model has a bad initialization causing NaN loss from the first epoch",
            "fix_type": "wrong_type",
            "fix_detail": "Try something different",
            "confidence": 0.5,
        }
        score, breakdown, _ = grade_nan_init(action, self.gt)
        assert 0.3 <= score < 0.8

    def test_zero_score(self):
        action = {
            "diagnosis": "The model is overfitting to the training data",
            "fix_type": "config_change",
            "fix_detail": "Add dropout and reduce model capacity",
            "confidence": 0.4,
        }
        score, _, _ = grade_nan_init(action, self.gt)
        assert score <= 0.2