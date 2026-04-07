"""
Full test suite for ML Experiment Debugger.
Tests: graders, environment logic, API endpoints.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

from app import app
from env.environment import MLDebugEnv
from env.graders import grade_overfitting, grade_lr_explosion, grade_data_poisoning
from env.models import Action
from env.tasks import generate_overfitting_task, generate_lr_explosion_task, generate_poisoning_task

client = TestClient(app)


# ─── Grader Tests ──────────────────────────────────────────────────────────────

class TestOverfittingGrader:
    def setup_method(self):
        self.gt = generate_overfitting_task(42)["ground_truth"]

    def test_perfect_score(self):
        action = {
            "diagnosis": "The model is overfitting — train loss keeps dropping but val loss diverges after epoch 10",
            "fix_type": "config_change",
            "fix_detail": "Add dropout=0.3 and weight_decay=1e-4 to prevent overfitting",
            "confidence": 0.95,
        }
        score, breakdown, feedback = grade_overfitting(action, self.gt)
        assert score >= 0.8, f"Expected >=0.8, got {score}"
        assert breakdown["diagnosis"] == 0.5

    def test_partial_score_wrong_fix(self):
        action = {
            "diagnosis": "The model is overfitting badly",
            "fix_type": "wrong_type",
            "fix_detail": "Change something",
            "confidence": 0.5,
        }
        score, _, _ = grade_overfitting(action, self.gt)
        assert 0.3 <= score < 0.8

    def test_zero_score(self):
        action = {
            "diagnosis": "The GPU ran out of memory",
            "fix_type": "architecture_change",
            "fix_detail": "Use a smaller model",
            "confidence": 0.3,
        }
        score, _, _ = grade_overfitting(action, self.gt)
        assert score < 0.4


class TestLRExplosionGrader:
    def setup_method(self):
        self.gt = generate_lr_explosion_task(99)["ground_truth"]

    def test_perfect_score(self):
        action = {
            "diagnosis": "Learning rate 0.5 is too high for SGD — causing gradient explosion and NaN loss",
            "fix_type": "config_change",
            "fix_detail": "Reduce learning rate from 0.5 to 0.001 and add gradient clipping",
            "confidence": 0.9,
        }
        score, _, _ = grade_lr_explosion(action, self.gt)
        assert score >= 0.8

    def test_lr_value_in_range(self):
        action = {
            "diagnosis": "The learning rate is exploding causing NaN",
            "fix_type": "config_change",
            "fix_detail": "Set lr=0.0005",
            "confidence": 0.8,
        }
        score, breakdown, _ = grade_lr_explosion(action, self.gt)
        assert breakdown["diagnosis"] == 0.5
        assert score >= 0.7

    def test_wrong_diagnosis(self):
        action = {
            "diagnosis": "The model has too many parameters — reduce layers",
            "fix_type": "architecture_change",
            "fix_detail": "Use smaller model",
            "confidence": 0.4,
        }
        score, _, _ = grade_lr_explosion(action, self.gt)
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
        assert obs.steps_remaining == 4  # 5 - 1

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
            diagnosis="The model is overfitting — val loss diverges",
            fix_type="config_change",
            fix_detail="Add dropout=0.3",
            confidence=0.9,
        )
        obs, reward, done, info = env.step(action)
        assert done is False  # Not done — 2 more tasks
        assert obs.difficulty == "medium"  # Moved to task 2

    def test_full_episode(self):
        env = MLDebugEnv()
        env.reset()
        diagnose_easy = Action(
            action_type="diagnose",
            diagnosis="overfitting detected — val loss diverges",
            fix_type="config_change",
            fix_detail="Add dropout=0.3 weight_decay=1e-4",
            confidence=0.9,
        )
        # Task 1 — easy
        _, _, done, _ = env.step(diagnose_easy)
        assert not done

        # Task 2 — medium
        _, _, done, _ = env.step(Action(
            action_type="diagnose",
            diagnosis="learning rate 0.5 too high causing gradient explosion nan",
            fix_type="config_change",
            fix_detail="reduce lr to 0.001",
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
        data = resp.json()
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
        client.post("/reset")
        resp = client.post("/step", json={
            "action_type": "diagnose",
            "diagnosis": "The model is overfitting — val loss diverges significantly",
            "fix_type": "config_change",
            "fix_detail": "Add dropout=0.3 and weight_decay=1e-4",
            "confidence": 0.9,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reward"]["score"] >= 0.5

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
        tasks = resp.json()
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
        client.post("/reset")
        resp = client.post("/step", json={
            "action_type": "diagnose",
            "diagnosis": "The model is overfitting",
            "fix_type": "wrong_type",
            "fix_detail": "unknown",
            "confidence": 0.5,
        })
        data = resp.json()
        score = data["reward"]["score"]
        assert 0.0 < score < 1.0, "Should give partial credit"