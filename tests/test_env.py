"""
Pytest suite for ML Experiment Debugger — multi-step action_type design.
"""
import pytest
from fastapi.testclient import TestClient

from env.environment import MLDebugEnv
from env.models import Action, Observation, Reward
from env.graders import grade_overfitting, grade_lr_schedule, grade_data_poisoning
from env.reward import apply_trajectory_bonus, compute_episode_summary, compute_efficiency_bonus
from env.tasks import generate_tasks
from app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

# ── Investigation actions ──

def fetch_logs(epochs="1-10"):
    return Action(action_type="fetch_logs", epochs=epochs)

def fetch_config(keys=None):
    return Action(action_type="fetch_config", keys=keys)

def fetch_loss_curve(split="val"):
    return Action(action_type="fetch_loss_curve", split=split)

def fetch_diagnostics(check="trends"):
    return Action(action_type="fetch_diagnostics", check=check)

def fetch_class_data(class_id=0):
    return Action(action_type="fetch_class_data", class_id=class_id)

# ── Terminal actions ──

def diagnose_easy_perfect():
    return Action(
        action_type="diagnose",
        diagnosis="The model is overfitting. Train loss drops but val loss diverges after epoch 10.",
        fix_type="config_change",
        fix_detail="Add dropout=0.3 and weight_decay=1e-4 to prevent overfitting.",
        confidence=0.95,
    )

def diagnose_easy_partial():
    return Action(
        action_type="diagnose",
        diagnosis="The val loss is increasing which is a problem.",
        fix_type="config_change",
        fix_detail="Reduce epochs to 10.",
        confidence=0.5,
    )

def diagnose_empty():
    return Action(
        action_type="diagnose",
        diagnosis="",
        fix_type="config_change",
        fix_detail="",
        confidence=0.0,
    )

def diagnose_medium_perfect():
    return Action(
        action_type="diagnose",
        diagnosis="The learning rate is too high causing gradient explosion and NaN losses.",
        fix_type="config_change",
        fix_detail="Reduce lr from current value to 0.001 and add gradient_clip=1.0.",
        confidence=0.9,
    )

def diagnose_hard_perfect(poisoned_class: int):
    return Action(
        action_type="diagnose",
        diagnosis=(
            f"Silent label corruption in class_{poisoned_class}. "
            f"Per-class val accuracy stagnates at ~0.35 while others reach 0.9+."
        ),
        fix_type="data_fix",
        fix_detail=f"Re-annotate and filter class_{poisoned_class} training samples.",
        confidence=0.85,
    )


# ---------------------------------------------------------------------------
# Grader unit tests
# ---------------------------------------------------------------------------

class TestOverfittingGrader:
    tasks = generate_tasks()
    gt = tasks[0].ground_truth

    def test_perfect_answer_scores_high(self):
        assert grade_overfitting(diagnose_easy_perfect(), self.gt)["score"] >= 0.9

    def test_empty_answer_scores_low(self):
        assert grade_overfitting(diagnose_empty(), self.gt)["score"] <= 0.1

    def test_partial_answer_scores_between(self):
        r = grade_overfitting(diagnose_easy_partial(), self.gt)
        assert 0.1 < r["score"] < 0.8

    def test_result_has_required_keys(self):
        r = grade_overfitting(diagnose_easy_perfect(), self.gt)
        for k in ["score", "diagnosis_score", "fix_score", "reasoning"]:
            assert k in r

    def test_augmentation_fix_accepted(self):
        a = Action(action_type="diagnose", diagnosis="overfitting detected",
                   fix_type="config_change",
                   fix_detail="add data augmentation: random flips and crops",
                   confidence=0.8)
        assert grade_overfitting(a, self.gt)["fix_score"] >= 0.35

    def test_early_stopping_fix_accepted(self):
        a = Action(action_type="diagnose", diagnosis="overfitting",
                   fix_type="config_change",
                   fix_detail="add early stopping with patience=3",
                   confidence=0.8)
        assert grade_overfitting(a, self.gt)["fix_score"] >= 0.3


class TestLRScheduleGrader:
    tasks = generate_tasks()
    gt = tasks[1].ground_truth

    def test_perfect_answer_scores_high(self):
        assert grade_lr_schedule(diagnose_medium_perfect(), self.gt)["score"] >= 0.8

    def test_empty_answer_scores_zero(self):
        assert grade_lr_schedule(diagnose_empty(), self.gt)["score"] == 0.0

    def test_valid_lr_value_accepted(self):
        a = Action(action_type="diagnose",
                   diagnosis="learning rate is too high and causing NaN losses",
                   fix_type="config_change", fix_detail="set lr to 0.0001", confidence=0.9)
        assert grade_lr_schedule(a, self.gt)["fix_score"] >= 0.5


class TestDataPoisoningGrader:
    tasks = generate_tasks()
    gt = tasks[2].ground_truth
    pc = gt["poisoned_class"]

    def test_perfect_answer_scores_high(self):
        assert grade_data_poisoning(diagnose_hard_perfect(self.pc), self.gt)["score"] >= 0.85

    def test_empty_answer_scores_zero(self):
        assert grade_data_poisoning(diagnose_empty(), self.gt)["score"] == 0.0

    def test_wrong_class_penalized(self):
        wrong = (self.pc + 1) % 5
        a = Action(action_type="diagnose", diagnosis=f"class_{wrong} has corrupted labels",
                   fix_type="data_fix", fix_detail=f"Re-annotate class_{wrong}", confidence=0.7)
        assert grade_data_poisoning(a, self.gt)["score"] < \
               grade_data_poisoning(diagnose_hard_perfect(self.pc), self.gt)["score"]


# ---------------------------------------------------------------------------
# Environment lifecycle
# ---------------------------------------------------------------------------

class TestMLDebugEnvLifecycle:
    def test_reset_returns_observation(self):
        env = MLDebugEnv()
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.task_id != "done"
        assert obs.difficulty == "easy"

    def test_reset_provides_full_task_context(self):
        env = MLDebugEnv()
        obs = env.reset()
        assert obs.training_logs is not None
        assert obs.config_yaml is not None
        assert obs.loss_curve is not None
        assert obs.gpu_metrics is not None

    def test_available_actions_listed(self):
        env = MLDebugEnv()
        obs = env.reset()
        assert "diagnose" in obs.available_actions
        assert "fetch_logs" in obs.available_actions

    def test_fetch_logs_returns_result(self):
        env = MLDebugEnv()
        env.reset()
        obs, reward, done, info = env.step(fetch_logs("1-5"))
        assert not done
        assert obs.action_result is not None
        assert reward.fix_quality == 0.0

    def test_fetch_config_returns_result(self):
        env = MLDebugEnv()
        env.reset()
        obs, _, _, _ = env.step(fetch_config(["lr"]))
        assert obs.action_result is not None
        assert "lr" in obs.action_result

    def test_fetch_loss_curve_returns_result(self):
        env = MLDebugEnv()
        env.reset()
        obs, reward, _, _ = env.step(fetch_loss_curve("val"))
        assert obs.action_result is not None
        assert reward.intermediate_signal >= 0.0

    def test_fetch_diagnostics_returns_result(self):
        env = MLDebugEnv()
        env.reset()
        obs, _, _, _ = env.step(fetch_diagnostics("overfitting"))
        assert obs.action_result is not None

    def test_fetch_class_data_on_hard_task(self):
        tasks = generate_tasks()
        env = MLDebugEnv()
        env.reset()
        env.step(diagnose_easy_perfect())
        env.step(diagnose_medium_perfect())
        obs, reward, _, _ = env.step(fetch_class_data(tasks[2].ground_truth["poisoned_class"]))
        assert obs.action_result is not None
        assert reward.intermediate_signal == 0.03   # direct hit on poisoned class

    def test_context_absent_after_first_step(self):
        env = MLDebugEnv()
        env.reset()
        obs, _, _, _ = env.step(fetch_logs("1-5"))
        assert obs.training_logs is None
        assert obs.config_yaml is None

    def test_diagnose_ends_task_and_advances(self):
        env = MLDebugEnv()
        env.reset()
        obs, reward, done, info = env.step(diagnose_easy_perfect())
        assert not done
        assert obs.difficulty == "medium"
        assert reward.fix_quality > 0

    def test_action_history_resets_between_tasks(self):
        env = MLDebugEnv()
        env.reset()
        env.step(fetch_logs("1-5"))
        obs, _, _, _ = env.step(diagnose_easy_perfect())
        assert obs.action_history == []

    def test_redundant_call_penalised(self):
        env = MLDebugEnv()
        env.reset()
        env.step(fetch_logs("1-10"))
        _, reward, _, _ = env.step(fetch_logs("1-10"))   # identical
        assert reward.intermediate_signal == -0.01
        assert reward.score == -0.01

    def test_cumulative_signal_tracked(self):
        env = MLDebugEnv()
        env.reset()
        env.step(fetch_loss_curve("val"))
        _, reward, _, _ = env.step(fetch_config(["lr"]))
        assert reward.cumulative_episode_signal > 0.0

    def test_steps_remaining_decrements(self):
        env = MLDebugEnv(max_steps_per_task=15)
        obs = env.reset()
        assert obs.steps_remaining == 15
        obs, _, _, _ = env.step(fetch_logs("1-5"))
        assert obs.steps_remaining == 14

    def test_hint_after_step_8(self):
        env = MLDebugEnv(max_steps_per_task=15)
        obs = env.reset()
        assert obs.hint is None
        for _ in range(8):
            obs, _, _, _ = env.step(fetch_logs("1-5"))
        assert obs.hint is not None

    def test_efficiency_bonus_fast_solve(self):
        env = MLDebugEnv(max_steps_per_task=15)
        env.reset()
        _, reward, _, _ = env.step(diagnose_easy_perfect())
        assert reward.efficiency_bonus > 0.0

    def test_full_episode_completes(self):
        tasks = generate_tasks()
        env = MLDebugEnv()
        env.reset()
        env.step(fetch_logs("15-20"))
        env.step(fetch_diagnostics("overfitting"))
        _, r1, done, _ = env.step(diagnose_easy_perfect())
        assert not done

        env.step(fetch_config(["lr"]))
        env.step(fetch_loss_curve("train"))
        _, r2, done, _ = env.step(diagnose_medium_perfect())
        assert not done

        pc = tasks[2].ground_truth["poisoned_class"]
        env.step(fetch_diagnostics("class_balance"))
        env.step(fetch_class_data(pc))
        _, r3, done, info = env.step(diagnose_hard_perfect(pc))
        assert done
        assert "episode_summary" in info
        for r in [r1, r2, r3]:
            assert 0.0 <= r.score <= 1.0

    def test_step_after_done_raises(self):
        tasks = generate_tasks()
        env = MLDebugEnv()
        env.reset()
        pc = tasks[2].ground_truth["poisoned_class"]
        for a in [diagnose_easy_perfect(), diagnose_medium_perfect(), diagnose_hard_perfect(pc)]:
            env.step(a)
        with pytest.raises(RuntimeError):
            env.step(diagnose_empty())

    def test_reset_clears_state(self):
        tasks = generate_tasks()
        env = MLDebugEnv()
        env.reset()
        pc = tasks[2].ground_truth["poisoned_class"]
        for a in [diagnose_easy_perfect(), diagnose_medium_perfect(), diagnose_hard_perfect(pc)]:
            env.step(a)
        obs = env.reset()
        assert obs.difficulty == "easy"
        assert not env.state()["done"]

    def test_state_fields(self):
        env = MLDebugEnv()
        env.reset()
        s = env.state()
        for k in ["task_index", "task_step", "done", "current_task_id", "scores_so_far"]:
            assert k in s


# ---------------------------------------------------------------------------
# Reward logic
# ---------------------------------------------------------------------------

class TestRewardLogic:
    def test_trajectory_bonus(self):
        assert apply_trajectory_bonus(0.8, 0.7, 0.5) == 0.55

    def test_trajectory_bonus_not_applied(self):
        assert apply_trajectory_bonus(0.5, 0.7, 0.5) == 0.5

    def test_trajectory_bonus_capped(self):
        assert apply_trajectory_bonus(1.0, 1.0, 0.99) == 1.0

    def test_efficiency_bonus_fast(self):
        assert compute_efficiency_bonus(1, 15) == 0.05

    def test_efficiency_bonus_moderate(self):
        assert compute_efficiency_bonus(7, 15) == 0.03

    def test_efficiency_bonus_slow(self):
        assert compute_efficiency_bonus(15, 15) == 0.0

    def test_episode_summary(self):
        s = compute_episode_summary({"easy": 0.9, "medium": 0.8, "hard": 0.7})
        assert 0.0 <= s["average_score"] <= 1.0


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestAPI:
    def test_health(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_reset_observation(self):
        r = client.post("/reset")
        assert r.status_code == 200
        data = r.json()
        assert data["difficulty"] == "easy"
        assert "diagnose" in data["available_actions"]
        assert data["training_logs"] is not None

    def test_fetch_logs_via_api(self):
        client.post("/reset")
        r = client.post("/step", json={"action_type": "fetch_logs", "epochs": "1-5"})
        assert r.status_code == 200
        d = r.json()
        assert d["done"] is False
        assert d["observation"]["action_result"] is not None
        assert d["reward"]["fix_quality"] == 0.0

    def test_fetch_config_via_api(self):
        client.post("/reset")
        r = client.post("/step", json={"action_type": "fetch_config", "keys": ["lr"]})
        assert r.status_code == 200
        assert "lr" in r.json()["observation"]["action_result"]

    def test_fetch_loss_curve_via_api(self):
        client.post("/reset")
        r = client.post("/step", json={"action_type": "fetch_loss_curve", "split": "val"})
        assert r.status_code == 200
        assert r.json()["observation"]["action_result"] is not None

    def test_diagnose_advances_task(self):
        client.post("/reset")
        r = client.post("/step", json={
            "action_type": "diagnose",
            "diagnosis": "overfitting — val loss diverges",
            "fix_type": "config_change",
            "fix_detail": "add dropout=0.3 and weight_decay=1e-4",
            "confidence": 0.9,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["observation"]["difficulty"] == "medium"
        assert d["reward"]["fix_quality"] > 0.0

    def test_redundant_call_penalty_via_api(self):
        client.post("/reset")
        client.post("/step", json={"action_type": "fetch_config", "keys": ["lr"]})
        r = client.post("/step", json={"action_type": "fetch_config", "keys": ["lr"]})
        assert r.json()["reward"]["score"] == -0.01

    def test_full_episode_via_api(self):
        client.post("/reset")
        tasks = generate_tasks()
        pc = tasks[2].ground_truth["poisoned_class"]
        actions = [
            {"action_type": "diagnose", "diagnosis": "overfitting",
             "fix_type": "config_change", "fix_detail": "add dropout", "confidence": 0.9},
            {"action_type": "diagnose", "diagnosis": "lr too high causing NaN",
             "fix_type": "config_change", "fix_detail": "set lr to 0.001", "confidence": 0.9},
            {"action_type": "diagnose", "diagnosis": f"label corruption in class_{pc}",
             "fix_type": "data_fix", "fix_detail": f"re-annotate class_{pc}", "confidence": 0.8},
        ]
        last = None
        for a in actions:
            last = client.post("/step", json=a).json()
        assert last["done"] is True
        assert "episode_summary" in last["info"]

    def test_step_after_done_returns_400(self):
        client.post("/reset")
        tasks = generate_tasks()
        pc = tasks[2].ground_truth["poisoned_class"]
        for a in [
            {"action_type": "diagnose", "diagnosis": "overfit", "fix_type": "config_change",
             "fix_detail": "add dropout", "confidence": 0.9},
            {"action_type": "diagnose", "diagnosis": "lr too high", "fix_type": "config_change",
             "fix_detail": "set lr to 0.001", "confidence": 0.9},
            {"action_type": "diagnose", "diagnosis": f"label corruption class_{pc}",
             "fix_type": "data_fix", "fix_detail": f"re-annotate class_{pc}", "confidence": 0.8},
        ]:
            client.post("/step", json=a)
        r = client.post("/step", json={"action_type": "diagnose", "diagnosis": "x",
                                        "fix_type": "config_change", "fix_detail": "x"})
        assert r.status_code == 400

    def test_state_endpoint(self):
        client.post("/reset")
        r = client.get("/state")
        assert r.status_code == 200
        d = r.json()
        assert "task_index" in d and "done" in d and "task_step" in d

    def test_tasks_endpoint(self):
        r = client.get("/tasks")
        assert len(r.json()) == 3
        assert {t["difficulty"] for t in r.json()} == {"easy", "medium", "hard"}
