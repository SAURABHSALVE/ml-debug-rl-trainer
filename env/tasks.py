import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

@dataclass
class Task:
    id: str
    difficulty: str
    description: str
    training_logs: List[str]
    config_yaml: str
    loss_curve: Dict[str, List[float]]
    gpu_metrics: Dict[str, List[float]]
    ground_truth: Dict[str, Any]   # Hidden from agent, used by grader


def _make_overfitting_task(seed: int) -> Task:
    """Easy: Classic overfitting — train loss drops, val loss rises"""
    rng = random.Random(seed)
    epochs = 20
    noise = lambda: rng.uniform(-0.02, 0.02)

    train_loss = [2.5 * (0.82 ** i) + noise() for i in range(epochs)]
    val_loss = [2.5 * (0.85 ** i) + noise() for i in range(10)] + \
               [val + 0.04 * (i - 9) + noise() for i, val in enumerate(
                   [2.5 * (0.85 ** 9)] * 10)]

    config = {
        "model": "resnet50",
        "lr": 0.001,
        "epochs": epochs,
        "batch_size": 32,
        "dropout": 0.0,          # <-- The bug (no regularization)
        "weight_decay": 0.0,
        "dataset_size": 1200,
    }

    logs = [
        f"Epoch {i+1}: train_loss={train_loss[i]:.4f}, val_loss={val_loss[i]:.4f}, "
        f"train_acc={0.6 + i*0.018:.3f}, val_acc={0.58 + i*0.01 if i < 10 else 0.68 - i*0.005:.3f}"
        for i in range(epochs)
    ]

    return Task(
        id=f"overfitting_{seed}",
        difficulty="easy",
        description=(
            "A ResNet-50 was trained for 20 epochs on a small image classification dataset. "
            "Analyze the training logs and loss curves. Identify what is going wrong "
            "and suggest a concrete fix."
        ),
        training_logs=logs,
        config_yaml=yaml.dump(config),
        loss_curve={"train": train_loss, "val": val_loss},
        gpu_metrics={
            "memory_mb": [rng.uniform(4800, 5200) for _ in range(epochs)],
            "util_pct": [rng.uniform(85, 98) for _ in range(epochs)],
        },
        ground_truth={
            "bug_type": "overfitting",
            "key_signal": "val_loss_divergence",
            "correct_fix_type": "config_change",
            "correct_fix_keys": ["dropout", "weight_decay"],
            "acceptable_fixes": ["dropout > 0", "weight_decay > 0", "data augmentation", "early stopping"],
        }
    )


def _make_lr_schedule_task(seed: int) -> Task:
    """Medium: Learning rate is too high — loss explodes then NaN"""
    rng = random.Random(seed)
    lr = rng.choice([0.1, 0.5, 1.0])    # Too high
    epochs = 15

    train_loss = []
    for i in range(epochs):
        if i < 5:
            train_loss.append(2.0 - i * 0.1 + rng.uniform(-0.05, 0.05))
        elif i < 10:
            train_loss.append(train_loss[-1] + rng.uniform(0.2, 0.8))  # exploding
        else:
            train_loss.append(float('nan'))

    val_loss = [l + rng.uniform(0.1, 0.3) if not np.isnan(l) else float('nan')
                for l in train_loss]

    config = {
        "model": "transformer_base",
        "optimizer": "SGD",
        "lr": lr,                          # <-- The bug
        "lr_schedule": "none",
        "warmup_steps": 0,
        "gradient_clip": None,
        "epochs": epochs,
    }

    logs = []
    for i in range(epochs):
        tl = train_loss[i]
        if np.isnan(tl):
            logs.append(f"Epoch {i+1}: train_loss=nan, val_loss=nan, grad_norm=inf — TRAINING UNSTABLE")
        else:
            grad_norm = 0.5 + i * 0.4 + rng.uniform(-0.1, 0.1)
            logs.append(f"Epoch {i+1}: train_loss={tl:.4f}, val_loss={val_loss[i]:.4f}, grad_norm={grad_norm:.3f}")

    return Task(
        id=f"lr_explosion_{seed}",
        difficulty="medium",
        description=(
            f"A Transformer model is being trained with SGD. Training collapses mid-way. "
            f"Identify the misconfigured hyperparameter and provide the corrected value."
        ),
        training_logs=logs,
        config_yaml=yaml.dump(config),
        loss_curve={
            "train": [0.0 if np.isnan(x) else x for x in train_loss],
            "val":   [0.0 if np.isnan(x) else x for x in val_loss],
        },
        gpu_metrics={
            "memory_mb": [rng.uniform(6000, 7000) for _ in range(epochs)],
            "util_pct": [rng.uniform(90, 99) for _ in range(10)] + [5.0] * 5,
        },
        ground_truth={
            "bug_type": "lr_too_high",
            "buggy_key": "lr",
            "buggy_value": lr,
            "correct_fix_type": "config_change",
            "correct_fix_key": "lr",
            "correct_value_range": [1e-5, 1e-2],    # Any lr in this range is correct
            "also_acceptable": ["gradient_clip", "warmup_steps", "lr_schedule"],
        }
    )


def _make_data_poisoning_task(seed: int) -> Task:
    """Hard: Silent label corruption in N% of training batch"""
    rng = random.Random(seed)
    poison_pct = rng.choice([0.15, 0.20, 0.25])
    poison_class = rng.randint(0, 4)
    epochs = 25

    # Poisoned training shows: class-specific accuracy anomaly + slower convergence
    train_loss = [2.3 * (0.88 ** i) + rng.uniform(-0.03, 0.03) for i in range(epochs)]
    val_loss   = [2.3 * (0.84 ** i) + rng.uniform(-0.03, 0.03) for i in range(epochs)]

    # Per-class accuracy — poisoned class stays low
    per_class_acc = {}
    for c in range(5):
        if c == poison_class:
            per_class_acc[f"class_{c}"] = [
                round(0.35 + rng.uniform(-0.05, 0.05), 3) for _ in range(epochs)
            ]
        else:
            per_class_acc[f"class_{c}"] = [
                round(min(0.95, 0.4 + i * 0.022 + rng.uniform(-0.02, 0.02)), 3)
                for i in range(epochs)
            ]

    config = {
        "model": "efficientnet_b0",
        "num_classes": 5,
        "lr": 0.0003,
        "epochs": epochs,
        "batch_size": 64,
        "dataset": "custom_manufacturing_defects",
        "data_loader": "standard",
    }

    logs = []
    for i in range(epochs):
        accs = ", ".join(f"c{c}={per_class_acc[f'class_{c}'][i]}" for c in range(5))
        logs.append(
            f"Epoch {i+1}: train_loss={train_loss[i]:.4f}, val_loss={val_loss[i]:.4f}, "
            f"per_class_val_acc=[{accs}]"
        )
    # Plant a subtle clue mid-training
    logs[12] += f" [WARNING: label_consistency_check skipped for batch 47]"

    return Task(
        id=f"data_poisoning_{seed}",
        difficulty="hard",
        description=(
            "An EfficientNet model trained on a 5-class manufacturing defect dataset "
            "is converging well on global metrics but will fail in production. "
            "Analyze all available signals — logs, per-class accuracy, config — "
            "and identify the silent data quality issue. Name the affected class."
        ),
        training_logs=logs,
        config_yaml=yaml.dump(config),
        loss_curve={"train": train_loss, "val": val_loss, **per_class_acc},
        gpu_metrics={
            "memory_mb": [rng.uniform(7500, 8500) for _ in range(epochs)],
            "util_pct": [rng.uniform(88, 96) for _ in range(epochs)],
        },
        ground_truth={
            "bug_type": "label_corruption",
            "poisoned_class": poison_class,
            "poison_pct": poison_pct,
            "correct_fix_type": "data_fix",
            "correct_fix_detail": f"Re-annotate or filter class_{poison_class} training samples",
        }
    )


def generate_tasks(seeds: Optional[List[int]] = None) -> List[Task]:
    if seeds is None:
        seeds = [42, 99, 7]

    return [
        _make_overfitting_task(seeds[0]),
        _make_lr_schedule_task(seeds[1]),
        _make_data_poisoning_task(seeds[2]),
    ]
