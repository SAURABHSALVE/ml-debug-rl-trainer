"""
Task generators for the ML Experiment Debugger environment.
Each task returns a scenario dict with:
  - description   : what the agent sees at reset
  - data          : the raw data (logs, config, curves, gpu, class_metrics)
  - ground_truth  : what the correct answer is (hidden from agent)
"""

import random
from typing import Any, Dict


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _smooth(start: float, end: float, epochs: int, noise: float = 0.0, rng=None) -> list:
    """Generate a smooth curve from start to end with optional noise."""
    if rng is None:
        rng = random
    step = (end - start) / epochs
    return [
        round(start + i * step + rng.uniform(-noise, noise), 4)
        for i in range(epochs)
    ]


# ─── Task 1: Overfitting (Easy) ────────────────────────────────────────────────

def generate_overfitting_task(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    epochs = 20

    train_loss = _smooth(2.1, 0.05, epochs, noise=0.02, rng=rng)
    val_loss = _smooth(2.0, 0.45, 10, noise=0.03, rng=rng) + _smooth(0.45, 1.85, 10, noise=0.04, rng=rng)
    train_acc = _smooth(0.45, 0.99, epochs, noise=0.01, rng=rng)
    val_acc = _smooth(0.44, 0.81, 10, noise=0.02, rng=rng) + _smooth(0.81, 0.61, 10, noise=0.03, rng=rng)

    logs = []
    for i in range(epochs):
        logs.append(
            f"Epoch {i+1:02d} | train_loss={train_loss[i]:.4f} "
            f"train_acc={train_acc[i]:.4f} | "
            f"val_loss={val_loss[i]:.4f} val_acc={val_acc[i]:.4f}"
        )

    config = {
        "model": "ResNet50",
        "dataset": "CIFAR10_subset_5000",
        "epochs": 20,
        "batch_size": 32,
        "optimizer": "Adam",
        "lr": 0.001,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "data_augmentation": False,
        "early_stopping": False,
    }

    gpu_metrics = {
        "memory_mb": [rng.randint(3800, 4200) for _ in range(epochs)],
        "util_pct": [rng.randint(85, 99) for _ in range(epochs)],
    }

    return {
        "task_id": f"overfitting_{seed}",
        "difficulty": "easy",
        "description": (
            "A ResNet-50 was trained for 20 epochs on a small image classification dataset. "
            "Training finished but the deployed model performs poorly on new data. "
            "Investigate what went wrong and prescribe a fix."
        ),
        "data": {
            "logs": logs,
            "config": config,
            "loss_curve": {"train": train_loss, "val": val_loss},
            "acc_curve": {"train": train_acc, "val": val_acc},
            "gpu_metrics": gpu_metrics,
            "class_metrics": {i: round(rng.uniform(0.58, 0.68), 3) for i in range(10)},
        },
        "ground_truth": {
            "bug_type": "overfitting",
            "root_cause": "No regularization — dropout=0, weight_decay=0, no augmentation",
            "affected_config_keys": ["dropout", "weight_decay", "data_augmentation"],
            "valid_fix_types": ["config_change"],
            "valid_fix_keywords": ["dropout", "weight_decay", "augmentation", "early_stopping"],
            "diagnosis_keywords": ["overfit", "overfitting", "regularization", "val loss", "diverge"],
        },
    }


# ─── Task 2: LR Explosion (Medium) ─────────────────────────────────────────────

def generate_lr_explosion_task(seed: int = 99) -> Dict[str, Any]:
    rng = random.Random(seed)
    stable_epochs = 5
    unstable_epochs = 10

    train_loss_stable = _smooth(2.3, 1.8, stable_epochs, noise=0.03, rng=rng)
    train_loss_explode = [
        round(1.8 * (1.6 ** i) + rng.uniform(0, 0.5), 4)
        for i in range(unstable_epochs)
    ]
    train_loss_explode[-1] = float("nan")
    train_loss = train_loss_stable + train_loss_explode

    val_loss = _smooth(2.4, 1.9, stable_epochs, noise=0.04, rng=rng) + [
        round(min(99.9, 1.9 * (1.7 ** i)), 4) for i in range(unstable_epochs)
    ]

    grad_norms = [round(rng.uniform(0.8, 1.5), 3) for _ in range(stable_epochs)] + [
        round(1.5 * (2.1 ** i), 3) for i in range(unstable_epochs)
    ]

    logs = []
    for i in range(stable_epochs + unstable_epochs):
        loss_val = train_loss[i]
        loss_str = "nan" if loss_val != loss_val else f"{loss_val:.4f}"
        logs.append(
            f"Epoch {i+1:02d} | train_loss={loss_str} | "
            f"grad_norm={grad_norms[i]:.3f} | lr={0.5:.4f}"
        )

    config = {
        "model": "TransformerSmall",
        "dataset": "IMDB_sentiment",
        "epochs": 15,
        "batch_size": 64,
        "optimizer": "SGD",
        "lr": 0.5,
        "momentum": 0.9,
        "lr_scheduler": None,
        "gradient_clipping": None,
        "dropout": 0.1,
        "weight_decay": 1e-4,
    }

    gpu_metrics = {
        "memory_mb": [rng.randint(6000, 7000) for _ in range(stable_epochs + unstable_epochs)],
        "util_pct": [99 if i > stable_epochs else rng.randint(75, 90) for i in range(stable_epochs + unstable_epochs)],
    }

    return {
        "task_id": f"lr_explosion_{seed}",
        "difficulty": "medium",
        "description": (
            "A Transformer model was trained on sentiment classification. "
            "Loss was stable for the first 5 epochs then suddenly exploded and "
            "produced NaN. Training was automatically stopped. "
            "Investigate the root cause and prescribe an exact fix with values."
        ),
        "data": {
            "logs": logs,
            "config": config,
            "loss_curve": {"train": [x if x == x else 99.9 for x in train_loss], "val": val_loss},
            "grad_norms": grad_norms,
            "gpu_metrics": gpu_metrics,
            "class_metrics": {0: 0.52, 1: 0.49},
        },
        "ground_truth": {
            "bug_type": "learning_rate_explosion",
            "root_cause": "Learning rate 0.5 is far too high for SGD on transformer — causes gradient explosion",
            "affected_config_keys": ["lr"],
            "valid_fix_types": ["config_change"],
            "valid_fix_keywords": ["learning rate", "lr", "gradient clipping", "scheduler"],
            "valid_lr_range": (1e-5, 1e-2),
            "diagnosis_keywords": [
                "learning rate", "lr", "gradient", "explod", "nan",
                "instab", "too high", "diverge"
            ],
        },
    }


# ─── Task 3: Silent Data Poisoning (Hard) ──────────────────────────────────────

def generate_poisoning_task(seed: int = 7) -> Dict[str, Any]:
    rng = random.Random(seed)
    epochs = 25
    poisoned_class = rng.randint(0, 4)

    train_loss = _smooth(2.4, 0.28, epochs, noise=0.015, rng=rng)
    val_loss = _smooth(2.3, 0.41, epochs, noise=0.02, rng=rng)
    train_acc = _smooth(0.42, 0.93, epochs, noise=0.01, rng=rng)
    val_acc = _smooth(0.40, 0.84, epochs, noise=0.015, rng=rng)

    logs = []
    for i in range(epochs):
        line = (
            f"Epoch {i+1:02d} | train_loss={train_loss[i]:.4f} "
            f"train_acc={train_acc[i]:.4f} | "
            f"val_loss={val_loss[i]:.4f} val_acc={val_acc[i]:.4f}"
        )
        if i == 13:
            line += f" | WARNING: label_consistency_check class_{poisoned_class} score=0.61"
        logs.append(line)

    config = {
        "model": "EfficientNetB0",
        "dataset": "manufacturing_defects_5class",
        "epochs": 25,
        "batch_size": 32,
        "optimizer": "AdamW",
        "lr": 0.0003,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "data_augmentation": True,
        "label_smoothing": 0.0,
    }

    class_metrics = {}
    for c in range(5):
        if c == poisoned_class:
            class_metrics[c] = {
                "accuracy": round(rng.uniform(0.28, 0.42), 3),
                "f1": round(rng.uniform(0.22, 0.38), 3),
                "support": rng.randint(180, 220),
                "note": f"{rng.randint(15,25)}% of labels may be corrupted",
            }
        else:
            class_metrics[c] = {
                "accuracy": round(rng.uniform(0.87, 0.95), 3),
                "f1": round(rng.uniform(0.85, 0.93), 3),
                "support": rng.randint(180, 220),
                "note": "healthy",
            }

    gpu_metrics = {
        "memory_mb": [rng.randint(4800, 5200) for _ in range(epochs)],
        "util_pct": [rng.randint(80, 95) for _ in range(epochs)],
    }

    return {
        "task_id": f"data_poisoning_{seed}",
        "difficulty": "hard",
        "description": (
            "An EfficientNet-B0 was trained on a 5-class manufacturing defect dataset. "
            "Overall accuracy looks acceptable at 84% but production clients are "
            "complaining that the model gives wrong results on their data. "
            "Investigate deeply — the problem may not be obvious from top-level metrics."
        ),
        "data": {
            "logs": logs,
            "config": config,
            "loss_curve": {"train": train_loss, "val": val_loss},
            "acc_curve": {"train": train_acc, "val": val_acc},
            "gpu_metrics": gpu_metrics,
            "class_metrics": class_metrics,
        },
        "ground_truth": {
            "bug_type": "silent_data_poisoning",
            "root_cause": f"15-25% of labels in class_{poisoned_class} are corrupted",
            "poisoned_class": poisoned_class,
            "affected_config_keys": [],
            "valid_fix_types": ["data_fix"],
            "valid_fix_keywords": ["label", "corrupt", "annotate", "clean", "audit", "reannotate"],
            "diagnosis_keywords": [
                "poison", "corrupt", "label", "class", "per-class",
                "data", "annotation", f"class_{poisoned_class}", str(poisoned_class)
            ],
        },
    }