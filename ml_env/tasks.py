"""
Task generators for the ML Experiment Debugger environment.
Each task returns a scenario dict with:
  - description   : what the agent sees at reset
  - data          : the raw data (logs, config, curves, gpu, class_metrics)
  - ground_truth  : what the correct answer is (hidden from agent)

Task catalogue (5 tasks, 3 selected per episode):
  1. overfitting_detection  (easy)
  2. lr_explosion           (medium)
  3. silent_data_poisoning  (hard)
  4. class_imbalance        (medium)
  5. catastrophic_forgetting (hard)
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


# ─── Task 1: Data Leakage (Easy) ────────────────────────────────────────────────

def generate_data_leakage_task(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    epochs = 5

    train_loss = _smooth(0.05, 0.01, epochs, noise=0.005, rng=rng)
    val_loss = _smooth(0.06, 0.012, epochs, noise=0.005, rng=rng)
    train_acc = _smooth(0.98, 0.999, epochs, noise=0.001, rng=rng)
    val_acc = _smooth(0.98, 0.998, epochs, noise=0.001, rng=rng)

    logs = []
    for i in range(epochs):
        logs.append(
            f"Epoch {i+1:02d} | train_loss={train_loss[i]:.4f} "
            f"train_acc={train_acc[i]:.4f} | "
            f"val_loss={val_loss[i]:.4f} val_acc={val_acc[i]:.4f}"
        )

    config = {
        "model": "XGBoostClassifier",
        "dataset": "customer_churn_tabular",
        "features_used": ["age", "account_age", "monthly_bill", "latest_churn_flag", "support_tickets"],
        "epochs": 5,
        "optimizer": "default",
    }

    gpu_metrics = {
        "memory_mb": [rng.randint(100, 200) for _ in range(epochs)],
        "util_pct": [rng.randint(10, 20) for _ in range(epochs)],
    }

    return {
        "task_id": f"data_leakage_{seed}",
        "difficulty": "easy",
        "description": (
            "An XGBoost model was trained on tabular customer churn data. "
            "It achieved 99.8% precision instantly. But in a live A/B test, it performed no better than random guessing. "
            "Investigate what went wrong and prescribe a fix."
        ),
        "data": {
            "logs": logs,
            "config": config,
            "loss_curve": {"train": train_loss, "val": val_loss},
            "acc_curve": {"train": train_acc, "val": val_acc},
            "gpu_metrics": gpu_metrics,
            "class_metrics": {i: round(rng.uniform(0.98, 0.99), 3) for i in range(2)},
        },
        "ground_truth": {
            "difficulty": "easy",
            "bug_type": "data_leakage",
            "root_cause": "The feature 'latest_churn_flag' is the target variable leaked into the training features.",
            "affected_config_keys": ["features_used"],
            "valid_fix_types": ["data_fix", "config_change"],
            "valid_fix_keywords": ["leak", "leakage", "target", "remove", "latest_churn_flag", "exclude"],
            "diagnosis_keywords": ["leak", "data leak", "data leakage", "target in features", "perfect accuracy", "too good"],
        },
    }


# ─── Task 2: FP16 Underflow (Medium) ─────────────────────────────────────────────

def generate_fp16_underflow_task(seed: int = 99) -> Dict[str, Any]:
    rng = random.Random(seed)
    epochs = 15

    # Loss stalls immediately
    train_loss = [round(rng.uniform(2.30, 2.31), 4) for _ in range(epochs)]
    val_loss = [round(rng.uniform(2.30, 2.31), 4) for _ in range(epochs)]

    # Gradients underflow to exact 0.0
    grad_norms = [0.000 for _ in range(epochs)]

    logs = []
    for i in range(epochs):
        logs.append(
            f"Epoch {i+1:02d} | train_loss={train_loss[i]:.4f} | "
            f"grad_norm={grad_norms[i]:.3f} | lr={0.001:.4f}"
        )

    config = {
        "model": "Llama-3-8B-LoRA",
        "dataset": "custom_chat",
        "epochs": 15,
        "batch_size": 16,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "precision": "fp16",
        "grad_scaler": False,
        "gradient_clipping": 1.0,
    }

    gpu_metrics = {
        "memory_mb": [rng.randint(75000, 78000) for _ in range(epochs)],
        "util_pct": [rng.randint(95, 99) for _ in range(epochs)],
    }

    return {
        "task_id": f"fp16_underflow_{seed}",
        "difficulty": "medium",
        "description": (
            "A Llama-3-8B model was fine-tuned using LoRA. The model runs fast on GPU, "
            "but doesn't seem to be learning at all. Loss is completely stalled. "
            "Investigate the root cause in the metrics or config and prescribe a fix."
        ),
        "data": {
            "logs": logs,
            "config": config,
            "loss_curve": {"train": train_loss, "val": val_loss},
            "grad_norms": grad_norms,
            "gpu_metrics": gpu_metrics,
            "class_metrics": {0: 0.1, 1: 0.1},
        },
        "ground_truth": {
            "difficulty": "medium",
            "bug_type": "fp16_underflow",
            "root_cause": "Using fp16 precision without a gradient scaler causes gradients to underflow to zero.",
            "affected_config_keys": ["precision", "grad_scaler"],
            "valid_fix_types": ["config_change"],
            "valid_fix_keywords": ["grad_scaler", "gradient scaler", "bf16", "bfloat16", "amp", "mixed precision scale"],
            "diagnosis_keywords": [
                "underflow", "fp16", "gradient zero", "grad norm 0", "scaler",
                "grad_scaler", "precision"
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
            line += " | WARNING: anomaly_detector flagged inconsistency in 1 class — per-class audit recommended"
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
            "difficulty": "hard",
            "bug_type": "silent_data_poisoning",
            "root_cause": f"15-25% of labels in class_{poisoned_class} are corrupted",
            "poisoned_class": poisoned_class,
            "affected_config_keys": [],
            "valid_fix_types": ["data_fix"],
            "valid_fix_keywords": ["label", "corrupt", "annotate", "clean", "audit", "reannotate"],
            "diagnosis_keywords": [
                "poison", "corrupt", "label corrupt", "corrupted label",
                "mislabel", "annotation error", "bad label", "label noise",
                f"class_{poisoned_class}",
            ],
        },
    }


# ─── Task 4: Class Imbalance (Medium) ──────────────────────────────────────────

def generate_class_imbalance_task(seed: int = 55) -> Dict[str, Any]:
    rng = random.Random(seed)
    epochs = 20

    # Model learns to always predict majority class → high overall acc, zero minority acc
    train_loss = _smooth(2.2, 0.18, epochs, noise=0.01, rng=rng)
    val_loss   = _smooth(2.1, 0.20, epochs, noise=0.015, rng=rng)
    train_acc  = _smooth(0.40, 0.95, epochs, noise=0.01, rng=rng)
    val_acc    = _smooth(0.38, 0.93, epochs, noise=0.015, rng=rng)

    minority_classes = [1, 2, 3]
    majority_class = 0

    logs = []
    for i in range(epochs):
        logs.append(
            f"Epoch {i+1:02d} | train_loss={train_loss[i]:.4f} "
            f"train_acc={train_acc[i]:.4f} | "
            f"val_loss={val_loss[i]:.4f} val_acc={val_acc[i]:.4f}"
        )

    config = {
        "model": "MobileNetV2",
        "dataset": "medical_xray_4class",
        "epochs": 20,
        "batch_size": 32,
        "optimizer": "Adam",
        "lr": 0.001,
        "class_weights": None,
        "sampler": "default",
        "loss_function": "CrossEntropyLoss",
    }

    class_dist = {0: 9500, 1: 167, 2: 198, 3: 135}  # Severe imbalance

    class_metrics = {}
    for c in range(4):
        if c == majority_class:
            class_metrics[c] = {
                "accuracy": round(rng.uniform(0.96, 0.99), 3),
                "f1": round(rng.uniform(0.97, 0.99), 3),
                "support": class_dist[c],
                "recall": round(rng.uniform(0.97, 0.99), 3),
            }
        else:
            class_metrics[c] = {
                "accuracy": round(rng.uniform(0.01, 0.08), 3),
                "f1": round(rng.uniform(0.01, 0.06), 3),
                "support": class_dist[c],
                "recall": round(rng.uniform(0.00, 0.05), 3),
            }

    return {
        "task_id": f"class_imbalance_{seed}",
        "difficulty": "medium",
        "description": (
            "A MobileNetV2 was trained on a 4-class medical X-ray classification dataset. "
            "Overall validation accuracy reached 93%, which looks excellent. "
            "However, clinicians report that the model almost never catches disease in minority cases. "
            "Investigate the root cause — the top-level metrics may be misleading."
        ),
        "data": {
            "logs": logs,
            "config": config,
            "loss_curve": {"train": train_loss, "val": val_loss},
            "acc_curve": {"train": train_acc, "val": val_acc},
            "class_metrics": class_metrics,
            "class_distribution": class_dist,
            "gpu_metrics": {
                "memory_mb": [rng.randint(3200, 3800) for _ in range(epochs)],
                "util_pct": [rng.randint(70, 90) for _ in range(epochs)],
            },
        },
        "ground_truth": {
            "difficulty": "medium",
            "bug_type": "class_imbalance",
            "root_cause": "Severe class imbalance — 95% of data is class 0; model predicts majority class",
            "affected_config_keys": ["class_weights", "sampler", "loss_function"],
            "valid_fix_types": ["config_change", "data_fix"],
            "valid_fix_keywords": [
                "class weight", "weighted loss", "oversample", "undersample",
                "smote", "balanced", "focal loss", "imbalance",
            ],
            "diagnosis_keywords": [
                "imbalance", "class imbalance", "majority", "minority",
                "skew", "overrepresent", "weighted", "dominant class",
            ],
        },
    }


# ─── Task 5: Catastrophic Forgetting (Hard) ────────────────────────────────────

def generate_forgetting_task(seed: int = 33) -> Dict[str, Any]:
    rng = random.Random(seed)
    pretrain_epochs = 10
    finetune_epochs = 15

    # Phase 1: Pretraining — good performance on original task
    pretrain_loss = _smooth(2.1, 0.35, pretrain_epochs, noise=0.02, rng=rng)
    pretrain_acc  = _smooth(0.40, 0.92, pretrain_epochs, noise=0.01, rng=rng)

    # Phase 2: Fine-tuning — new task accuracy rises, original task accuracy collapses
    finetune_loss_new = _smooth(1.9, 0.22, finetune_epochs, noise=0.015, rng=rng)
    finetune_acc_new  = _smooth(0.35, 0.91, finetune_epochs, noise=0.01, rng=rng)

    logs = []
    for i in range(pretrain_epochs):
        logs.append(
            f"[Pretrain] Epoch {i+1:02d} | loss={pretrain_loss[i]:.4f} "
            f"acc={pretrain_acc[i]:.4f} | original_task_acc={pretrain_acc[i]:.4f}"
        )
    for i in range(finetune_epochs):
        # Original task accuracy drops catastrophically during fine-tuning
        original_acc = max(0.08, round(0.92 - (i * 0.058) + rng.uniform(-0.02, 0.02), 3))
        logs.append(
            f"[Finetune] Epoch {i+1:02d} | loss={finetune_loss_new[i]:.4f} "
            f"new_task_acc={finetune_acc_new[i]:.4f} | original_task_acc={original_acc:.4f}"
        )

    config = {
        "model": "ResNet50_pretrained_ImageNet",
        "original_dataset": "ImageNet_subset",
        "finetune_dataset": "satellite_imagery_binary",
        "finetune_epochs": 15,
        "optimizer": "Adam",
        "lr": 0.01,
        "freeze_backbone": False,
        "ewc_lambda": None,
        "replay_buffer": None,
        "lr_schedule": None,
    }

    # By end of fine-tuning original task accuracy is 8-15%
    final_original_acc = round(rng.uniform(0.08, 0.15), 3)
    class_metrics = {
        "original_task_classes": {
            "avg_accuracy": final_original_acc,
            "note": f"Collapsed from 92% to {final_original_acc*100:.0f}% after fine-tuning"
        },
        "new_task_classes": {
            "avg_accuracy": round(rng.uniform(0.88, 0.93), 3),
            "note": "New task performing well"
        }
    }

    return {
        "task_id": f"catastrophic_forgetting_{seed}",
        "difficulty": "hard",
        "description": (
            "A ResNet-50 pretrained on ImageNet was fine-tuned on a satellite imagery binary classification task. "
            "The new task accuracy reached 91%, but after deployment, the model completely fails on the original "
            "ImageNet classification tasks it was built for. The original-task accuracy dropped from 92% to under 15%. "
            "Investigate why and propose a fix that preserves both capabilities."
        ),
        "data": {
            "logs": logs,
            "config": config,
            "loss_curve": {
                "train": pretrain_loss + finetune_loss_new,
                # Val curve: original-task loss stable during pretrain, then rises as forgetting occurs
                "val_original_task": _smooth(2.1, 0.38, pretrain_epochs, noise=0.02, rng=rng)
                              + _smooth(0.38, 3.85, finetune_epochs, noise=0.08, rng=rng),
                "val_new_task": [None] * pretrain_epochs
                              + _smooth(1.9, 0.25, finetune_epochs, noise=0.02, rng=rng),
                "note": "val_original_task rises sharply during fine-tuning — catastrophic forgetting signal",
            },
            "class_metrics": class_metrics,
            "gpu_metrics": {
                "memory_mb": [rng.randint(5000, 6000) for _ in range(pretrain_epochs + finetune_epochs)],
                "util_pct": [rng.randint(80, 95) for _ in range(pretrain_epochs + finetune_epochs)],
            },
        },
        "ground_truth": {
            "difficulty": "hard",
            "bug_type": "catastrophic_forgetting",
            "root_cause": (
                "No regularization during fine-tuning — model overwrites original weights. "
                "freeze_backbone=False with high lr=0.01 destroys pretrained representations."
            ),
            "affected_config_keys": ["freeze_backbone", "lr", "ewc_lambda", "replay_buffer"],
            "valid_fix_types": ["config_change", "architecture_change"],
            "valid_fix_keywords": [
                "freeze", "ewc", "elastic weight", "replay", "distillation",
                "lower lr", "learning rate", "regularization", "backbone",
            ],
            "diagnosis_keywords": [
                "catastrophic", "forget", "forgetting", "overwrite",
                "original task", "pretrain", "fine-tun", "collapse",
                "backbone", "representation",
            ],
        },
    }


# ─── Task 6: NaN from Bad Initialization (Easy) ───────────────────────────────

def generate_nan_init_task(seed: int = 77) -> Dict[str, Any]:
    """
    NaN loss from day 1 due to weight initialization std being 500× too large.
    Easy: the loss curve and logs make it instantly obvious if the agent calls
    fetch_loss_curve or fetch_logs. Agent must read the config to name init_std as the fix.
    """
    rng = random.Random(seed)
    epochs = 8  # Training stopped early when NaN detected

    # Loss is NaN from epoch 1
    train_loss = [float("nan")] * epochs
    val_loss   = [float("nan")] * epochs

    logs = []
    for i in range(epochs):
        norm = round(rng.uniform(1e5, 1e7), 1) if i < 3 else float("nan")
        norm_str = f"{norm:.1f}" if norm == norm else "nan"
        logs.append(
            f"Epoch {i+1:02d} | train_loss=nan | val_loss=nan | "
            f"grad_norm={norm_str} | lr=0.0001"
        )
    logs[0] += " | WARNING: loss=nan on first forward pass — check model initialization"

    config = {
        "model": "BERT_small_custom",
        "dataset": "news_classification_10class",
        "epochs": 20,
        "batch_size": 16,
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.01,
        "init_std": 10.0,          # BUG: should be ~0.02 for BERT-style models
        "init_mean": 0.0,
        "layer_norm_eps": 1e-12,
    }

    gpu_metrics = {
        "memory_mb": [rng.randint(4000, 4500) for _ in range(epochs)],
        "util_pct": [rng.randint(5, 15) for _ in range(epochs)],  # Low util — training crashed fast
    }

    return {
        "task_id": f"nan_init_{seed}",
        "difficulty": "easy",
        "description": (
            "A custom BERT-small model was trained on a 10-class news classification dataset. "
            "Training was automatically stopped after 8 epochs because the loss was NaN from "
            "the very first forward pass. GPU utilization is near zero — the model isn't learning at all. "
            "Investigate the root cause and prescribe a specific config fix."
        ),
        "data": {
            "logs": logs,
            "config": config,
            "loss_curve": {
                "train": [99.9] * epochs,   # NaN replaced for API transport
                "val":   [99.9] * epochs,
                "note":  "train and val loss are NaN every epoch — catastrophic failure from epoch 1",
            },
            "gpu_metrics": gpu_metrics,
            "class_metrics": {i: 0.1 for i in range(10)},  # All ~random — model never learned
        },
        "ground_truth": {
            "difficulty": "easy",
            "bug_type": "bad_initialization",
            "root_cause": "Weight init_std=10.0 is ~500x too large for a BERT-style model (should be ~0.02), causing activations to overflow to NaN immediately",
            "affected_config_keys": ["init_std"],
            "valid_fix_types": ["config_change"],
            "valid_fix_keywords": [
                "init", "initialization", "init_std", "weight init",
                "std", "xavier", "kaiming", "normal init", "0.02",
            ],
            "diagnosis_keywords": [
                "init", "initialization", "weight init", "init_std",
                "nan", "overflow", "explod", "first epoch", "bad init",
                "std too large", "too high",
            ],
        },
    }