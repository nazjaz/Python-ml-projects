"""Knowledge distillation for model compression with teacher-student architecture.

This module implements knowledge distillation using only NumPy: a larger
teacher MLP and a smaller student MLP, softmax with temperature scaling
for soft labels, and a combined distillation loss (KL divergence toward
teacher soft labels plus optional cross-entropy with hard labels). The
teacher is trained first; then the student is trained to mimic the
teacher's soft outputs for compression.
"""

import argparse
import json
import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_EPS = 1e-8


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with optional temperature scaling."""
    x = x / max(temperature, _EPS)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x - x_max, -500, 500))
    return e / (np.sum(e, axis=axis, keepdims=True) + _EPS)


def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """KL(p || q) per element along axis; p and q are probability vectors."""
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=axis)


class MLP:
    """Two-layer MLP with ReLU for teacher or student."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        limit1 = np.sqrt(1.0 / max(1, in_dim))
        limit2 = np.sqrt(1.0 / max(1, hidden_dim))
        self.w1 = np.random.uniform(-limit1, limit1, (in_dim, hidden_dim)).astype(
            np.float32
        )
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.w2 = np.random.uniform(-limit2, limit2, (hidden_dim, out_dim)).astype(
            np.float32
        )
        self.b2 = np.zeros((out_dim,), dtype=np.float32)
        self._x: Optional[np.ndarray] = None
        self._h: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        self._h = h
        return h @ self.w2 + self.b2

    def backward(self, grad_out: np.ndarray, lr: float) -> None:
        if self._x is None or self._h is None:
            raise RuntimeError("Forward must be called before backward.")
        x, h = self._x, self._h
        batch = x.shape[0]
        d_h = grad_out @ self.w2.T
        d_z = d_h * (h > 0.0).astype(np.float32)
        self.w2 -= lr * (self._h.T @ grad_out) / float(batch)
        self.b2 -= lr * np.mean(grad_out, axis=0)
        self.w1 -= lr * (x.T @ d_z) / float(batch)
        self.b1 -= lr * np.mean(d_z, axis=0)


def train_teacher(
    teacher: MLP,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    random_seed: Optional[int] = None,
) -> List[float]:
    """Train teacher with cross-entropy; return list of mean losses per epoch."""
    n = train_x.shape[0]
    out_dim = train_y.max() + 1
    losses: List[float] = []
    rng = np.random.default_rng(random_seed)

    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            x = train_x[idx]
            y = train_y[idx]
            logits = teacher.forward(x)
            probs = softmax(logits, axis=1, temperature=1.0)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(y)), y.astype(int)] = 1.0
            grad = (probs - one_hot) / float(x.shape[0])
            teacher.backward(grad, lr)
            ce = -np.sum(one_hot * np.log(probs + _EPS)) / float(x.shape[0])
            epoch_loss += ce
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
        logger.info("Teacher epoch %d loss=%.4f", epoch, losses[-1])
    return losses


def train_student_with_distillation(
    teacher: MLP,
    student: MLP,
    train_x: np.ndarray,
    train_y: np.ndarray,
    temperature: float,
    alpha: float,
    epochs: int,
    lr: float,
    batch_size: int,
    random_seed: Optional[int] = None,
) -> List[float]:
    """Train student with distillation loss: alpha * KL(teacher_soft, student_soft) + (1-alpha) * CE(student, hard)."""
    n = train_x.shape[0]
    losses: List[float] = []
    rng = np.random.default_rng(random_seed)
    T = max(temperature, _EPS)

    for epoch in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            x = train_x[idx]
            y = train_y[idx]

            with np.errstate(over="ignore", invalid="ignore"):
                teacher_logits = teacher.forward(x)
                student_logits = student.forward(x)
            soft_teacher = softmax(teacher_logits, axis=1, temperature=T)
            soft_student = softmax(student_logits, axis=1, temperature=T)
            soft_student_ce = softmax(student_logits, axis=1, temperature=1.0)

            one_hot = np.zeros_like(soft_student_ce)
            one_hot[np.arange(len(y)), y.astype(int)] = 1.0

            kl = np.mean(kl_divergence(soft_teacher, soft_student, axis=1))
            ce = -np.mean(np.sum(one_hot * np.log(soft_student_ce + _EPS), axis=1))
            loss = alpha * kl + (1.0 - alpha) * ce
            epoch_loss += loss
            n_batches += 1

            grad_kl = (soft_student - soft_teacher) / T
            grad_ce = soft_student_ce - one_hot
            grad = alpha * grad_kl + (1.0 - alpha) * grad_ce
            student.backward(grad, lr)

        losses.append(epoch_loss / max(n_batches, 1))
        logger.info("Student epoch %d distillation loss=%.4f", epoch, losses[-1])
    return losses


def evaluate(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """Return (accuracy, mean cross-entropy)."""
    logits = model.forward(x)
    probs = softmax(logits, axis=1, temperature=1.0)
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred.astype(np.int64) == y.astype(np.int64))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y)), y.astype(int)] = 1.0
    ce = float(np.mean(-np.sum(one_hot * np.log(probs + _EPS), axis=1)))
    return float(acc), ce


def load_data(
    train_ratio: float = 0.8,
    max_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load digits or synthetic data; return train_x, train_y, val_x, val_y."""
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        data = load_digits()
        x = np.asarray(data.data, dtype=np.float32) / 16.0
        y = np.asarray(data.target, dtype=np.int64)
        if max_samples is not None:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(len(x), min(max_samples, len(x)), replace=False)
            x, y = x[idx], y[idx]
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=train_ratio, random_state=random_seed
        )
        return x_train, y_train, x_val, y_val
    except ImportError:
        rng = np.random.default_rng(random_seed or 42)
        n = max_samples or 500
        x = np.asarray(rng.standard_normal((n, 64)), dtype=np.float32) * 0.1
        y = np.asarray(rng.integers(0, 10, size=n), dtype=np.int64)
        split = int(n * train_ratio)
        return x[:split], y[:split], x[split:], y[split:]


@dataclass
class DistillationConfig:
    """Configuration for teacher training and distillation."""

    teacher_hidden: int = 128
    student_hidden: int = 32
    teacher_epochs: int = 10
    student_epochs: int = 10
    temperature: float = 4.0
    alpha: float = 0.7
    learning_rate: float = 0.01
    batch_size: int = 32
    train_ratio: float = 0.8
    max_samples: Optional[int] = None
    random_seed: Optional[int] = 0


def run_distillation(config: DistillationConfig) -> Dict:
    """Train teacher, then student with distillation; return metrics and sizes."""
    train_x, train_y, val_x, val_y = load_data(
        train_ratio=config.train_ratio,
        max_samples=config.max_samples,
        random_seed=config.random_seed,
    )
    in_dim = train_x.shape[1]
    out_dim = int(train_y.max()) + 1

    teacher = MLP(
        in_dim, config.teacher_hidden, out_dim, random_seed=config.random_seed
    )
    train_teacher(
        teacher,
        train_x,
        train_y,
        epochs=config.teacher_epochs,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        random_seed=config.random_seed,
    )
    teacher_acc, teacher_ce = evaluate(teacher, val_x, val_y)

    student = MLP(
        in_dim, config.student_hidden, out_dim,
        random_seed=(config.random_seed + 1000) if config.random_seed else None,
    )
    train_student_with_distillation(
        teacher,
        student,
        train_x,
        train_y,
        temperature=config.temperature,
        alpha=config.alpha,
        epochs=config.student_epochs,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        random_seed=config.random_seed,
    )
    student_acc, student_ce = evaluate(student, val_x, val_y)

    teacher_params = (
        teacher.w1.size + teacher.b1.size + teacher.w2.size + teacher.b2.size
    )
    student_params = (
        student.w1.size + student.b1.size + student.w2.size + student.b2.size
    )
    compression_ratio = teacher_params / max(student_params, 1)

    return {
        "teacher_val_accuracy": teacher_acc,
        "teacher_val_ce": teacher_ce,
        "student_val_accuracy": student_acc,
        "student_val_ce": student_ce,
        "teacher_params": int(teacher_params),
        "student_params": int(student_params),
        "compression_ratio": float(compression_ratio),
    }


def main() -> None:
    """Entry point: parse args, run distillation, print and optionally save results."""
    parser = argparse.ArgumentParser(
        description="Knowledge distillation with teacher-student and temperature"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config or Path(__file__).parent.parent / "config.yaml"
    config_dict: Dict = {}
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Config not found: %s", config_path)

    log_cfg = config_dict.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    log_file = log_cfg.get("file", "logs/app.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.setLevel(level)
    logger.addHandler(handler)

    d = config_dict.get("distillation", {})
    config = DistillationConfig(
        teacher_hidden=d.get("teacher_hidden", 128),
        student_hidden=d.get("student_hidden", 32),
        teacher_epochs=d.get("teacher_epochs", 10),
        student_epochs=d.get("student_epochs", 10),
        temperature=d.get("temperature", 4.0),
        alpha=d.get("alpha", 0.7),
        learning_rate=d.get("learning_rate", 0.01),
        batch_size=d.get("batch_size", 32),
        train_ratio=d.get("train_ratio", 0.8),
        max_samples=d.get("max_samples"),
        random_seed=d.get("random_seed", 0),
    )

    results = run_distillation(config)
    print("\nDistillation Results:")
    print("========================================")
    for k, v in results.items():
        if isinstance(v, float):
            print("  %s: %.4f" % (k, v))
        else:
            print("  %s: %s" % (k, v))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
