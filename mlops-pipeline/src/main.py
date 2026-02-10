"""End-to-end ML system with MLOps pipeline.

Implements data versioning (hash-based versions with metadata), model
monitoring (metrics history and drift detection), and automated retraining
triggered by performance degradation or drift.
"""

import hashlib
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

load_dotenv()

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration from file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging with optional file handler.

    Args:
        level: Log level name (e.g. INFO, DEBUG).
        log_file: Optional path to log file.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1024 * 1024, backupCount=3
        )
        fh.setLevel(log_level)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(fh)


class DataVersioner:
    """Data versioning with content-addressed storage and metadata."""

    def __init__(
        self,
        base_dir: str,
        hash_algorithm: str = "sha256",
    ) -> None:
        """Initialize data versioner.

        Args:
            base_dir: Root directory for versioned data.
            hash_algorithm: Hash algorithm for content addressing.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.hash_algorithm = hash_algorithm
        self._index_file = self.base_dir / "versions.json"

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute content hash of DataFrame (structure and values)."""
        content = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        h = hashlib.new(self.hash_algorithm)
        h.update(content)
        return h.hexdigest()

    def _metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build metadata for a dataset version."""
        numeric = df.select_dtypes(include=[np.number])
        means = numeric.mean().to_dict() if not numeric.empty else {}
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "feature_means": {k: float(v) for k, v in means.items()},
            "created_at": datetime.now().isoformat(),
        }

    def register(self, df: pd.DataFrame, version_id: Optional[str] = None) -> str:
        """Register a dataset version; use content hash as version if not provided.

        Args:
            df: Dataset to version.
            version_id: Optional explicit version id (default: content hash).

        Returns:
            Version identifier (e.g. content hash).
        """
        version_id = version_id or self._compute_hash(df)
        version_dir = self.base_dir / f"v_{version_id}"
        version_dir.mkdir(parents=True, exist_ok=True)
        meta = self._metadata(df)
        meta["version_id"] = version_id
        data_path = version_dir / "data.parquet"
        df.to_parquet(data_path, index=False)
        with open(version_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        index = self._read_index()
        if version_id not in index:
            index[version_id] = meta["created_at"]
            self._write_index(index)
        logger.info("Registered data version %s (%d rows)", version_id[:12], len(df))
        return version_id

    def _read_index(self) -> Dict[str, str]:
        """Read versions index."""
        if not self._index_file.exists():
            return {}
        with open(self._index_file) as f:
            return json.load(f)

    def _write_index(self, index: Dict[str, str]) -> None:
        """Write versions index."""
        with open(self._index_file, "w") as f:
            json.dump(index, f, indent=2)

    def list_versions(self) -> List[str]:
        """List version ids, newest first by creation time."""
        index = self._read_index()
        items = [(v, t) for v, t in index.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in items]

    def load(self, version_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load dataset and metadata for a version.

        Args:
            version_id: Version identifier.

        Returns:
            (DataFrame, metadata dict).

        Raises:
            FileNotFoundError: If version does not exist.
        """
        version_dir = self.base_dir / f"v_{version_id}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Data version not found: {version_id}")
        df = pd.read_parquet(version_dir / "data.parquet")
        with open(version_dir / "meta.json") as f:
            meta = json.load(f)
        return df, meta

    def get_latest_version(self) -> Optional[str]:
        """Return latest version id or None if no versions."""
        versions = self.list_versions()
        return versions[0] if versions else None


class ModelRegistry:
    """Simple model registry with versioned saves and metadata."""

    def __init__(self, base_dir: str) -> None:
        """Initialize registry.

        Args:
            base_dir: Root directory for model artifacts.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.base_dir / "versions.json"

    def _read_index(self) -> Dict[str, str]:
        """Read version index (version_id -> created_at)."""
        if not self._index_file.exists():
            return {}
        with open(self._index_file) as f:
            return json.load(f)

    def _write_index(self, index: Dict[str, str]) -> None:
        """Write version index."""
        with open(self._index_file, "w") as f:
            json.dump(index, f, indent=2)

    def save(
        self,
        model: Any,
        version_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model and metadata under version.

        Args:
            model: Sklearn-style model (joblib-serializable).
            version_id: Version string (e.g. v1, or timestamp).
            metadata: Optional metadata to store.
        """
        version_dir = self.base_dir / f"v_{version_id}"
        version_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, version_dir / "model.joblib")
        meta = metadata or {}
        meta["version_id"] = version_id
        meta["created_at"] = datetime.now().isoformat()
        with open(version_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        index = self._read_index()
        index[version_id] = meta["created_at"]
        self._write_index(index)
        logger.info("Saved model version %s", version_id)

    def load(self, version_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model and metadata.

        Args:
            version_id: Version identifier.

        Returns:
            (model, metadata dict).

        Raises:
            FileNotFoundError: If version does not exist.
        """
        version_dir = self.base_dir / f"v_{version_id}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version not found: {version_id}")
        model = joblib.load(version_dir / "model.joblib")
        with open(version_dir / "meta.json") as f:
            meta = json.load(f)
        return model, meta

    def list_versions(self) -> List[str]:
        """List version ids, newest first."""
        index = self._read_index()
        items = [(v, t) for v, t in index.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in items]

    def get_latest_version(self) -> Optional[str]:
        """Return latest model version or None."""
        versions = self.list_versions()
        return versions[0] if versions else None


class ModelMonitor:
    """Model monitoring: metrics history and drift-based retrain triggers."""

    def __init__(
        self,
        monitoring_dir: str,
        metrics_file: str = "metrics_history.json",
        drift_file: str = "drift_history.json",
        accuracy_min_threshold: float = 0.70,
        drift_threshold: float = 0.15,
        drift_min_samples: int = 50,
    ) -> None:
        """Initialize monitor.

        Args:
            monitoring_dir: Directory for monitoring artifacts.
            metrics_file: Filename for metrics history.
            drift_file: Filename for drift history.
            accuracy_min_threshold: Retrain if accuracy drops below this.
            drift_threshold: Retrain if relative feature drift exceeds this.
            drift_min_samples: Minimum samples to compute drift.
        """
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.monitoring_dir / metrics_file
        self.drift_path = self.monitoring_dir / drift_file
        self.accuracy_min = accuracy_min_threshold
        self.drift_threshold = drift_threshold
        self.drift_min_samples = drift_min_samples

    def _read_metrics(self) -> List[Dict[str, Any]]:
        """Read metrics history."""
        if not self.metrics_path.exists():
            return []
        with open(self.metrics_path) as f:
            return json.load(f)

    def _write_metrics(self, records: List[Dict[str, Any]]) -> None:
        """Write metrics history."""
        with open(self.metrics_path, "w") as f:
            json.dump(records, f, indent=2)

    def record_metrics(
        self,
        accuracy: float,
        data_version: Optional[str] = None,
        model_version: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a metrics record."""
        records = self._read_metrics()
        record = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "data_version": data_version,
            "model_version": model_version,
            **(extra or {}),
        }
        records.append(record)
        self._write_metrics(records)
        logger.info(
            "Recorded metrics: accuracy=%.4f, data=%s, model=%s",
            accuracy,
            data_version,
            model_version,
        )

    def get_latest_accuracy(self) -> Optional[float]:
        """Return most recent recorded accuracy or None."""
        records = self._read_metrics()
        if not records:
            return None
        return float(records[-1]["accuracy"])

    def _read_drift_ref(self) -> Optional[Dict[str, float]]:
        """Read reference feature means for drift (from last training)."""
        if not self.drift_path.exists():
            return None
        with open(self.drift_path) as f:
            data = json.load(f)
        return data.get("reference_means")

    def _write_drift_ref(self, reference_means: Dict[str, float]) -> None:
        """Write reference feature means and timestamp."""
        with open(self.drift_path, "w") as f:
            json.dump(
                {
                    "reference_means": reference_means,
                    "updated_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def update_reference_distribution(self, feature_means: Dict[str, float]) -> None:
        """Set reference feature distribution (e.g. from training data)."""
        self._write_drift_ref(feature_means)

    def compute_drift(self, feature_means: Dict[str, float]) -> float:
        """Compute relative L1 drift vs reference. Returns 0 if no reference.

        Per-feature relative change is capped at 10.0 to avoid huge values
        when reference mean is near zero.
        """
        ref = self._read_drift_ref()
        if not ref:
            return 0.0
        common = [k for k in feature_means if k in ref]
        if not common:
            return 0.0
        total = 0.0
        for k in common:
            r = ref[k]
            denom = max(abs(r), 1e-6)
            rel = abs(feature_means[k] - r) / denom
            total += min(rel, 10.0)
        return total / len(common)

    def should_retrain(
        self,
        current_accuracy: Optional[float] = None,
        current_feature_means: Optional[Dict[str, float]] = None,
        no_model_yet: bool = False,
    ) -> bool:
        """Determine if retraining should be triggered.

        Args:
            current_accuracy: Latest evaluation accuracy (optional).
            current_feature_means: Feature means of current eval data (optional).
            no_model_yet: True if no model is deployed yet.

        Returns:
            True if retrain is recommended.
        """
        if no_model_yet:
            return True
        if current_accuracy is not None and current_accuracy < self.accuracy_min:
            logger.warning(
                "Retrain triggered: accuracy %.4f below threshold %.4f",
                current_accuracy,
                self.accuracy_min,
            )
            return True
        if current_feature_means and self.drift_min_samples > 0:
            drift = self.compute_drift(current_feature_means)
            if drift >= self.drift_threshold:
                logger.warning(
                    "Retrain triggered: drift %.4f above threshold %.4f",
                    drift,
                    self.drift_threshold,
                )
                return True
        return False


def generate_synthetic_data(
    n_samples: int = 500,
    n_features: int = 10,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic classification dataset for pipeline demo.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with features and target column 'target'.
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    coef = rng.standard_normal(n_features)
    logit = X @ coef + rng.standard_normal(n_samples) * 0.5
    y = (logit > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> Any:
    """Train a classifier from config.

    Args:
        X_train: Training features.
        y_train: Training labels.
        config: Training and model config.

    Returns:
        Fitted sklearn estimator.
    """
    train_cfg = config.get("training", {})
    model_type = train_cfg.get("model_type", "logistic")
    if model_type == "logistic":
        model = LogisticRegression(
            max_iter=train_cfg.get("max_iter", 500),
            C=train_cfg.get("C", 1.0),
            random_state=train_cfg.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model.fit(X_train, y_train)
    return model


def run_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Run end-to-end MLOps pipeline: data versioning, monitor check, retrain if needed, evaluate.

    Args:
        config_path: Path to YAML config.

    Returns:
        Summary dict with data_version, model_version, retrained, accuracy, etc.
    """
    config = _load_config(config_path)
    paths = config.get("paths", {})
    data_versions_dir = paths.get("data_versions", "data_versions")
    model_registry_dir = paths.get("model_registry", "model_registry")
    monitoring_dir = paths.get("monitoring", "monitoring")
    mon_cfg = config.get("monitoring", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data_versioning", {})

    data_versioner = DataVersioner(
        data_versions_dir,
        hash_algorithm=data_cfg.get("hash_algorithm", "sha256"),
    )
    model_registry = ModelRegistry(model_registry_dir)
    monitor = ModelMonitor(
        monitoring_dir,
        metrics_file=mon_cfg.get("metrics_history_file", "metrics_history.json"),
        drift_file=mon_cfg.get("drift_history_file", "drift_history.json"),
        accuracy_min_threshold=mon_cfg.get("accuracy_min_threshold", 0.70),
        drift_threshold=mon_cfg.get("drift_threshold", 0.15),
        drift_min_samples=mon_cfg.get("drift_min_samples", 50),
    )

    # Ingest and version data (demo: generate synthetic)
    df = generate_synthetic_data(
        n_samples=600,
        n_features=10,
        random_state=train_cfg.get("random_state", 42),
    )
    data_version = data_versioner.register(df)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg.get("test_size", 0.2), random_state=train_cfg.get("random_state")
    )

    # Reference distribution for drift (from training data)
    numeric = X_train.select_dtypes(include=[np.number])
    ref_means = numeric.mean().to_dict()
    ref_means = {str(k): float(v) for k, v in ref_means.items()}

    current_model_version = model_registry.get_latest_version()
    no_model = current_model_version is None
    current_accuracy = None
    if not no_model:
        model, _ = model_registry.load(current_model_version)
        preds = model.predict(X_test)
        current_accuracy = float(accuracy_score(y_test, preds))
    test_means = {c: float(X_test[c].mean()) for c in feature_cols}

    should_retrain = monitor.should_retrain(
        current_accuracy=current_accuracy,
        current_feature_means=test_means,
        no_model_yet=no_model,
    )

    retrained = False
    model_version = current_model_version
    if should_retrain:
        model = train_model(X_train, y_train, config)
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_registry.save(
            model,
            model_version,
            metadata={"data_version": data_version, "accuracy": None},
        )
        monitor.update_reference_distribution(ref_means)
        retrained = True

    if retrained or no_model:
        model, _ = model_registry.load(model_version)
        preds = model.predict(X_test)
        final_accuracy = float(accuracy_score(y_test, preds))
    else:
        final_accuracy = current_accuracy or 0.0

    if config.get("pipeline", {}).get("evaluate_after_retrain", True):
        monitor.record_metrics(
            final_accuracy,
            data_version=data_version,
            model_version=model_version,
        )

    summary = {
        "data_version": data_version,
        "model_version": model_version,
        "retrained": retrained,
        "accuracy": final_accuracy,
    }
    logger.info(
        "Pipeline complete: data=%s, model=%s, retrained=%s, accuracy=%.4f",
        data_version[:12],
        model_version,
        retrained,
        final_accuracy,
    )
    return summary


def main() -> None:
    """Entry point: parse CLI, setup logging, run pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="MLOps pipeline runner")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    log_cfg = config.get("logging", {})
    _setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
