"""Tests for MLOps pipeline: data versioning, model monitoring, retraining."""

from pathlib import Path

import pandas as pd
import pytest

from src.main import (
    DataVersioner,
    ModelMonitor,
    ModelRegistry,
    generate_synthetic_data,
    train_model,
    _load_config,
)


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_config_returns_dict(self) -> None:
        """Config loader returns a dictionary."""
        config = _load_config("config.yaml")
        assert isinstance(config, dict)
        assert "paths" in config or "logging" in config


class TestDataVersioner:
    """Tests for DataVersioner."""

    def test_register_and_list_versions(self, tmp_path: Path) -> None:
        """Registering data creates version and list_versions returns it."""
        v = DataVersioner(str(tmp_path))
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        vid = v.register(df)
        assert isinstance(vid, str)
        assert len(vid) > 0
        versions = v.list_versions()
        assert vid in versions

    def test_register_and_load(self, tmp_path: Path) -> None:
        """Loading a version returns same data and metadata."""
        v = DataVersioner(str(tmp_path))
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        vid = v.register(df)
        loaded, meta = v.load(vid)
        pd.testing.assert_frame_equal(loaded, df)
        assert meta["rows"] == 2
        assert "x" in meta["columns"]

    def test_same_content_same_version(self, tmp_path: Path) -> None:
        """Same DataFrame content yields same version id."""
        v = DataVersioner(str(tmp_path))
        df = pd.DataFrame({"a": [1, 2]})
        v1 = v.register(df)
        v2 = v.register(df.copy())
        assert v1 == v2

    def test_get_latest_version(self, tmp_path: Path) -> None:
        """get_latest_version returns None when empty, else latest id."""
        v = DataVersioner(str(tmp_path))
        assert v.get_latest_version() is None
        df = pd.DataFrame({"a": [1]})
        vid = v.register(df)
        assert v.get_latest_version() == vid


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Saving and loading round-trips model and metadata."""
        from sklearn.linear_model import LogisticRegression

        r = ModelRegistry(str(tmp_path))
        model = LogisticRegression().fit([[1], [2], [3]], [0, 1, 0])
        r.save(model, "v1", metadata={"metric": 0.9})
        loaded_model, meta = r.load("v1")
        assert meta["version_id"] == "v1"
        assert "metric" in meta
        assert loaded_model.predict([[1.5]])[0] in (0, 1)

    def test_list_versions_and_latest(self, tmp_path: Path) -> None:
        """list_versions and get_latest_version behave correctly."""
        from sklearn.linear_model import LogisticRegression

        r = ModelRegistry(str(tmp_path))
        model = LogisticRegression().fit([[1], [2]], [0, 1])
        r.save(model, "v1")
        r.save(model, "v2")
        versions = r.list_versions()
        assert "v1" in versions and "v2" in versions
        assert r.get_latest_version() is not None


class TestModelMonitor:
    """Tests for ModelMonitor."""

    def test_record_and_get_latest_accuracy(self, tmp_path: Path) -> None:
        """Recording metrics and get_latest_accuracy returns last value."""
        m = ModelMonitor(str(tmp_path), accuracy_min_threshold=0.5)
        assert m.get_latest_accuracy() is None
        m.record_metrics(0.85, data_version="d1", model_version="m1")
        assert m.get_latest_accuracy() == 0.85
        m.record_metrics(0.90)
        assert m.get_latest_accuracy() == 0.90

    def test_should_retrain_no_model(self, tmp_path: Path) -> None:
        """should_retrain returns True when no_model_yet=True."""
        m = ModelMonitor(str(tmp_path))
        assert m.should_retrain(no_model_yet=True) is True

    def test_should_retrain_low_accuracy(self, tmp_path: Path) -> None:
        """should_retrain returns True when accuracy below threshold."""
        m = ModelMonitor(str(tmp_path), accuracy_min_threshold=0.80)
        assert m.should_retrain(current_accuracy=0.70) is True
        assert m.should_retrain(current_accuracy=0.90) is False

    def test_drift_reference_and_compute_drift(self, tmp_path: Path) -> None:
        """update_reference_distribution and compute_drift work together."""
        m = ModelMonitor(str(tmp_path), drift_threshold=0.2)
        m.update_reference_distribution({"f0": 0.0, "f1": 1.0})
        drift_same = m.compute_drift({"f0": 0.0, "f1": 1.0})
        assert drift_same == 0.0
        drift_diff = m.compute_drift({"f0": 0.5, "f1": 2.0})
        assert drift_diff > 0


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_shapes_and_columns(self) -> None:
        """Output has expected shape and target column."""
        df = generate_synthetic_data(n_samples=100, n_features=5, random_state=42)
        assert len(df) == 100
        assert df.shape[1] == 6
        assert "target" in df.columns
        assert set(df["target"].unique()).issubset({0, 1})


class TestTrainModel:
    """Tests for train_model."""

    def test_returns_fitted_model(self) -> None:
        """train_model returns a fitted classifier."""
        df = generate_synthetic_data(n_samples=80, n_features=4, random_state=42)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols]
        y = df["target"]
        config = {"training": {"model_type": "logistic", "random_state": 42}}
        model = train_model(X, y, config)
        assert hasattr(model, "predict")
        preds = model.predict(X)
        assert len(preds) == len(X)


class TestRunPipeline:
    """Integration tests for run_pipeline."""

    def test_run_pipeline_returns_summary(self, tmp_path: Path) -> None:
        """run_pipeline completes and returns summary with expected keys."""
        import yaml

        from src.main import run_pipeline

        config = {
            "logging": {"level": "INFO"},
            "paths": {
                "data_versions": str(tmp_path / "data_versions"),
                "model_registry": str(tmp_path / "model_registry"),
                "monitoring": str(tmp_path / "monitoring"),
            },
            "data_versioning": {"hash_algorithm": "sha256"},
            "monitoring": {
                "accuracy_min_threshold": 0.70,
                "drift_threshold": 0.15,
                "drift_min_samples": 50,
            },
            "training": {
                "model_type": "logistic",
                "random_state": 42,
                "test_size": 0.2,
                "max_iter": 500,
                "C": 1.0,
            },
            "pipeline": {"evaluate_after_retrain": True},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        summary = run_pipeline(str(config_file))
        assert "data_version" in summary
        assert "model_version" in summary
        assert "retrained" in summary
        assert "accuracy" in summary
        assert 0 <= summary["accuracy"] <= 1.0
