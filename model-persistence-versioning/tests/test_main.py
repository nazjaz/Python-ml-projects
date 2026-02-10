"""Unit tests for model persistence and versioning module."""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from src.main import (
    ModelMetadata,
    ModelPersistence,
    ModelPersistenceManager,
    ModelSerializer,
    ModelVersion,
)


class TestModelMetadata:
    """Test cases for ModelMetadata."""

    def test_initialization(self):
        """Test metadata initialization."""
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0.0",
        )

        assert metadata.model_name == "test_model"
        assert metadata.model_type == "RandomForest"
        assert metadata.version == "1.0.0"
        assert metadata.created_at is not None

    def test_initialization_with_custom_fields(self):
        """Test metadata initialization with custom fields."""
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0.0",
            accuracy=0.95,
            training_samples=1000,
        )

        assert metadata.metadata["accuracy"] == 0.95
        assert metadata.metadata["training_samples"] == 1000

    def test_to_dict(self):
        """Test metadata to dictionary conversion."""
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0.0",
            accuracy=0.95,
        )

        data = metadata.to_dict()
        assert data["model_name"] == "test_model"
        assert data["model_type"] == "RandomForest"
        assert data["version"] == "1.0.0"
        assert data["accuracy"] == 0.95
        assert "created_at" in data

    def test_from_dict(self):
        """Test metadata creation from dictionary."""
        data = {
            "model_name": "test_model",
            "model_type": "RandomForest",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00",
            "accuracy": 0.95,
        }

        metadata = ModelMetadata.from_dict(data)
        assert metadata.model_name == "test_model"
        assert metadata.model_type == "RandomForest"
        assert metadata.version == "1.0.0"
        assert metadata.metadata["accuracy"] == 0.95

    def test_update(self):
        """Test metadata update."""
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0.0",
        )

        original_updated_at = metadata.updated_at
        metadata.update(accuracy=0.95, f1_score=0.92)

        assert metadata.metadata["accuracy"] == 0.95
        assert metadata.metadata["f1_score"] == 0.92
        assert metadata.updated_at != original_updated_at


class TestModelSerializer:
    """Test cases for ModelSerializer."""

    def test_initialization_joblib(self):
        """Test serializer initialization with joblib."""
        serializer = ModelSerializer(serialization_format="joblib")
        assert serializer.serialization_format == "joblib"

    def test_initialization_pickle(self):
        """Test serializer initialization with pickle."""
        serializer = ModelSerializer(serialization_format="pickle")
        assert serializer.serialization_format == "pickle"

    def test_invalid_format(self):
        """Test error with invalid serialization format."""
        with pytest.raises(ValueError, match="Unsupported serialization format"):
            ModelSerializer(serialization_format="invalid")

    def test_serialize_joblib(self):
        """Test model serialization with joblib."""
        model = LinearRegression()
        model.fit(np.random.randn(100, 5), np.random.randn(100))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.joblib"
            serializer = ModelSerializer(serialization_format="joblib")
            serializer.serialize(model, filepath)

            assert filepath.exists()

    def test_serialize_pickle(self):
        """Test model serialization with pickle."""
        model = LinearRegression()
        model.fit(np.random.randn(100, 5), np.random.randn(100))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            serializer = ModelSerializer(serialization_format="pickle")
            serializer.serialize(model, filepath, compress=False)

            assert filepath.exists()

    def test_deserialize_joblib(self):
        """Test model deserialization with joblib."""
        model = LinearRegression()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.joblib"
            serializer = ModelSerializer(serialization_format="joblib")
            serializer.serialize(model, filepath)

            loaded_model = serializer.deserialize(filepath)
            assert isinstance(loaded_model, LinearRegression)
            np.testing.assert_array_almost_equal(
                model.predict(X), loaded_model.predict(X)
            )

    def test_deserialize_pickle(self):
        """Test model deserialization with pickle."""
        model = LinearRegression()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "model.pkl"
            serializer = ModelSerializer(serialization_format="pickle")
            serializer.serialize(model, filepath, compress=False)

            loaded_model = serializer.deserialize(filepath)
            assert isinstance(loaded_model, LinearRegression)
            np.testing.assert_array_almost_equal(
                model.predict(X), loaded_model.predict(X)
            )

    def test_deserialize_file_not_found(self):
        """Test error when deserializing non-existent file."""
        serializer = ModelSerializer(serialization_format="joblib")
        with pytest.raises(FileNotFoundError):
            serializer.deserialize(Path("nonexistent.joblib"))


class TestModelPersistence:
    """Test cases for ModelPersistence."""

    def test_initialization(self):
        """Test persistence manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)
            assert persistence.models_dir == Path(tmpdir)

    def test_save_and_load(self):
        """Test saving and loading a model."""
        model = LinearRegression()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)
            model_path = persistence.save(
                model, "test_model", "1.0.0", accuracy=0.95
            )

            assert model_path.exists()

            loaded_model, metadata = persistence.load(
                "test_model", "1.0.0", return_metadata=True
            )

            assert isinstance(loaded_model, LinearRegression)
            np.testing.assert_array_almost_equal(
                model.predict(X), loaded_model.predict(X)
            )
            assert metadata.model_name == "test_model"
            assert metadata.version == "1.0.0"
            assert metadata.metadata["accuracy"] == 0.95

    def test_save_multiple_versions(self):
        """Test saving multiple versions of a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            model1 = LinearRegression()
            model1.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(model1, "test_model", "1.0.0")

            model2 = LinearRegression()
            model2.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(model2, "test_model", "1.1.0")

            versions = persistence.list_versions("test_model")
            assert "1.0.0" in versions
            assert "1.1.0" in versions

    def test_load_latest_version(self):
        """Test loading latest version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            model1 = LinearRegression()
            model1.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(model1, "test_model", "1.0.0")

            model2 = LinearRegression()
            model2.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(model2, "test_model", "2.0.0")

            loaded_model = persistence.load("test_model")
            assert isinstance(loaded_model, LinearRegression)

    def test_get_metadata(self):
        """Test getting model metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            model = LinearRegression()
            model.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(
                model, "test_model", "1.0.0", accuracy=0.95, f1_score=0.92
            )

            metadata = persistence.get_metadata("test_model", "1.0.0")
            assert metadata is not None
            assert metadata.model_name == "test_model"
            assert metadata.version == "1.0.0"
            assert metadata.metadata["accuracy"] == 0.95
            assert metadata.metadata["f1_score"] == 0.92

    def test_list_models(self):
        """Test listing all models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            model1 = LinearRegression()
            model1.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(model1, "model1", "1.0.0")

            model2 = RandomForestClassifier()
            model2.fit(np.random.randn(100, 5), np.random.randint(0, 2, 100))
            persistence.save(model2, "model2", "1.0.0")

            models = persistence.list_models()
            assert "model1" in models
            assert "model2" in models

    def test_compare_versions(self):
        """Test comparing model versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            model1 = LinearRegression()
            model1.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(model1, "test_model", "1.0.0", accuracy=0.90)

            model2 = LinearRegression()
            model2.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(model2, "test_model", "1.1.0", accuracy=0.95)

            comparison = persistence.compare_versions("test_model")
            assert len(comparison) == 2
            assert "version" in comparison.columns
            assert "accuracy" in comparison.columns

    def test_delete_version(self):
        """Test deleting a model version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            model = LinearRegression()
            model.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(model, "test_model", "1.0.0")

            versions = persistence.list_versions("test_model")
            assert "1.0.0" in versions

            persistence.delete_version("test_model", "1.0.0")

            versions = persistence.list_versions("test_model")
            assert "1.0.0" not in versions

    def test_delete_nonexistent_version(self):
        """Test error when deleting non-existent version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            with pytest.raises(FileNotFoundError):
                persistence.delete_version("test_model", "1.0.0")

    def test_load_nonexistent_model(self):
        """Test error when loading non-existent model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            with pytest.raises(FileNotFoundError):
                persistence.load("nonexistent_model")

    def test_invalid_model_type(self):
        """Test error when saving non-estimator model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            with pytest.raises(ValueError, match="must be a scikit-learn BaseEstimator"):
                persistence.save("not_a_model", "test_model", "1.0.0")


class TestModelPersistenceManager:
    """Test cases for ModelPersistenceManager."""

    def test_initialization_with_config(self):
        """Test manager initialization with config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "logging": {"level": "DEBUG", "file": "logs/test.log"},
                "persistence": {
                    "models_dir": "models",
                    "serialization_format": "joblib",
                    "compress": True,
                },
            }
            import yaml

            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            manager = ModelPersistenceManager(config_path=config_path)
            assert manager.persistence is not None
        finally:
            config_path.unlink()

    def test_initialization_without_config(self):
        """Test manager initialization without config file."""
        manager = ModelPersistenceManager()
        assert manager.persistence is not None

    def test_save_and_load_model(self):
        """Test saving and loading through manager."""
        model = LinearRegression()
        model.fit(np.random.randn(100, 5), np.random.randn(100))

        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                config = {
                    "persistence": {"models_dir": tmpdir},
                }
                import yaml

                yaml.dump(config, f)
                config_path = Path(f.name)

            try:
                manager = ModelPersistenceManager(config_path=config_path)
                model_path = manager.save_model(
                    model, "test_model", "1.0.0", accuracy=0.95
                )

                assert model_path.exists()

                loaded_model = manager.load_model("test_model", "1.0.0")
                assert isinstance(loaded_model, LinearRegression)
            finally:
                config_path.unlink()

    def test_get_model_metadata(self):
        """Test getting metadata through manager."""
        model = LinearRegression()
        model.fit(np.random.randn(100, 5), np.random.randn(100))

        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                config = {"persistence": {"models_dir": tmpdir}}
                import yaml

                yaml.dump(config, f)
                config_path = Path(f.name)

            try:
                manager = ModelPersistenceManager(config_path=config_path)
                manager.save_model(model, "test_model", "1.0.0", accuracy=0.95)

                metadata = manager.get_model_metadata("test_model", "1.0.0")
                assert metadata is not None
                assert metadata.metadata["accuracy"] == 0.95
            finally:
                config_path.unlink()


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test complete workflow: save, load, compare, delete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            # Save multiple versions
            for version in ["1.0.0", "1.1.0", "2.0.0"]:
                model = LinearRegression()
                X = np.random.randn(100, 5)
                y = np.random.randn(100)
                model.fit(X, y)
                persistence.save(
                    model,
                    "test_model",
                    version,
                    accuracy=0.90 + float(version.split(".")[0]) * 0.01,
                )

            # List models
            models = persistence.list_models()
            assert "test_model" in models

            # List versions
            versions = persistence.list_versions("test_model")
            assert len(versions) == 3

            # Load latest
            latest_model = persistence.load("test_model")
            assert isinstance(latest_model, LinearRegression)

            # Compare versions
            comparison = persistence.compare_versions("test_model")
            assert len(comparison) == 3

            # Get metadata
            metadata = persistence.get_metadata("test_model", "1.0.0")
            assert metadata is not None

            # Delete version
            persistence.delete_version("test_model", "1.0.0")
            versions = persistence.list_versions("test_model")
            assert "1.0.0" not in versions
            assert len(versions) == 2

    def test_different_model_types(self):
        """Test persistence with different model types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            # Linear Regression
            lr = LinearRegression()
            lr.fit(np.random.randn(100, 5), np.random.randn(100))
            persistence.save(lr, "linear_model", "1.0.0")

            # Random Forest
            rf = RandomForestClassifier(n_estimators=10)
            rf.fit(np.random.randn(100, 5), np.random.randint(0, 2, 100))
            persistence.save(rf, "rf_model", "1.0.0")

            # Load both
            loaded_lr = persistence.load("linear_model")
            loaded_rf = persistence.load("rf_model")

            assert isinstance(loaded_lr, LinearRegression)
            assert isinstance(loaded_rf, RandomForestClassifier)

    def test_metadata_persistence(self):
        """Test that metadata persists correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = ModelPersistence(models_dir=tmpdir)

            model = LinearRegression()
            model.fit(np.random.randn(100, 5), np.random.randn(100))

            metadata_dict = {
                "accuracy": 0.95,
                "f1_score": 0.92,
                "training_samples": 1000,
                "features": ["feature1", "feature2", "feature3"],
            }

            persistence.save(model, "test_model", "1.0.0", **metadata_dict)

            metadata = persistence.get_metadata("test_model", "1.0.0")
            assert metadata.metadata["accuracy"] == 0.95
            assert metadata.metadata["f1_score"] == 0.92
            assert metadata.metadata["training_samples"] == 1000
            assert metadata.metadata["features"] == ["feature1", "feature2", "feature3"]
