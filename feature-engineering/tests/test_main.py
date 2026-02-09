"""Unit tests for feature engineering implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import FeatureEngineer


class TestFeatureEngineer:
    """Test FeatureEngineer functionality."""

    def create_temp_config(self, config_dict: dict) -> str:
        """Create temporary config file for testing.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Path to temporary config file.
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def test_initialization_with_default_config(self):
        """Test initialization with default config file."""
        engineer = FeatureEngineer()
        assert engineer.default_polynomial_degree == 2
        assert engineer.include_bias is False

    def test_create_polynomial_features_degree_1(self):
        """Test polynomial features with degree 1."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        engineer = FeatureEngineer()
        result = engineer.create_polynomial_features(df, degree=1)

        assert result.shape[0] == df.shape[0]
        assert "x" in result.columns
        assert "y" in result.columns

    def test_create_polynomial_features_degree_2(self):
        """Test polynomial features with degree 2."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        engineer = FeatureEngineer()
        result = engineer.create_polynomial_features(df, degree=2)

        assert result.shape[0] == df.shape[0]
        assert result.shape[1] > df.shape[1]

    def test_create_polynomial_features_with_bias(self):
        """Test polynomial features with bias term."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        engineer = FeatureEngineer()
        result = engineer.create_polynomial_features(df, degree=1, include_bias=True)

        assert "bias" in result.columns

    def test_create_polynomial_features_invalid_degree(self):
        """Test that invalid degree raises error."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="Degree must be >= 1"):
            engineer.create_polynomial_features(df, degree=0)

    def test_create_polynomial_features_numpy_array(self):
        """Test polynomial features with numpy array."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        engineer = FeatureEngineer()
        result = engineer.create_polynomial_features(
            X, degree=2, columns=["a", "b"]
        )

        assert result.shape[0] == X.shape[0]
        assert result.shape[1] > X.shape[1]

    def test_create_interaction_terms(self):
        """Test interaction term creation."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        engineer = FeatureEngineer()
        result = engineer.create_interaction_terms(df)

        assert result.shape[0] == df.shape[0]
        assert result.shape[1] > df.shape[1]
        assert "x * y" in result.columns or "y * x" in result.columns

    def test_create_interaction_terms_max_interactions(self):
        """Test interaction terms with max_interactions limit."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        engineer = FeatureEngineer()
        result = engineer.create_interaction_terms(df, max_interactions=1)

        interaction_cols = [
            col for col in result.columns if "*" in col
        ]
        assert len(interaction_cols) == 1

    def test_create_interaction_terms_custom_pairs(self):
        """Test interaction terms with custom feature pairs."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        engineer = FeatureEngineer()
        result = engineer.create_interaction_terms(
            df, feature_pairs=[("x", "y")]
        )

        assert "x * y" in result.columns
        interaction_cols = [col for col in result.columns if "*" in col]
        assert len(interaction_cols) == 1

    def test_create_interaction_terms_invalid_pair(self):
        """Test that invalid feature pair raises error."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="invalid column names"):
            engineer.create_interaction_terms(
                df, feature_pairs=[("x", "invalid")]
            )

    def test_create_interaction_terms_wrong_pair_length(self):
        """Test that wrong pair length raises error."""
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="exactly 2 features"):
            engineer.create_interaction_terms(
                df, feature_pairs=[("x", "y", "z")]
            )

    def test_create_polynomial_and_interactions(self):
        """Test creating both polynomial and interaction features."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        engineer = FeatureEngineer()
        result = engineer.create_polynomial_and_interactions(
            df, polynomial_degree=2, include_interactions=True
        )

        assert result.shape[0] == df.shape[0]
        assert result.shape[1] > df.shape[1]

    def test_create_polynomial_and_interactions_no_interactions(self):
        """Test creating polynomial features without interactions."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        engineer = FeatureEngineer()
        result = engineer.create_polynomial_and_interactions(
            df, polynomial_degree=2, include_interactions=False
        )

        assert result.shape[0] == df.shape[0]
        interaction_cols = [col for col in result.columns if "*" in col]
        assert len(interaction_cols) == 0

    def test_get_feature_info(self):
        """Test getting feature information."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        engineer = FeatureEngineer()
        engineer.create_polynomial_features(df, degree=2)

        info = engineer.get_feature_info()
        assert "n_features" in info
        assert "feature_names" in info
        assert len(info["feature_names"]) > 0

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            FeatureEngineer(config_path="nonexistent.yaml")

    def test_polynomial_features_preserve_data(self):
        """Test that polynomial features preserve original data values."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        engineer = FeatureEngineer()
        result = engineer.create_polynomial_features(df, degree=1)

        assert result["x"].equals(df["x"])
        assert result["y"].equals(df["y"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
