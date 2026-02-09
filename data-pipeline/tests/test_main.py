"""Unit tests for data pipeline implementation."""

import pandas as pd
import pytest

from src.main import (
    DataPipeline,
    ImputerTransformer,
    MinMaxScalerTransformer,
    OneHotEncoderTransformer,
    StandardScalerTransformer,
)


class TestStandardScalerTransformer:
    """Test StandardScalerTransformer functionality."""

    def test_fit_and_transform(self):
        """Test fitting and transforming data."""
        df = pd.DataFrame({"age": [25, 30, 35], "score": [85, 90, 95]})
        scaler = StandardScalerTransformer()
        scaler.fit(df)
        transformed = scaler.transform(df)

        assert transformed.shape == df.shape
        assert transformed["age"].mean() < 1e-10
        assert abs(transformed["age"].std() - 1.0) < 1e-10

    def test_fit_transform(self):
        """Test fit_transform method."""
        df = pd.DataFrame({"age": [25, 30, 35], "score": [85, 90, 95]})
        scaler = StandardScalerTransformer()
        transformed = scaler.fit_transform(df)

        assert transformed.shape == df.shape

    def test_specific_columns(self):
        """Test scaling specific columns."""
        df = pd.DataFrame({"age": [25, 30, 35], "score": [85, 90, 95], "name": ["A", "B", "C"]})
        scaler = StandardScalerTransformer(columns=["age"])
        transformed = scaler.fit_transform(df)

        assert "age" in transformed.columns
        assert "score" in transformed.columns
        assert "name" in transformed.columns

    def test_transform_before_fit(self):
        """Test that transform before fit raises error."""
        df = pd.DataFrame({"age": [25, 30, 35]})
        scaler = StandardScalerTransformer()
        with pytest.raises(ValueError, match="not fitted"):
            scaler.transform(df)

    def test_invalid_column(self):
        """Test that invalid column raises error."""
        df = pd.DataFrame({"age": [25, 30, 35]})
        scaler = StandardScalerTransformer(columns=["invalid"])
        with pytest.raises(ValueError, match="not found"):
            scaler.fit(df)


class TestMinMaxScalerTransformer:
    """Test MinMaxScalerTransformer functionality."""

    def test_fit_and_transform(self):
        """Test fitting and transforming data."""
        df = pd.DataFrame({"age": [25, 30, 35], "score": [85, 90, 95]})
        scaler = MinMaxScalerTransformer()
        scaler.fit(df)
        transformed = scaler.transform(df)

        assert transformed.shape == df.shape
        assert transformed["age"].min() >= 0
        assert transformed["age"].max() <= 1

    def test_custom_range(self):
        """Test custom feature range."""
        df = pd.DataFrame({"age": [25, 30, 35]})
        scaler = MinMaxScalerTransformer(feature_range=(-1, 1))
        transformed = scaler.fit_transform(df)

        assert transformed["age"].min() >= -1
        assert transformed["age"].max() <= 1


class TestImputerTransformer:
    """Test ImputerTransformer functionality."""

    def test_mean_strategy(self):
        """Test mean imputation strategy."""
        df = pd.DataFrame({"age": [25, None, 35], "score": [85, 90, None]})
        imputer = ImputerTransformer(strategy="mean")
        transformed = imputer.fit_transform(df)

        assert transformed["age"].isna().sum() == 0
        assert transformed["score"].isna().sum() == 0

    def test_median_strategy(self):
        """Test median imputation strategy."""
        df = pd.DataFrame({"age": [25, None, 35]})
        imputer = ImputerTransformer(strategy="median")
        transformed = imputer.fit_transform(df)

        assert transformed["age"].isna().sum() == 0

    def test_mode_strategy(self):
        """Test mode imputation strategy."""
        df = pd.DataFrame({"age": [25, None, 25, 30]})
        imputer = ImputerTransformer(strategy="mode")
        transformed = imputer.fit_transform(df)

        assert transformed["age"].isna().sum() == 0

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        df = pd.DataFrame({"age": [25, 30, 35]})
        imputer = ImputerTransformer(strategy="invalid")
        with pytest.raises(ValueError, match="Invalid strategy"):
            imputer.fit(df)


class TestOneHotEncoderTransformer:
    """Test OneHotEncoderTransformer functionality."""

    def test_fit_and_transform(self):
        """Test fitting and transforming data."""
        df = pd.DataFrame({"category": ["A", "B", "A", "C"]})
        encoder = OneHotEncoderTransformer()
        encoder.fit(df)
        transformed = encoder.transform(df)

        assert "category" not in transformed.columns
        assert "category_A" in transformed.columns
        assert "category_B" in transformed.columns
        assert "category_C" in transformed.columns

    def test_specific_columns(self):
        """Test encoding specific columns."""
        df = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"], "num": [1, 2]})
        encoder = OneHotEncoderTransformer(columns=["cat1"])
        transformed = encoder.fit_transform(df)

        assert "cat1" not in transformed.columns
        assert "cat2" in transformed.columns
        assert "num" in transformed.columns


class TestDataPipeline:
    """Test DataPipeline functionality."""

    def test_empty_transformers(self):
        """Test that empty transformers list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DataPipeline(transformers=[])

    def test_pipeline_fit_and_transform(self):
        """Test pipeline fit and transform."""
        df = pd.DataFrame({"age": [25, None, 35], "score": [85, 90, None]})
        pipeline = DataPipeline(
            transformers=[
                ImputerTransformer(strategy="mean"),
                StandardScalerTransformer(),
            ]
        )

        transformed = pipeline.fit_transform(df)

        assert transformed.shape[0] == df.shape[0]
        assert transformed.isna().sum().sum() == 0

    def test_pipeline_separate_fit_transform(self):
        """Test pipeline with separate fit and transform."""
        df_train = pd.DataFrame({"age": [25, None, 35], "score": [85, 90, None]})
        df_test = pd.DataFrame({"age": [30, None], "score": [88, None]})

        pipeline = DataPipeline(
            transformers=[
                ImputerTransformer(strategy="mean"),
                StandardScalerTransformer(),
            ]
        )

        pipeline.fit(df_train)
        transformed_train = pipeline.transform(df_train)
        transformed_test = pipeline.transform(df_test)

        assert transformed_train.shape[0] == df_train.shape[0]
        assert transformed_test.shape[0] == df_test.shape[0]

    def test_pipeline_multiple_transformers(self):
        """Test pipeline with multiple transformers."""
        df = pd.DataFrame(
            {
                "age": [25, None, 35],
                "score": [85, 90, None],
                "category": ["A", "B", "A"],
            }
        )

        pipeline = DataPipeline(
            transformers=[
                ImputerTransformer(strategy="mean"),
                StandardScalerTransformer(),
                OneHotEncoderTransformer(columns=["category"]),
            ]
        )

        transformed = pipeline.fit_transform(df)

        assert transformed.isna().sum().sum() == 0
        assert "category" not in transformed.columns

    def test_pipeline_info(self):
        """Test getting pipeline information."""
        pipeline = DataPipeline(
            transformers=[
                ImputerTransformer(),
                StandardScalerTransformer(),
            ]
        )

        info = pipeline.get_pipeline_info()
        assert info["n_transformers"] == 2
        assert info["is_fitted"] is False

        df = pd.DataFrame({"age": [25, 30, 35]})
        pipeline.fit(df)

        info = pipeline.get_pipeline_info()
        assert info["is_fitted"] is True

    def test_pipeline_transform_before_fit(self):
        """Test that transform before fit raises error."""
        df = pd.DataFrame({"age": [25, 30, 35]})
        pipeline = DataPipeline(transformers=[StandardScalerTransformer()])

        with pytest.raises(ValueError, match="not fitted"):
            pipeline.transform(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
