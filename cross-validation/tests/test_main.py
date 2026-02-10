"""Unit tests for cross-validation implementation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.main import CrossValidator


class TestCrossValidator:
    """Test CrossValidator functionality."""

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
        cv = CrossValidator()
        assert cv.config is not None
        assert "logging" in cv.config
        assert "cross_validation" in cv.config

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config file."""
        config = {
            "logging": {"level": "WARNING", "file": "logs/test.log"},
            "cross_validation": {"n_splits": 10, "random_state": 123},
        }
        config_path = self.create_temp_config(config)
        try:
            cv = CrossValidator(config_path=config_path)
            assert cv.config["logging"]["level"] == "WARNING"
            assert cv.default_n_splits == 10
            assert cv.default_random_state == 123
        finally:
            Path(config_path).unlink()

    def test_validate_inputs_empty_data(self):
        """Test that empty data raises ValueError."""
        cv = CrossValidator()
        X = []
        with pytest.raises(ValueError, match="cannot be empty"):
            cv._validate_inputs(X)

    def test_validate_inputs_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        cv = CrossValidator()
        X = [[1], [2], [3]]
        y = [0, 1]
        with pytest.raises(ValueError, match="Length mismatch"):
            cv._validate_inputs(X, y)

    def test_k_fold_split_basic(self):
        """Test basic k-fold split."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5]]
        splits = cv.k_fold_split(X, n_splits=3)

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_k_fold_split_all_samples_used(self):
        """Test that all samples are used in k-fold split."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5]]
        splits = cv.k_fold_split(X, n_splits=3)

        all_test_indices = []
        for _, test_idx in splits:
            all_test_indices.extend(test_idx)

        assert len(set(all_test_indices)) == len(X)

    def test_k_fold_split_no_shuffle(self):
        """Test k-fold split without shuffling."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5]]
        splits = cv.k_fold_split(X, n_splits=3, shuffle=False)

        assert len(splits) == 3
        assert splits[0][1][0] == 0

    def test_k_fold_split_with_shuffle(self):
        """Test k-fold split with shuffling."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5]]
        splits = cv.k_fold_split(X, n_splits=3, shuffle=True, random_state=42)

        assert len(splits) == 3

    def test_k_fold_split_invalid_n_splits(self):
        """Test that invalid n_splits raises ValueError."""
        cv = CrossValidator()
        X = [[1], [2], [3]]
        with pytest.raises(ValueError):
            cv.k_fold_split(X, n_splits=1)
        with pytest.raises(ValueError):
            cv.k_fold_split(X, n_splits=10)

    def test_stratified_k_fold_split_basic(self):
        """Test basic stratified k-fold split."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5], [6]]
        y = [0, 0, 1, 1, 0, 1]
        splits = cv.stratified_k_fold_split(X, y, n_splits=3)

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_stratified_k_fold_split_class_distribution(self):
        """Test that stratified k-fold maintains class distribution."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        splits = cv.stratified_k_fold_split(X, y, n_splits=5)

        for train_idx, test_idx in splits:
            train_y = np.array(y)[train_idx]
            test_y = np.array(y)[test_idx]

            train_class_0 = np.sum(train_y == 0)
            train_class_1 = np.sum(train_y == 1)
            test_class_0 = np.sum(test_y == 0)
            test_class_1 = np.sum(test_y == 1)

            assert train_class_0 + test_class_0 == 5
            assert train_class_1 + test_class_1 == 5

    def test_stratified_k_fold_split_requires_y(self):
        """Test that stratified k-fold requires y."""
        cv = CrossValidator()
        X = [[1], [2], [3]]
        with pytest.raises(ValueError, match="y is required"):
            cv.stratified_k_fold_split(X, y=None)

    def test_stratified_k_fold_split_invalid_n_splits(self):
        """Test that invalid n_splits raises ValueError."""
        cv = CrossValidator()
        X = [[1], [2], [3]]
        y = [0, 1, 0]
        with pytest.raises(ValueError):
            cv.stratified_k_fold_split(X, y, n_splits=10)

    def test_leave_one_out_split_basic(self):
        """Test basic leave-one-out split."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4]]
        splits = cv.leave_one_out_split(X)

        assert len(splits) == 4
        for train_idx, test_idx in splits:
            assert len(test_idx) == 1
            assert len(train_idx) == 3

    def test_leave_one_out_split_all_samples(self):
        """Test that all samples are used as test in leave-one-out."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4]]
        splits = cv.leave_one_out_split(X)

        all_test_indices = []
        for _, test_idx in splits:
            all_test_indices.extend(test_idx)

        assert len(set(all_test_indices)) == len(X)
        assert sorted(all_test_indices) == list(range(len(X)))

    def test_get_split_summary(self):
        """Test split summary generation."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5]]
        splits = cv.k_fold_split(X, n_splits=3)
        summary = cv.get_split_summary(splits)

        assert "n_folds" in summary
        assert "folds" in summary
        assert summary["n_folds"] == 3
        assert len(summary["folds"]) == 3

    def test_get_split_summary_with_y(self):
        """Test split summary with target labels."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5], [6]]
        y = [0, 0, 1, 1, 0, 1]
        splits = cv.stratified_k_fold_split(X, y, n_splits=3)
        summary = cv.get_split_summary(splits, y)

        assert "folds" in summary
        for fold_info in summary["folds"]:
            assert "train_class_distribution" in fold_info
            assert "test_class_distribution" in fold_info

    def test_print_summary_no_exception(self):
        """Test that print_summary doesn't raise exceptions."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5]]
        splits = cv.k_fold_split(X, n_splits=3)
        cv.print_summary(splits)

    def test_print_summary_with_y(self):
        """Test print_summary with target labels."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5], [6]]
        y = [0, 0, 1, 1, 0, 1]
        splits = cv.stratified_k_fold_split(X, y, n_splits=3)
        cv.print_summary(splits, y)

    def test_save_splits(self):
        """Test saving splits to JSON."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5]]
        splits = cv.k_fold_split(X, n_splits=3)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name
            try:
                cv.save_splits(splits, output_path=save_path)
                assert Path(save_path).exists()

                import json

                with open(save_path) as f:
                    data = json.load(f)
                    assert "summary" in data
                    assert "splits" in data
            finally:
                Path(save_path).unlink()

    def test_with_pandas_dataframe(self):
        """Test cross-validation with pandas DataFrame."""
        cv = CrossValidator()
        df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [6, 7, 8, 9, 10]})
        splits = cv.k_fold_split(df, n_splits=3)

        assert len(splits) == 3

    def test_with_pandas_series(self):
        """Test cross-validation with pandas Series."""
        cv = CrossValidator()
        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])
        splits = cv.stratified_k_fold_split(X, y, n_splits=3)

        assert len(splits) == 3

    def test_k_fold_reproducibility(self):
        """Test that k-fold split is reproducible with same random_state."""
        cv = CrossValidator()
        X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

        splits1 = cv.k_fold_split(X, n_splits=3, shuffle=True, random_state=42)
        splits2 = cv.k_fold_split(X, n_splits=3, shuffle=True, random_state=42)

        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)
