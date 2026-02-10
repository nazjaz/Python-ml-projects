"""Tests for imbalanced dataset handling module."""

import numpy as np
import pytest

from src.main import (
    ClassWeightCalculator,
    ImbalancedDatasetHandler,
    SMOTE,
    Undersampler,
)


class TestSMOTE:
    """Test cases for SMOTE oversampling."""

    def test_smote_basic(self):
        """Test SMOTE with basic imbalanced dataset."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 0, 1, 1, 1])

        smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy=1.0)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        assert X_resampled.shape[0] > X.shape[0]
        assert len(y_resampled) == len(X_resampled)
        assert np.sum(y_resampled == 0) == np.sum(y_resampled == 1)

    def test_smote_shape_mismatch(self):
        """Test SMOTE raises error on shape mismatch."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 1, 1])

        smote = SMOTE()
        with pytest.raises(ValueError, match="X and y must have same number"):
            smote.fit_resample(X, y)

    def test_smote_single_class(self):
        """Test SMOTE raises error for single class."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 0])

        smote = SMOTE()
        with pytest.raises(ValueError, match="SMOTE requires at least 2 classes"):
            smote.fit_resample(X, y)

    def test_smote_sampling_strategy(self):
        """Test SMOTE with different sampling strategy."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 0, 0, 1])

        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        n_minority = np.sum(y_resampled == 1)
        n_majority = np.sum(y_resampled == 0)
        ratio = n_minority / n_majority

        assert ratio <= 0.6


class TestUndersampler:
    """Test cases for undersampling techniques."""

    def test_random_undersample_basic(self):
        """Test random undersampling with basic dataset."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 0, 1, 1, 1])

        X_resampled, y_resampled = Undersampler.random_undersample(
            X, y, sampling_strategy=1.0, random_state=42
        )

        assert X_resampled.shape[0] < X.shape[0]
        assert np.sum(y_resampled == 0) == np.sum(y_resampled == 1)

    def test_random_undersample_shape_mismatch(self):
        """Test random undersampling raises error on shape mismatch."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="X and y must have same number"):
            Undersampler.random_undersample(X, y)

    def test_random_undersample_single_class(self):
        """Test random undersampling raises error for single class."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 0])

        with pytest.raises(ValueError, match="Undersampling requires at least 2 classes"):
            Undersampler.random_undersample(X, y)

    def test_tomek_links_undersample(self):
        """Test Tomek Links undersampling."""
        X = np.array(
            [
                [1, 1],
                [1.1, 1.1],
                [2, 2],
                [2.1, 2.1],
                [3, 3],
                [3.1, 3.1],
                [10, 10],
            ]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0])

        X_resampled, y_resampled = Undersampler.tomek_links_undersample(X, y)

        assert X_resampled.shape[0] <= X.shape[0]

    def test_tomek_links_shape_mismatch(self):
        """Test Tomek Links raises error on shape mismatch."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="X and y must have same number"):
            Undersampler.tomek_links_undersample(X, y)

    def test_enn_undersample(self):
        """Test Edited Nearest Neighbours undersampling."""
        X = np.array(
            [
                [1, 1],
                [1.1, 1.1],
                [2, 2],
                [2.1, 2.1],
                [3, 3],
                [3.1, 3.1],
                [10, 10],
            ]
        )
        y = np.array([0, 1, 0, 1, 0, 1, 0])

        X_resampled, y_resampled = Undersampler.edited_nearest_neighbours_undersample(
            X, y
        )

        assert X_resampled.shape[0] <= X.shape[0]

    def test_enn_shape_mismatch(self):
        """Test ENN raises error on shape mismatch."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 1, 1])

        with pytest.raises(ValueError, match="X and y must have same number"):
            Undersampler.edited_nearest_neighbours_undersample(X, y)


class TestClassWeightCalculator:
    """Test cases for class weight calculation."""

    def test_balanced_weights_basic(self):
        """Test balanced weights calculation."""
        y = np.array([0, 0, 0, 0, 1, 1])
        weights = ClassWeightCalculator.balanced_weights(y)

        assert 0 in weights
        assert 1 in weights
        assert weights[1] > weights[0]

    def test_balanced_weights_single_class(self):
        """Test balanced weights raises error for single class."""
        y = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="Class weights require at least 2 classes"):
            ClassWeightCalculator.balanced_weights(y)

    def test_compute_class_weight_balanced(self):
        """Test compute class weight with balanced method."""
        y = np.array([0, 0, 0, 0, 1, 1])
        weights = ClassWeightCalculator.compute_class_weight(y, method="balanced")

        assert 0 in weights
        assert 1 in weights

    def test_compute_class_weight_inverse(self):
        """Test compute class weight with inverse method."""
        y = np.array([0, 0, 0, 0, 1, 1])
        weights = ClassWeightCalculator.compute_class_weight(y, method="inverse")

        assert 0 in weights
        assert 1 in weights
        assert weights[1] > weights[0]

    def test_compute_class_weight_invalid_method(self):
        """Test compute class weight raises error for invalid method."""
        y = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="Unknown method"):
            ClassWeightCalculator.compute_class_weight(y, method="invalid")

    def test_custom_weights(self):
        """Test custom weights application."""
        y = np.array([0, 0, 1, 1, 2, 2])
        weight_dict = {0: 1.0, 1: 2.0, 2: 3.0}
        weights = ClassWeightCalculator.custom_weights(y, weight_dict)

        assert weights[0] == 1.0
        assert weights[1] == 2.0
        assert weights[2] == 3.0


class TestImbalancedDatasetHandler:
    """Test cases for ImbalancedDatasetHandler."""

    def test_handler_initialization(self):
        """Test handler can be initialized."""
        handler = ImbalancedDatasetHandler()
        assert handler is not None

    def test_get_class_distribution(self):
        """Test class distribution calculation."""
        handler = ImbalancedDatasetHandler()
        y = np.array([0, 0, 0, 0, 1, 1])
        distribution = handler.get_class_distribution(y)

        assert "total_samples" in distribution
        assert "n_classes" in distribution
        assert "class_counts" in distribution
        assert distribution["total_samples"] == 6
        assert distribution["n_classes"] == 2

    def test_apply_smote(self):
        """Test applying SMOTE through handler."""
        handler = ImbalancedDatasetHandler()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 0, 1, 1, 1])

        X_resampled, y_resampled = handler.apply_smote(X, y, scale_features=False)

        assert X_resampled.shape[0] > X.shape[0]
        assert len(y_resampled) == len(X_resampled)

    def test_apply_undersampling_random(self):
        """Test applying random undersampling through handler."""
        handler = ImbalancedDatasetHandler()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 0, 1, 1, 1])

        X_resampled, y_resampled = handler.apply_undersampling(
            X, y, method="random"
        )

        assert X_resampled.shape[0] < X.shape[0]

    def test_apply_undersampling_tomek(self):
        """Test applying Tomek Links undersampling through handler."""
        handler = ImbalancedDatasetHandler()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 0, 1, 1, 1])

        X_resampled, y_resampled = handler.apply_undersampling(X, y, method="tomek")

        assert X_resampled.shape[0] <= X.shape[0]

    def test_apply_undersampling_enn(self):
        """Test applying ENN undersampling through handler."""
        handler = ImbalancedDatasetHandler()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 0, 1, 1, 1])

        X_resampled, y_resampled = handler.apply_undersampling(X, y, method="enn")

        assert X_resampled.shape[0] <= X.shape[0]

    def test_apply_undersampling_invalid_method(self):
        """Test applying invalid undersampling method raises error."""
        handler = ImbalancedDatasetHandler()
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 1])

        with pytest.raises(ValueError, match="Unknown method"):
            handler.apply_undersampling(X, y, method="invalid")

    def test_compute_class_weights_balanced(self):
        """Test computing balanced class weights through handler."""
        handler = ImbalancedDatasetHandler()
        y = np.array([0, 0, 0, 0, 1, 1])
        weights = handler.compute_class_weights(y, method="balanced")

        assert 0 in weights
        assert 1 in weights

    def test_compute_class_weights_inverse(self):
        """Test computing inverse class weights through handler."""
        handler = ImbalancedDatasetHandler()
        y = np.array([0, 0, 0, 0, 1, 1])
        weights = handler.compute_class_weights(y, method="inverse")

        assert 0 in weights
        assert 1 in weights


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_smote_then_undersample(self):
        """Test applying SMOTE then undersampling."""
        handler = ImbalancedDatasetHandler()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 0, 0, 1])

        X_smote, y_smote = handler.apply_smote(X, y, scale_features=False)
        X_final, y_final = handler.apply_undersampling(
            X_smote, y_smote, method="random"
        )

        assert X_final.shape[0] > X.shape[0]

    def test_complete_workflow(self):
        """Test complete workflow with all techniques."""
        handler = ImbalancedDatasetHandler()
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 0, 1, 1, 1])

        distribution = handler.get_class_distribution(y)
        weights = handler.compute_class_weights(y)
        X_resampled, y_resampled = handler.apply_smote(X, y, scale_features=False)

        assert distribution is not None
        assert weights is not None
        assert X_resampled.shape[0] > X.shape[0]
