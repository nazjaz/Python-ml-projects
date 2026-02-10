"""Tests for multi-label classification module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.main import (
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset,
    MultiLabelClassifier,
    MultiLabelEvaluator,
)


class TestBinaryRelevance:
    """Test cases for Binary Relevance classifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        br = BinaryRelevance()
        assert br.base_estimator is not None

    def test_initialization_custom_estimator(self):
        """Test initialization with custom estimator."""
        estimator = LogisticRegression()
        br = BinaryRelevance(base_estimator=estimator)
        assert br.base_estimator == estimator

    def test_fit(self):
        """Test model fitting."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        br = BinaryRelevance()
        br.fit(X, y)

        assert br.estimators_ is not None
        assert len(br.estimators_) == 3

    def test_fit_list_labels(self):
        """Test fitting with list of label sets."""
        X = np.random.randn(100, 5)
        y = [[0, 1], [1], [0, 1, 2], [2], [0]] * 20

        br = BinaryRelevance()
        br.fit(X, y)

        assert br.estimators_ is not None

    def test_fit_shape_mismatch(self):
        """Test fitting with shape mismatch."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (50, 3))

        br = BinaryRelevance()
        with pytest.raises(ValueError, match="X and y must have same number"):
            br.fit(X, y)

    def test_fit_1d_labels(self):
        """Test fitting with 1D labels (should fail)."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        br = BinaryRelevance()
        with pytest.raises(ValueError, match="y must be 2D array"):
            br.fit(X, y)

    def test_predict(self):
        """Test prediction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        br = BinaryRelevance()
        br.fit(X, y)

        X_test = np.random.randn(10, 5)
        predictions = br.predict(X_test)

        assert predictions.shape == (10, 3)
        assert np.all((predictions == 0) | (predictions == 1))

    def test_predict_not_fitted(self):
        """Test prediction without fitting."""
        br = BinaryRelevance()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Model must be fitted"):
            br.predict(X)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        br = BinaryRelevance(base_estimator=LogisticRegression(random_state=42))
        br.fit(X, y)

        X_test = np.random.randn(10, 5)
        probabilities = br.predict_proba(X_test)

        assert probabilities.shape == (10, 3)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestClassifierChain:
    """Test cases for Classifier Chain."""

    def test_initialization(self):
        """Test classifier initialization."""
        cc = ClassifierChain()
        assert cc.base_estimator is not None

    def test_initialization_custom_order(self):
        """Test initialization with custom order."""
        cc = ClassifierChain(order=[2, 0, 1])
        assert cc.order == [2, 0, 1]

    def test_fit(self):
        """Test model fitting."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        cc = ClassifierChain(random_state=42)
        cc.fit(X, y)

        assert cc.estimators_ is not None
        assert len(cc.estimators_) == 3
        assert cc.chain_order_ is not None

    def test_fit_custom_order(self):
        """Test fitting with custom order."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        cc = ClassifierChain(order=[2, 0, 1])
        cc.fit(X, y)

        assert cc.chain_order_ == [2, 0, 1]

    def test_fit_shape_mismatch(self):
        """Test fitting with shape mismatch."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (50, 3))

        cc = ClassifierChain()
        with pytest.raises(ValueError, match="X and y must have same number"):
            cc.fit(X, y)

    def test_predict(self):
        """Test prediction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        cc = ClassifierChain(random_state=42)
        cc.fit(X, y)

        X_test = np.random.randn(10, 5)
        predictions = cc.predict(X_test)

        assert predictions.shape == (10, 3)
        assert np.all((predictions == 0) | (predictions == 1))

    def test_predict_not_fitted(self):
        """Test prediction without fitting."""
        cc = ClassifierChain()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Model must be fitted"):
            cc.predict(X)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        cc = ClassifierChain(base_estimator=LogisticRegression(random_state=42))
        cc.fit(X, y)

        X_test = np.random.randn(10, 5)
        probabilities = cc.predict_proba(X_test)

        assert probabilities.shape == (10, 3)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestLabelPowerset:
    """Test cases for Label Powerset classifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        lp = LabelPowerset()
        assert lp.base_estimator is not None

    def test_initialization_custom_estimator(self):
        """Test initialization with custom estimator."""
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        lp = LabelPowerset(base_estimator=estimator)
        assert lp.base_estimator == estimator

    def test_fit(self):
        """Test model fitting."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        lp = LabelPowerset()
        lp.fit(X, y)

        assert lp.estimator_ is not None
        assert lp.label_encoder_ is not None

    def test_fit_shape_mismatch(self):
        """Test fitting with shape mismatch."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (50, 3))

        lp = LabelPowerset()
        with pytest.raises(ValueError, match="X and y must have same number"):
            lp.fit(X, y)

    def test_predict(self):
        """Test prediction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        lp = LabelPowerset()
        lp.fit(X, y)

        X_test = np.random.randn(10, 5)
        predictions = lp.predict(X_test)

        assert predictions.shape == (10, 3)
        assert np.all((predictions == 0) | (predictions == 1))

    def test_predict_not_fitted(self):
        """Test prediction without fitting."""
        lp = LabelPowerset()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Model must be fitted"):
            lp.predict(X)

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        lp = LabelPowerset(
            base_estimator=RandomForestClassifier(n_estimators=10, random_state=42)
        )
        lp.fit(X, y)

        X_test = np.random.randn(10, 5)
        probabilities = lp.predict_proba(X_test)

        assert probabilities.shape == (10, 3)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestMultiLabelEvaluator:
    """Test cases for multi-label evaluator."""

    def test_evaluate(self):
        """Test evaluation metrics."""
        y_true = np.random.randint(0, 2, (100, 3))
        y_pred = np.random.randint(0, 2, (100, 3))

        metrics = MultiLabelEvaluator.evaluate(y_true, y_pred)

        assert "hamming_loss" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "jaccard_score" in metrics

        assert all(0 <= v <= 1 for v in metrics.values() if v != "hamming_loss")
        assert metrics["hamming_loss"] >= 0


class TestMultiLabelClassifier:
    """Test cases for main multi-label classifier class."""

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = MultiLabelClassifier()
        assert classifier.binary_relevance is None
        assert classifier.classifier_chain is None
        assert classifier.label_powerset is None

    def test_load_data(self):
        """Test data loading."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)

        assert classifier.X is not None
        assert classifier.y is not None

    def test_load_data_dataframe(self):
        """Test loading with DataFrame."""
        df = pd.DataFrame(
            {
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "label_1": np.random.randint(0, 2, 100),
                "label_2": np.random.randint(0, 2, 100),
            }
        )

        X = df[["feature_1", "feature_2"]]
        y = df[["label_1", "label_2"]].values

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)

        assert classifier.X is not None
        assert classifier.y is not None

    def test_fit_binary_relevance(self):
        """Test fitting Binary Relevance."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)
        classifier.fit_binary_relevance()

        assert classifier.binary_relevance is not None

    def test_fit_binary_relevance_no_data(self):
        """Test fitting without loading data."""
        classifier = MultiLabelClassifier()
        with pytest.raises(ValueError, match="Data must be loaded"):
            classifier.fit_binary_relevance()

    def test_fit_classifier_chain(self):
        """Test fitting Classifier Chain."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)
        classifier.fit_classifier_chain()

        assert classifier.classifier_chain is not None

    def test_fit_label_powerset(self):
        """Test fitting Label Powerset."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)
        classifier.fit_label_powerset()

        assert classifier.label_powerset is not None

    def test_fit_all(self):
        """Test fitting all models."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)
        classifier.fit_all()

        assert classifier.binary_relevance is not None
        assert classifier.classifier_chain is not None
        assert classifier.label_powerset is not None

    def test_predict(self):
        """Test prediction."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)
        classifier.fit_binary_relevance()

        X_test = np.random.randn(10, 5)
        predictions = classifier.predict(X_test, method="binary_relevance")

        assert predictions.shape == (10, 3)

    def test_predict_invalid_method(self):
        """Test prediction with invalid method."""
        classifier = MultiLabelClassifier()
        with pytest.raises(ValueError, match="Unknown method"):
            classifier.predict(np.random.randn(10, 5), method="invalid")

    def test_evaluate(self):
        """Test evaluation."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, (100, 3))

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)
        classifier.fit_binary_relevance()

        X_test = np.random.randn(50, 5)
        y_test = np.random.randint(0, 2, (50, 3))

        evaluation = classifier.evaluate(X_test, y_test, method="binary_relevance")

        assert "hamming_loss" in evaluation
        assert "accuracy" in evaluation
        assert "f1_score" in evaluation


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self):
        """Test complete multi-label classification workflow."""
        X = np.random.randn(200, 10)
        y = np.random.randint(0, 2, (200, 5))

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)
        classifier.fit_all()

        X_test = np.random.randn(50, 10)
        y_test = np.random.randint(0, 2, (50, 5))

        br_pred = classifier.predict(X_test, method="binary_relevance")
        cc_pred = classifier.predict(X_test, method="classifier_chain")
        lp_pred = classifier.predict(X_test, method="label_powerset")

        br_eval = classifier.evaluate(X_test, y_test, method="binary_relevance")
        cc_eval = classifier.evaluate(X_test, y_test, method="classifier_chain")
        lp_eval = classifier.evaluate(X_test, y_test, method="label_powerset")

        assert br_pred.shape == (50, 5)
        assert cc_pred.shape == (50, 5)
        assert lp_pred.shape == (50, 5)

        assert "f1_score" in br_eval
        assert "f1_score" in cc_eval
        assert "f1_score" in lp_eval

    def test_list_labels_workflow(self):
        """Test workflow with list of label sets."""
        X = np.random.randn(100, 5)
        y = [[0, 1], [1], [0, 1, 2], [2], [0]] * 20

        classifier = MultiLabelClassifier()
        classifier.load_data(X, y)
        classifier.fit_binary_relevance()

        X_test = np.random.randn(10, 5)
        predictions = classifier.predict(X_test, method="binary_relevance")

        assert predictions.shape[1] > 0
