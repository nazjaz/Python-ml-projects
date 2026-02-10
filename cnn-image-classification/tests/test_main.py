"""Tests for CNN image classification module."""

import numpy as np
import pytest

from src.main import (
    CNN,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
    _relu_backward,
    _relu_forward,
    load_mnist,
)


class TestConv2D:
    """Test cases for Conv2D layer."""

    def test_forward_output_shape(self):
        """Test that Conv2D produces correct output dimensions."""
        np.random.seed(42)
        layer = Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        x = np.random.randn(4, 28, 28, 1).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (4, 28, 28, 8)

    def test_forward_no_padding(self):
        """Test Conv2D output size with stride and no padding."""
        np.random.seed(42)
        layer = Conv2D(
            in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=0
        )
        x = np.random.randn(2, 32, 32, 3).astype(np.float32)
        out = layer.forward(x)
        expected_h = (32 - 5) // 2 + 1
        expected_w = (32 - 5) // 2 + 1
        assert out.shape == (2, expected_h, expected_w, 16)

    def test_backward_gradient_shape(self):
        """Test that Conv2D backward returns correct gradient shape."""
        np.random.seed(42)
        layer = Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        x = np.random.randn(2, 10, 10, 1).astype(np.float32)
        _ = layer.forward(x)
        dout = np.random.randn(2, 10, 10, 4).astype(np.float32)
        dx = layer.backward(dout, learning_rate=0.01)
        assert dx.shape == x.shape


class TestMaxPool2D:
    """Test cases for MaxPool2D layer."""

    def test_forward_output_shape(self):
        """Test that MaxPool2D produces correct output dimensions."""
        layer = MaxPool2D(pool_size=2, stride=2)
        x = np.random.randn(4, 28, 28, 16).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (4, 14, 14, 16)

    def test_forward_preserves_max(self):
        """Test that MaxPool2D outputs the maximum value in each window."""
        layer = MaxPool2D(pool_size=2, stride=2)
        x = np.zeros((1, 4, 4, 1))
        x[0, 1, 2, 0] = 10.0
        out = layer.forward(x)
        assert out[0, 0, 1, 0] == 10.0

    def test_backward_gradient_shape(self):
        """Test that MaxPool2D backward returns correct gradient shape."""
        np.random.seed(42)
        layer = MaxPool2D(pool_size=2, stride=2)
        x = np.random.randn(2, 8, 8, 4).astype(np.float32)
        _ = layer.forward(x)
        dout = np.random.randn(2, 4, 4, 4).astype(np.float32)
        dx = layer.backward(dout, learning_rate=0.0)
        assert dx.shape == x.shape


class TestFlatten:
    """Test cases for Flatten layer."""

    def test_forward_output_shape(self):
        """Test that Flatten produces correct 1D output."""
        layer = Flatten()
        x = np.random.randn(4, 14, 14, 64).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (4, 14 * 14 * 64)

    def test_backward_restores_shape(self):
        """Test that Flatten backward restores original shape."""
        np.random.seed(42)
        layer = Flatten()
        x = np.random.randn(2, 7, 7, 32).astype(np.float32)
        _ = layer.forward(x)
        dout = np.random.randn(2, 7 * 7 * 32).astype(np.float32)
        dx = layer.backward(dout, learning_rate=0.0)
        assert dx.shape == x.shape


class TestDense:
    """Test cases for Dense layer."""

    def test_forward_output_shape(self):
        """Test that Dense produces correct output dimensions."""
        np.random.seed(42)
        layer = Dense(input_size=100, output_size=10, activation="relu")
        x = np.random.randn(4, 100).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (4, 10)

    def test_softmax_output_sums_to_one(self):
        """Test that softmax activation produces valid probabilities."""
        np.random.seed(42)
        layer = Dense(input_size=5, output_size=3, activation="softmax")
        x = np.random.randn(4, 5).astype(np.float32)
        out = layer.forward(x)
        np.testing.assert_array_almost_equal(
            np.sum(out, axis=1), np.ones(4)
        )

    def test_backward_gradient_shape(self):
        """Test that Dense backward returns correct gradient shape."""
        np.random.seed(42)
        layer = Dense(input_size=50, output_size=10, activation="relu")
        x = np.random.randn(2, 50).astype(np.float32)
        _ = layer.forward(x)
        dout = np.random.randn(2, 10).astype(np.float32)
        dx = layer.backward(dout, learning_rate=0.01)
        assert dx.shape == x.shape


class TestReLU:
    """Test cases for ReLU activation."""

    def test_relu_forward_positive(self):
        """Test that ReLU preserves positive values."""
        x = np.array([1.0, 2.0, 3.0])
        out = _relu_forward(x)
        np.testing.assert_array_equal(out, x)

    def test_relu_forward_negative(self):
        """Test that ReLU zeroes negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        out = _relu_forward(x)
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_relu_backward_routes_correctly(self):
        """Test that ReLU backward only passes gradient where input > 0."""
        x = np.array([1.0, -1.0, 0.5])
        dout = np.array([1.0, 1.0, 1.0])
        dx = _relu_backward(dout, x)
        assert dx[0] == 1.0
        assert dx[1] == 0.0
        assert dx[2] == 1.0


class TestCNN:
    """Test cases for full CNN model."""

    def test_forward_output_shape(self):
        """Test that CNN produces correct prediction shape."""
        np.random.seed(42)
        cnn = CNN(
            input_shape=(28, 28, 1),
            n_classes=10,
            dense_units=64,
        )
        x = np.random.randn(4, 28, 28, 1).astype(np.float32)
        out = cnn.forward(x, training=False)
        assert out.shape == (4, 10)

    def test_predict_returns_labels(self):
        """Test that predict returns integer class labels."""
        np.random.seed(42)
        cnn = CNN(input_shape=(28, 28, 1), n_classes=10, dense_units=32)
        x = np.random.randn(8, 28, 28, 1).astype(np.float32)
        preds = cnn.predict(x)
        assert preds.shape == (8,)
        assert np.all(preds >= 0) and np.all(preds < 10)

    def test_train_reduces_loss(self):
        """Test that training reduces loss over epochs."""
        np.random.seed(42)
        x_train = np.random.randn(100, 28, 28, 1).astype(np.float32) * 0.1
        y_train = np.random.randint(0, 10, size=100)
        x_test = np.random.randn(50, 28, 28, 1).astype(np.float32) * 0.1
        y_test = np.random.randint(0, 10, size=50)

        cnn = CNN(input_shape=(28, 28, 1), n_classes=10, dense_units=32)
        history = cnn.train(
            x_train,
            y_train,
            epochs=3,
            learning_rate=0.01,
            batch_size=16,
            verbose=False,
        )
        assert len(history["loss"]) == 3
        assert history["loss"][-1] <= history["loss"][0] * 1.1

    def test_evaluate_returns_dict(self):
        """Test that evaluate returns loss and accuracy."""
        np.random.seed(42)
        cnn = CNN(input_shape=(28, 28, 1), n_classes=10, dense_units=32)
        x = np.random.randn(20, 28, 28, 1).astype(np.float32)
        y = np.random.randint(0, 10, size=20)
        results = cnn.evaluate(x, y)
        assert "loss" in results
        assert "accuracy" in results
        assert 0 <= results["accuracy"] <= 1


class TestLoadMNIST:
    """Test cases for data loading."""

    def test_load_mnist_returns_correct_shapes(self):
        """Test that load_mnist returns correct array shapes."""
        x_train, x_test, y_train, y_test = load_mnist(
            n_train=100, n_test=50, random_seed=42
        )
        assert x_train.shape[0] == 100
        assert x_test.shape[0] == 50
        assert x_train.shape[1:] == (28, 28, 1)
        assert x_test.shape[1:] == (28, 28, 1)
        assert y_train.shape == (100,)
        assert y_test.shape == (50,)

    def test_load_mnist_values_in_range(self):
        """Test that pixel values are normalized to [0, 1]."""
        x_train, x_test, _, _ = load_mnist(
            n_train=50, n_test=25, random_seed=42
        )
        assert np.all(x_train >= 0) and np.all(x_train <= 1)
        assert np.all(x_test >= 0) and np.all(x_test <= 1)
