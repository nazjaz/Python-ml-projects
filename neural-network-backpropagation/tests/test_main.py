"""Tests for neural network backpropagation module."""

import numpy as np
import pytest

from src.main import (
    ActivationFunction,
    DenseLayer,
    NetworkRunner,
    NeuralNetwork,
    WeightInitializer,
    generate_classification_data,
    generate_regression_data,
    normalize_features,
)


class TestActivationFunction:
    """Test cases for activation functions and their derivatives."""

    def test_sigmoid_output_range(self):
        """Test that sigmoid output is bounded between 0 and 1."""
        z = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        output, derivative = ActivationFunction.sigmoid(z)

        assert np.all(output > 0)
        assert np.all(output < 1)
        assert output.shape == z.shape
        assert derivative.shape == z.shape

    def test_sigmoid_at_zero(self):
        """Test that sigmoid(0) = 0.5."""
        z = np.array([0.0])
        output, _ = ActivationFunction.sigmoid(z)
        assert abs(output[0] - 0.5) < 1e-10

    def test_sigmoid_derivative_at_zero(self):
        """Test that sigmoid derivative at z=0 is 0.25."""
        z = np.array([0.0])
        _, derivative = ActivationFunction.sigmoid(z)
        assert abs(derivative[0] - 0.25) < 1e-10

    def test_sigmoid_numerical_stability(self):
        """Test sigmoid with extreme values does not produce NaN or Inf."""
        z = np.array([-1000.0, 1000.0])
        output, derivative = ActivationFunction.sigmoid(z)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        assert not np.any(np.isnan(derivative))

    def test_tanh_output_range(self):
        """Test that tanh output is bounded between -1 and 1."""
        z = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        output, derivative = ActivationFunction.tanh(z)

        assert np.all(output >= -1)
        assert np.all(output <= 1)
        assert output.shape == z.shape

    def test_tanh_at_zero(self):
        """Test that tanh(0) = 0 and derivative at 0 is 1."""
        z = np.array([0.0])
        output, derivative = ActivationFunction.tanh(z)
        assert abs(output[0]) < 1e-10
        assert abs(derivative[0] - 1.0) < 1e-10

    def test_relu_positive_input(self):
        """Test that ReLU preserves positive values."""
        z = np.array([1.0, 2.0, 3.0])
        output, derivative = ActivationFunction.relu(z)

        np.testing.assert_array_equal(output, z)
        np.testing.assert_array_equal(derivative, np.ones_like(z))

    def test_relu_negative_input(self):
        """Test that ReLU zeroes out negative values."""
        z = np.array([-1.0, -2.0, -3.0])
        output, derivative = ActivationFunction.relu(z)

        np.testing.assert_array_equal(output, np.zeros_like(z))
        np.testing.assert_array_equal(derivative, np.zeros_like(z))

    def test_relu_mixed_input(self):
        """Test ReLU with mixed positive and negative values."""
        z = np.array([-2.0, 0.0, 3.0])
        output, derivative = ActivationFunction.relu(z)

        expected_output = np.array([0.0, 0.0, 3.0])
        expected_deriv = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(derivative, expected_deriv)

    def test_leaky_relu_positive_input(self):
        """Test that Leaky ReLU preserves positive values."""
        z = np.array([1.0, 2.0, 3.0])
        output, derivative = ActivationFunction.leaky_relu(z, alpha=0.01)

        np.testing.assert_array_equal(output, z)
        np.testing.assert_array_equal(derivative, np.ones_like(z))

    def test_leaky_relu_negative_input(self):
        """Test that Leaky ReLU scales negative values by alpha."""
        z = np.array([-1.0, -2.0])
        alpha = 0.01
        output, derivative = ActivationFunction.leaky_relu(z, alpha=alpha)

        expected_output = np.array([-0.01, -0.02])
        np.testing.assert_allclose(output, expected_output)
        np.testing.assert_array_equal(derivative, np.full_like(z, alpha))

    def test_softmax_sums_to_one(self):
        """Test that softmax output sums to 1 for each sample."""
        z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output, derivative = ActivationFunction.softmax(z)

        np.testing.assert_allclose(
            np.sum(output, axis=1), np.ones(2), atol=1e-10
        )

    def test_softmax_all_positive(self):
        """Test that softmax output is strictly positive."""
        z = np.array([[-100.0, 0.0, 100.0]])
        output, _ = ActivationFunction.softmax(z)

        assert np.all(output > 0)

    def test_softmax_numerical_stability(self):
        """Test softmax with large values does not overflow."""
        z = np.array([[1000.0, 1001.0, 1002.0]])
        output, _ = ActivationFunction.softmax(z)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
        np.testing.assert_allclose(np.sum(output), 1.0, atol=1e-10)

    def test_linear_passthrough(self):
        """Test that linear activation returns input unchanged."""
        z = np.array([-2.0, 0.0, 3.0])
        output, derivative = ActivationFunction.linear(z)

        np.testing.assert_array_equal(output, z)
        np.testing.assert_array_equal(derivative, np.ones_like(z))


class TestWeightInitializer:
    """Test cases for weight initialization strategies."""

    def test_xavier_shape(self):
        """Test Xavier initialization produces correct shape."""
        weights = WeightInitializer.xavier(10, 5)
        assert weights.shape == (10, 5)

    def test_xavier_variance(self):
        """Test Xavier initialization has approximately correct variance."""
        np.random.seed(42)
        fan_in, fan_out = 1000, 500
        weights = WeightInitializer.xavier(fan_in, fan_out)

        expected_var = 2.0 / (fan_in + fan_out)
        actual_var = np.var(weights)
        assert abs(actual_var - expected_var) < 0.001

    def test_he_shape(self):
        """Test He initialization produces correct shape."""
        weights = WeightInitializer.he(10, 5)
        assert weights.shape == (10, 5)

    def test_he_variance(self):
        """Test He initialization has approximately correct variance."""
        np.random.seed(42)
        fan_in, fan_out = 1000, 500
        weights = WeightInitializer.he(fan_in, fan_out)

        expected_var = 2.0 / fan_in
        actual_var = np.var(weights)
        assert abs(actual_var - expected_var) < 0.001

    def test_random_normal_shape(self):
        """Test random normal initialization produces correct shape."""
        weights = WeightInitializer.random_normal(10, 5)
        assert weights.shape == (10, 5)

    def test_random_normal_small_variance(self):
        """Test random normal has small variance around 0.01^2."""
        np.random.seed(42)
        weights = WeightInitializer.random_normal(1000, 500)
        assert np.var(weights) < 0.001


class TestDenseLayer:
    """Test cases for individual dense layers."""

    def test_layer_creation(self):
        """Test layer initializes with correct shapes."""
        layer = DenseLayer(10, 5, activation="relu", weight_init="he")

        assert layer.weights.shape == (10, 5)
        assert layer.biases.shape == (1, 5)
        assert layer.activation_name == "relu"

    def test_layer_forward_shape(self):
        """Test forward pass produces correct output shape."""
        layer = DenseLayer(10, 5, activation="relu")
        inputs = np.random.randn(32, 10)
        output = layer.forward(inputs)

        assert output.shape == (32, 5)

    def test_layer_forward_invalid_input_dimension(self):
        """Test forward pass raises error for wrong input dimension."""
        layer = DenseLayer(10, 5, activation="relu")
        inputs = np.random.randn(32, 8)

        with pytest.raises(ValueError, match="Input dimension"):
            layer.forward(inputs)

    def test_layer_backward_shape(self):
        """Test backward pass produces correct gradient shape."""
        layer = DenseLayer(10, 5, activation="relu")
        inputs = np.random.randn(32, 10)

        layer.forward(inputs)
        upstream = np.random.randn(32, 5)
        gradient = layer.backward(upstream, learning_rate=0.01)

        assert gradient.shape == (32, 10)

    def test_layer_backward_without_forward_raises(self):
        """Test backward pass raises error if forward was not called."""
        layer = DenseLayer(10, 5, activation="relu")
        upstream = np.random.randn(32, 5)

        with pytest.raises(RuntimeError, match="Forward pass must be called"):
            layer.backward(upstream, learning_rate=0.01)

    def test_layer_weights_update(self):
        """Test that backward pass modifies weights."""
        np.random.seed(42)
        layer = DenseLayer(10, 5, activation="relu")
        inputs = np.random.randn(32, 10)

        layer.forward(inputs)
        weights_before = layer.weights.copy()

        upstream = np.random.randn(32, 5)
        layer.backward(upstream, learning_rate=0.1)

        assert not np.array_equal(layer.weights, weights_before)

    def test_invalid_activation_raises(self):
        """Test layer creation with unknown activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            DenseLayer(10, 5, activation="invalid")

    def test_invalid_weight_init_raises(self):
        """Test layer creation with unknown weight init raises error."""
        with pytest.raises(ValueError, match="Unknown weight init"):
            DenseLayer(10, 5, weight_init="invalid")


class TestNeuralNetwork:
    """Test cases for the neural network class."""

    def test_network_creation(self):
        """Test network initializes with correct number of layers."""
        network = NeuralNetwork(
            layer_sizes=[10, 64, 32, 3],
            activations=["relu", "relu", "softmax"],
            loss="cross_entropy",
        )

        assert len(network.layers) == 3
        assert network.layers[0].input_size == 10
        assert network.layers[0].output_size == 64
        assert network.layers[1].input_size == 64
        assert network.layers[1].output_size == 32
        assert network.layers[2].input_size == 32
        assert network.layers[2].output_size == 3

    def test_network_forward_shape(self):
        """Test forward pass produces correct output shape."""
        network = NeuralNetwork(
            layer_sizes=[10, 64, 3],
            activations=["relu", "softmax"],
        )

        inputs = np.random.randn(32, 10)
        output = network.forward(inputs)

        assert output.shape == (32, 3)

    def test_network_invalid_layer_sizes(self):
        """Test network creation with too few layers raises error."""
        with pytest.raises(ValueError, match="at least an input and output"):
            NeuralNetwork(layer_sizes=[10], activations=[])

    def test_network_activation_count_mismatch(self):
        """Test network creation with wrong activation count raises error."""
        with pytest.raises(ValueError, match="Expected 2 activations"):
            NeuralNetwork(
                layer_sizes=[10, 5, 3],
                activations=["relu"],
            )

    def test_network_invalid_loss(self):
        """Test network creation with unknown loss raises error."""
        with pytest.raises(ValueError, match="Unknown loss"):
            NeuralNetwork(
                layer_sizes=[10, 3],
                activations=["softmax"],
                loss="invalid",
            )

    def test_network_predict_classification(self):
        """Test predict returns integer labels for classification."""
        np.random.seed(42)
        network = NeuralNetwork(
            layer_sizes=[10, 32, 3],
            activations=["relu", "softmax"],
            loss="cross_entropy",
        )

        inputs = np.random.randn(20, 10)
        predictions = network.predict(inputs)

        assert predictions.shape == (20,)
        assert predictions.dtype in (np.int64, np.intp)
        assert np.all(predictions >= 0)
        assert np.all(predictions < 3)

    def test_network_predict_proba(self):
        """Test predict_proba returns valid probabilities."""
        np.random.seed(42)
        network = NeuralNetwork(
            layer_sizes=[10, 32, 3],
            activations=["relu", "softmax"],
            loss="cross_entropy",
        )

        inputs = np.random.randn(20, 10)
        proba = network.predict_proba(inputs)

        assert proba.shape == (20, 3)
        np.testing.assert_allclose(
            np.sum(proba, axis=1), np.ones(20), atol=1e-6
        )

    def test_network_summary(self):
        """Test that summary returns a non-empty string."""
        network = NeuralNetwork(
            layer_sizes=[10, 64, 32, 3],
            activations=["relu", "relu", "softmax"],
        )

        summary = network.summary()
        assert isinstance(summary, str)
        assert "Dense" in summary
        assert "Total trainable parameters" in summary

    def test_network_training_reduces_loss(self):
        """Test that training reduces loss over epochs."""
        np.random.seed(42)
        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=200, n_features=10, n_classes=3, random_seed=42
        )
        x_train, x_test = normalize_features(x_train, x_test)

        network = NeuralNetwork(
            layer_sizes=[10, 32, 3],
            activations=["relu", "softmax"],
            loss="cross_entropy",
        )

        history = network.train(
            x_train,
            y_train,
            epochs=50,
            learning_rate=0.01,
            batch_size=32,
            verbose=False,
        )

        assert history["loss"][-1] < history["loss"][0]

    def test_network_training_with_validation(self):
        """Test training with validation data records val metrics."""
        np.random.seed(42)
        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=200, n_features=10, n_classes=3, random_seed=42
        )
        x_train, x_test = normalize_features(x_train, x_test)

        network = NeuralNetwork(
            layer_sizes=[10, 32, 3],
            activations=["relu", "softmax"],
            loss="cross_entropy",
        )

        history = network.train(
            x_train,
            y_train,
            epochs=20,
            learning_rate=0.01,
            batch_size=32,
            validation_data=(x_test, y_test),
            verbose=False,
        )

        assert "val_loss" in history
        assert "val_accuracy" in history
        assert len(history["val_loss"]) == 20

    def test_network_evaluate(self):
        """Test evaluate returns loss and accuracy."""
        np.random.seed(42)
        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=200, n_features=10, n_classes=3, random_seed=42
        )
        x_train, x_test = normalize_features(x_train, x_test)

        network = NeuralNetwork(
            layer_sizes=[10, 32, 3],
            activations=["relu", "softmax"],
            loss="cross_entropy",
        )

        network.train(
            x_train, y_train, epochs=20,
            learning_rate=0.01, verbose=False,
        )

        results = network.evaluate(x_test, y_test)
        assert "loss" in results
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_network_regression(self):
        """Test network for regression task with MSE loss."""
        np.random.seed(42)
        x_train, x_test, y_train, y_test = generate_regression_data(
            n_samples=200, n_features=5, random_seed=42
        )
        x_train, x_test = normalize_features(x_train, x_test)

        network = NeuralNetwork(
            layer_sizes=[5, 32, 16, 1],
            activations=["relu", "relu", "linear"],
            loss="mse",
        )

        history = network.train(
            x_train,
            y_train,
            epochs=50,
            learning_rate=0.001,
            batch_size=32,
            verbose=False,
        )

        assert history["loss"][-1] < history["loss"][0]

    def test_network_binary_classification(self):
        """Test network for binary classification with BCE loss."""
        np.random.seed(42)
        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=200, n_features=10, n_classes=2, random_seed=42
        )
        x_train, x_test = normalize_features(x_train, x_test)

        network = NeuralNetwork(
            layer_sizes=[10, 32, 1],
            activations=["relu", "sigmoid"],
            loss="binary_cross_entropy",
        )

        history = network.train(
            x_train,
            y_train,
            epochs=50,
            learning_rate=0.01,
            batch_size=32,
            verbose=False,
        )

        predictions = network.predict(x_test)
        assert np.all((predictions == 0) | (predictions == 1))

    def test_network_multiple_hidden_layers(self):
        """Test network with four hidden layers trains successfully."""
        np.random.seed(42)
        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=300, n_features=20, n_classes=5, random_seed=42
        )
        x_train, x_test = normalize_features(x_train, x_test)

        network = NeuralNetwork(
            layer_sizes=[20, 128, 64, 32, 16, 5],
            activations=["relu", "relu", "relu", "relu", "softmax"],
            loss="cross_entropy",
        )

        history = network.train(
            x_train,
            y_train,
            epochs=30,
            learning_rate=0.01,
            batch_size=32,
            verbose=False,
        )

        assert len(network.layers) == 5
        assert history["loss"][-1] < history["loss"][0]

    def test_network_different_activations(self):
        """Test network with mixed activation functions per layer."""
        np.random.seed(42)
        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=200, n_features=10, n_classes=3, random_seed=42
        )
        x_train, x_test = normalize_features(x_train, x_test)

        network = NeuralNetwork(
            layer_sizes=[10, 32, 16, 3],
            activations=["tanh", "relu", "softmax"],
            loss="cross_entropy",
            weight_init="xavier",
        )

        history = network.train(
            x_train,
            y_train,
            epochs=30,
            learning_rate=0.01,
            batch_size=32,
            verbose=False,
        )

        assert history["loss"][-1] < history["loss"][0]

    def test_network_get_training_history(self):
        """Test that training history is retrievable after training."""
        np.random.seed(42)
        network = NeuralNetwork(
            layer_sizes=[5, 10, 2],
            activations=["relu", "softmax"],
            loss="cross_entropy",
        )

        x = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        network.train(x, y, epochs=5, verbose=False)
        history = network.get_training_history()

        assert "loss" in history
        assert len(history["loss"]) == 5


class TestDataGeneration:
    """Test cases for data generation utilities."""

    def test_classification_data_shapes(self):
        """Test classification data has correct shapes."""
        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=100, n_features=10, n_classes=3
        )

        assert x_train.shape[1] == 10
        assert x_test.shape[1] == 10
        assert x_train.shape[0] + x_test.shape[0] <= 100
        assert len(y_train) == x_train.shape[0]
        assert len(y_test) == x_test.shape[0]

    def test_classification_data_labels(self):
        """Test classification labels are in valid range."""
        _, _, y_train, y_test = generate_classification_data(
            n_samples=300, n_features=5, n_classes=4
        )

        all_labels = np.concatenate([y_train, y_test])
        assert np.all(all_labels >= 0)
        assert np.all(all_labels < 4)

    def test_regression_data_shapes(self):
        """Test regression data has correct shapes."""
        x_train, x_test, y_train, y_test = generate_regression_data(
            n_samples=100, n_features=5
        )

        assert x_train.shape[1] == 5
        assert x_test.shape[1] == 5
        assert y_train.ndim == 2
        assert y_train.shape[1] == 1

    def test_data_reproducibility(self):
        """Test that same seed produces identical data."""
        data_1 = generate_classification_data(random_seed=123)
        data_2 = generate_classification_data(random_seed=123)

        np.testing.assert_array_equal(data_1[0], data_2[0])
        np.testing.assert_array_equal(data_1[2], data_2[2])


class TestNormalization:
    """Test cases for feature normalization."""

    def test_normalize_zero_mean(self):
        """Test normalized training data has approximately zero mean."""
        x_train = np.random.randn(100, 5) * 10 + 50
        x_test = np.random.randn(20, 5) * 10 + 50

        x_train_norm, _ = normalize_features(x_train, x_test)
        np.testing.assert_allclose(
            np.mean(x_train_norm, axis=0), np.zeros(5), atol=1e-10
        )

    def test_normalize_unit_variance(self):
        """Test normalized training data has approximately unit variance."""
        x_train = np.random.randn(1000, 5) * 10 + 50
        x_test = np.random.randn(20, 5) * 10 + 50

        x_train_norm, _ = normalize_features(x_train, x_test)
        np.testing.assert_allclose(
            np.std(x_train_norm, axis=0), np.ones(5), atol=0.1
        )

    def test_normalize_constant_feature(self):
        """Test normalization handles constant features without errors."""
        x_train = np.ones((50, 3))
        x_test = np.ones((10, 3))

        x_train_norm, x_test_norm = normalize_features(x_train, x_test)
        assert not np.any(np.isnan(x_train_norm))
        assert not np.any(np.isinf(x_train_norm))


class TestNetworkRunner:
    """Test cases for the NetworkRunner orchestrator."""

    def test_runner_initialization(self):
        """Test runner can be initialized without config file."""
        runner = NetworkRunner()
        assert runner is not None

    def test_runner_classification(self):
        """Test runner executes classification experiment."""
        runner = NetworkRunner()
        results = runner.run_classification()

        assert "loss" in results
        assert "accuracy" in results
        assert results["loss"] >= 0
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_runner_regression(self):
        """Test runner executes regression experiment."""
        runner = NetworkRunner()
        results = runner.run_regression()

        assert "loss" in results
        assert results["loss"] >= 0


class TestGradientNumericalVerification:
    """Verify analytical gradients against numerical approximations."""

    def test_dense_layer_gradient_accuracy(self):
        """Test layer gradient using finite differences."""
        np.random.seed(42)
        layer = DenseLayer(5, 3, activation="sigmoid")
        inputs = np.random.randn(10, 5)
        epsilon = 1e-5

        layer.forward(inputs)
        upstream = np.random.randn(10, 3)
        analytical_grad = layer.backward(upstream, learning_rate=0.0)

        numerical_grad = np.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                inputs_plus = inputs.copy()
                inputs_plus[i, j] += epsilon
                out_plus = layer.forward(inputs_plus)
                loss_plus = np.sum(upstream * out_plus)

                inputs_minus = inputs.copy()
                inputs_minus[i, j] -= epsilon
                out_minus = layer.forward(inputs_minus)
                loss_minus = np.sum(upstream * out_minus)

                numerical_grad[i, j] = (
                    (loss_plus - loss_minus) / (2 * epsilon)
                )

        np.testing.assert_allclose(
            analytical_grad, numerical_grad, rtol=1e-3, atol=1e-5
        )

    def test_mse_loss_gradient(self):
        """Test MSE loss gradient using finite differences."""
        np.random.seed(42)
        network = NeuralNetwork(
            layer_sizes=[5, 10, 1],
            activations=["relu", "linear"],
            loss="mse",
        )

        inputs = np.random.randn(20, 5)
        targets = np.random.randn(20, 1)
        epsilon = 1e-5

        output = network.forward(inputs)
        _, analytical_grad = network._compute_loss(targets, output)

        numerical_grad = np.zeros_like(output)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output_plus = output.copy()
                output_plus[i, j] += epsilon
                loss_plus, _ = network._compute_loss(targets, output_plus)

                output_minus = output.copy()
                output_minus[i, j] -= epsilon
                loss_minus, _ = network._compute_loss(targets, output_minus)

                numerical_grad[i, j] = (
                    (loss_plus - loss_minus) / (2 * epsilon)
                )

        np.testing.assert_allclose(
            analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6
        )
