"""Neural Network from Scratch with Backpropagation.

This module provides a complete implementation of a feedforward neural network
built from scratch using only NumPy. It supports multiple hidden layers,
configurable activation functions, weight initialization strategies, and
gradient-based optimization via backpropagation.
"""

import argparse
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ActivationFunction:
    """Collection of activation functions with their derivatives.

    Each method returns a tuple of (activation_output, derivative) to
    support both forward and backward passes efficiently.
    """

    @staticmethod
    def sigmoid(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sigmoid activation and its derivative.

        Sigmoid maps inputs to the range (0, 1). Numerically stable
        implementation using clipping to avoid overflow in exp().

        Args:
            z: Pre-activation input array of any shape.

        Returns:
            Tuple containing:
                - output: Sigmoid activation, same shape as z
                - derivative: Element-wise derivative, same shape as z
        """
        z_clipped = np.clip(z, -500, 500)
        output = 1.0 / (1.0 + np.exp(-z_clipped))
        derivative = output * (1.0 - output)
        return output, derivative

    @staticmethod
    def tanh(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tanh activation and its derivative.

        Tanh maps inputs to the range (-1, 1), centered at zero,
        which often leads to faster convergence than sigmoid.

        Args:
            z: Pre-activation input array of any shape.

        Returns:
            Tuple containing:
                - output: Tanh activation, same shape as z
                - derivative: Element-wise derivative, same shape as z
        """
        output = np.tanh(z)
        derivative = 1.0 - output ** 2
        return output, derivative

    @staticmethod
    def relu(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ReLU activation and its derivative.

        ReLU sets negative values to zero, preserving positive values.
        This avoids vanishing gradient for positive inputs but can
        cause dead neurons for persistently negative inputs.

        Args:
            z: Pre-activation input array of any shape.

        Returns:
            Tuple containing:
                - output: ReLU activation, same shape as z
                - derivative: Element-wise derivative, same shape as z
        """
        output = np.maximum(0, z)
        derivative = (z > 0).astype(float)
        return output, derivative

    @staticmethod
    def leaky_relu(
        z: np.ndarray, alpha: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Leaky ReLU activation and its derivative.

        Leaky ReLU allows a small gradient for negative inputs,
        preventing the dead neuron problem that standard ReLU has.

        Args:
            z: Pre-activation input array of any shape.
            alpha: Slope for negative values (default: 0.01).

        Returns:
            Tuple containing:
                - output: Leaky ReLU activation, same shape as z
                - derivative: Element-wise derivative, same shape as z
        """
        output = np.where(z > 0, z, alpha * z)
        derivative = np.where(z > 0, 1.0, alpha)
        return output, derivative

    @staticmethod
    def softmax(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute softmax activation and its Jacobian diagonal approximation.

        Softmax converts raw scores into probability distributions.
        Numerically stable implementation subtracts the max value per
        sample before exponentiation to prevent overflow.

        The derivative returned is the diagonal of the Jacobian matrix
        (s_i * (1 - s_i)), used as an approximation during backprop.
        When paired with cross-entropy loss, the combined gradient
        simplifies to (softmax_output - y_true), bypassing this derivative.

        Args:
            z: Pre-activation input, shape (n_samples, n_classes).

        Returns:
            Tuple containing:
                - output: Softmax probabilities, same shape as z
                - derivative: Diagonal Jacobian approximation, same shape as z
        """
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        derivative = output * (1.0 - output)
        return output, derivative

    @staticmethod
    def linear(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute linear (identity) activation and its derivative.

        Passes the input through unchanged. Used for regression
        output layers where no squashing is needed.

        Args:
            z: Pre-activation input array of any shape.

        Returns:
            Tuple containing:
                - output: Same as input z
                - derivative: Array of ones, same shape as z
        """
        return z.copy(), np.ones_like(z)


ACTIVATION_REGISTRY: Dict[
    str, Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]
] = {
    "sigmoid": ActivationFunction.sigmoid,
    "tanh": ActivationFunction.tanh,
    "relu": ActivationFunction.relu,
    "leaky_relu": ActivationFunction.leaky_relu,
    "softmax": ActivationFunction.softmax,
    "linear": ActivationFunction.linear,
}


class WeightInitializer:
    """Weight initialization strategies for neural network layers.

    Proper weight initialization is critical for training stability.
    Different strategies are suited to different activation functions.
    """

    @staticmethod
    def xavier(fan_in: int, fan_out: int) -> np.ndarray:
        """Xavier (Glorot) initialization for sigmoid and tanh activations.

        Draws weights from a normal distribution with variance
        2 / (fan_in + fan_out). This keeps signal variance consistent
        across layers when using symmetric activations.

        Args:
            fan_in: Number of input neurons to the layer.
            fan_out: Number of output neurons from the layer.

        Returns:
            Weight matrix of shape (fan_in, fan_out).
        """
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_in, fan_out) * std

    @staticmethod
    def he(fan_in: int, fan_out: int) -> np.ndarray:
        """He initialization for ReLU-family activations.

        Draws weights from a normal distribution with variance
        2 / fan_in. Accounts for the fact that ReLU zeroes out
        roughly half of the activations.

        Args:
            fan_in: Number of input neurons to the layer.
            fan_out: Number of output neurons from the layer.

        Returns:
            Weight matrix of shape (fan_in, fan_out).
        """
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_in, fan_out) * std

    @staticmethod
    def random_normal(fan_in: int, fan_out: int) -> np.ndarray:
        """Simple random normal initialization with small variance.

        Draws from N(0, 0.01). Useful as a baseline but may cause
        vanishing gradients in deep networks.

        Args:
            fan_in: Number of input neurons to the layer.
            fan_out: Number of output neurons from the layer.

        Returns:
            Weight matrix of shape (fan_in, fan_out).
        """
        return np.random.randn(fan_in, fan_out) * 0.01


INIT_REGISTRY: Dict[str, Callable[[int, int], np.ndarray]] = {
    "xavier": WeightInitializer.xavier,
    "he": WeightInitializer.he,
    "random_normal": WeightInitializer.random_normal,
}


class DenseLayer:
    """A single fully connected layer with activation.

    Stores weights, biases, and caches needed for backpropagation.
    Supports forward pass computation and gradient computation.

    Attributes:
        weights: Weight matrix of shape (input_size, output_size).
        biases: Bias vector of shape (1, output_size).
        activation_name: Name of the activation function.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        weight_init: str = "he",
    ) -> None:
        """Initialize a dense layer with weights and activation.

        Args:
            input_size: Number of input features.
            output_size: Number of output neurons.
            activation: Activation function name (default: "relu").
            weight_init: Weight initialization strategy (default: "he").

        Raises:
            ValueError: If activation or weight_init is not recognized.
        """
        if activation not in ACTIVATION_REGISTRY:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Available: {list(ACTIVATION_REGISTRY.keys())}"
            )
        if weight_init not in INIT_REGISTRY:
            raise ValueError(
                f"Unknown weight init '{weight_init}'. "
                f"Available: {list(INIT_REGISTRY.keys())}"
            )

        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self._activation_fn = ACTIVATION_REGISTRY[activation]
        self.weights = INIT_REGISTRY[weight_init](input_size, output_size)
        self.biases = np.zeros((1, output_size))

        # Cache for backpropagation
        self._input_cache: Optional[np.ndarray] = None
        self._z_cache: Optional[np.ndarray] = None
        self._activation_cache: Optional[np.ndarray] = None
        self._derivative_cache: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute forward pass through the layer.

        Computes z = inputs @ weights + biases, then applies
        the activation function. Caches intermediate values
        for use during backpropagation.

        Args:
            inputs: Input array of shape (n_samples, input_size).

        Returns:
            Activated output of shape (n_samples, output_size).

        Raises:
            ValueError: If input dimension does not match layer size.
        """
        if inputs.shape[1] != self.input_size:
            raise ValueError(
                f"Input dimension {inputs.shape[1]} does not match "
                f"layer input size {self.input_size}"
            )

        self._input_cache = inputs
        self._z_cache = inputs @ self.weights + self.biases
        self._activation_cache, self._derivative_cache = self._activation_fn(
            self._z_cache
        )

        return self._activation_cache

    def backward(
        self, upstream_gradient: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        """Compute backward pass and update weights.

        Applies the chain rule to compute gradients with respect
        to inputs, weights, and biases. Updates parameters using
        the provided learning rate.

        Args:
            upstream_gradient: Gradient from the next layer,
                shape (n_samples, output_size).
            learning_rate: Step size for gradient descent.

        Returns:
            Gradient with respect to this layer's inputs,
            shape (n_samples, input_size).

        Raises:
            RuntimeError: If forward pass has not been called first.
        """
        if self._input_cache is None:
            raise RuntimeError(
                "Forward pass must be called before backward pass"
            )

        n_samples = upstream_gradient.shape[0]

        # Gradient through activation function
        delta = upstream_gradient * self._derivative_cache

        # Gradients for weights and biases
        weight_gradient = self._input_cache.T @ delta / n_samples
        bias_gradient = np.mean(delta, axis=0, keepdims=True)

        # Gradient to pass to previous layer
        input_gradient = delta @ self.weights.T

        # Parameter update via gradient descent
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient

        return input_gradient


class NeuralNetwork:
    """Feedforward neural network with configurable architecture.

    Supports multiple hidden layers with different activation functions,
    various loss functions, and training via mini-batch gradient descent
    with backpropagation.

    Attributes:
        layers: List of DenseLayer objects forming the network.
        loss_name: Name of the loss function used for training.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        loss: str = "cross_entropy",
        weight_init: str = "he",
    ) -> None:
        """Initialize the neural network architecture.

        Args:
            layer_sizes: List of integers specifying the number of
                neurons in each layer, starting with the input layer.
                Example: [784, 128, 64, 10] creates a network with
                784 inputs, two hidden layers, and 10 outputs.
            activations: List of activation function names for each
                layer transition. Length must be len(layer_sizes) - 1.
            loss: Loss function name ("cross_entropy", "mse",
                "binary_cross_entropy"). Default: "cross_entropy".
            weight_init: Weight initialization strategy for all layers.
                Default: "he".

        Raises:
            ValueError: If layer_sizes or activations have invalid
                lengths or contain unrecognized names.
        """
        if len(layer_sizes) < 2:
            raise ValueError(
                "Network must have at least an input and output layer"
            )
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Expected {len(layer_sizes) - 1} activations, "
                f"got {len(activations)}"
            )
        if loss not in ("cross_entropy", "mse", "binary_cross_entropy"):
            raise ValueError(
                f"Unknown loss '{loss}'. Available: "
                "cross_entropy, mse, binary_cross_entropy"
            )

        self.loss_name = loss
        self.layers: List[DenseLayer] = []
        self._training_history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
        }

        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init,
            )
            self.layers.append(layer)

        logger.info(
            "Network initialized with %d layers: %s",
            len(self.layers),
            " -> ".join(str(s) for s in layer_sizes),
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Perform forward propagation through all layers.

        Args:
            inputs: Input data of shape (n_samples, n_features).

        Returns:
            Network output of shape (n_samples, output_size).
        """
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def _compute_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute loss value and initial gradient for backpropagation.

        Args:
            y_true: Ground truth labels. For cross-entropy, one-hot
                encoded shape (n_samples, n_classes). For MSE,
                shape (n_samples, output_size).
            y_pred: Network output predictions, same shape as y_true.

        Returns:
            Tuple containing:
                - loss: Scalar loss value.
                - gradient: Initial gradient for backpropagation,
                  same shape as y_pred.
        """
        n_samples = y_true.shape[0]
        epsilon = 1e-15

        if self.loss_name == "cross_entropy":
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.sum(y_true * np.log(y_pred_clipped)) / n_samples
            # Combined softmax + cross-entropy gradient simplification
            gradient = (y_pred_clipped - y_true) / n_samples
            return float(loss), gradient

        if self.loss_name == "binary_cross_entropy":
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(
                y_true * np.log(y_pred_clipped)
                + (1 - y_true) * np.log(1 - y_pred_clipped)
            )
            gradient = (
                -(y_true / y_pred_clipped
                  - (1 - y_true) / (1 - y_pred_clipped))
                / n_samples
            )
            return float(loss), gradient

        # MSE loss
        error = y_pred - y_true
        loss = np.mean(error ** 2)
        gradient = 2 * error / n_samples
        return float(loss), gradient

    def _backward(
        self, loss_gradient: np.ndarray, learning_rate: float
    ) -> None:
        """Perform backpropagation through all layers.

        Propagates the loss gradient backward through each layer,
        updating weights and biases along the way.

        Args:
            loss_gradient: Gradient of the loss with respect to the
                network output, shape (n_samples, output_size).
            learning_rate: Step size for gradient descent updates.
        """
        gradient = loss_gradient
        for layer in reversed(self.layers):
            # For softmax + cross-entropy, skip activation derivative
            # on the output layer because the gradient already accounts
            # for the combined derivative
            if (
                layer is self.layers[-1]
                and layer.activation_name == "softmax"
                and self.loss_name == "cross_entropy"
            ):
                n_samples = gradient.shape[0]
                weight_grad = layer._input_cache.T @ gradient / n_samples
                bias_grad = np.mean(gradient, axis=0, keepdims=True)
                input_grad = gradient @ layer.weights.T
                layer.weights -= learning_rate * weight_grad
                layer.biases -= learning_rate * bias_grad
                gradient = input_grad
            else:
                gradient = layer.backward(gradient, learning_rate)

    def _one_hot_encode(
        self, labels: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """Convert integer labels to one-hot encoded vectors.

        Args:
            labels: Integer class labels of shape (n_samples,).
            n_classes: Total number of classes.

        Returns:
            One-hot matrix of shape (n_samples, n_classes).
        """
        one_hot = np.zeros((labels.shape[0], n_classes))
        one_hot[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
        return one_hot

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the network using mini-batch gradient descent.

        Args:
            x_train: Training features of shape (n_samples, n_features).
            y_train: Training labels. Integer labels for classification,
                continuous values for regression.
            epochs: Number of training epochs (default: 100).
            learning_rate: Learning rate for gradient descent (default: 0.01).
            batch_size: Mini-batch size (default: 32).
            validation_data: Optional tuple (x_val, y_val) for monitoring
                validation performance during training.
            verbose: Whether to log training progress (default: True).

        Returns:
            Dictionary containing training history with keys:
                - loss: List of training loss per epoch
                - accuracy: List of training accuracy per epoch
                - val_loss: List of validation loss per epoch (if provided)
                - val_accuracy: List of validation accuracy per epoch
        """
        n_samples = x_train.shape[0]
        output_size = self.layers[-1].output_size
        is_classification = self.loss_name in (
            "cross_entropy",
            "binary_cross_entropy",
        )

        if self.loss_name == "binary_cross_entropy":
            # Binary CE expects shape (n_samples, 1) with values 0/1
            y_encoded = y_train.copy().astype(float)
            if y_encoded.ndim == 1:
                y_encoded = y_encoded.reshape(-1, 1)
        elif self.loss_name == "cross_entropy" and y_train.ndim == 1:
            y_encoded = self._one_hot_encode(y_train, output_size)
        else:
            y_encoded = y_train.copy()
            if y_encoded.ndim == 1:
                y_encoded = y_encoded.reshape(-1, 1)

        history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
        }
        if validation_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_encoded[indices]

            epoch_losses: List[float] = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                output = self.forward(x_batch)
                loss, gradient = self._compute_loss(y_batch, output)
                self._backward(gradient, learning_rate)

                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses))
            history["loss"].append(avg_loss)

            if is_classification:
                predictions = self.predict(x_train)
                if y_train.ndim == 1:
                    accuracy = float(np.mean(predictions == y_train))
                else:
                    accuracy = float(
                        np.mean(predictions == np.argmax(y_train, axis=1))
                    )
                history["accuracy"].append(accuracy)
            else:
                history["accuracy"].append(0.0)

            if validation_data is not None:
                x_val, y_val = validation_data
                val_output = self.forward(x_val)

                if self.loss_name == "binary_cross_entropy":
                    y_val_enc = y_val.copy().astype(float)
                    if y_val_enc.ndim == 1:
                        y_val_enc = y_val_enc.reshape(-1, 1)
                elif self.loss_name == "cross_entropy" and y_val.ndim == 1:
                    y_val_enc = self._one_hot_encode(y_val, output_size)
                else:
                    y_val_enc = y_val.copy()
                    if y_val_enc.ndim == 1:
                        y_val_enc = y_val_enc.reshape(-1, 1)

                val_loss, _ = self._compute_loss(y_val_enc, val_output)
                history["val_loss"].append(val_loss)

                if is_classification:
                    val_preds = self.predict(x_val)
                    if y_val.ndim == 1:
                        val_acc = float(np.mean(val_preds == y_val))
                    else:
                        val_acc = float(
                            np.mean(
                                val_preds == np.argmax(y_val, axis=1)
                            )
                        )
                    history["val_accuracy"].append(val_acc)
                else:
                    history["val_accuracy"].append(0.0)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}"
                if is_classification:
                    msg += f" - accuracy: {history['accuracy'][-1]:.4f}"
                if validation_data is not None:
                    msg += f" - val_loss: {history['val_loss'][-1]:.6f}"
                    if is_classification:
                        msg += (
                            f" - val_accuracy: "
                            f"{history['val_accuracy'][-1]:.4f}"
                        )
                logger.info(msg)
                if verbose:
                    print(msg)

        self._training_history = history
        return history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Generate predictions for the given inputs.

        For classification networks (cross-entropy loss), returns
        integer class labels. For regression, returns raw output.

        Args:
            inputs: Input data of shape (n_samples, n_features).

        Returns:
            Predictions array. For classification: integer labels
            of shape (n_samples,). For regression: continuous values
            of shape (n_samples, output_size).
        """
        output = self.forward(inputs)
        if self.loss_name in ("cross_entropy",):
            return np.argmax(output, axis=1)
        if self.loss_name == "binary_cross_entropy":
            return (output >= 0.5).astype(int).flatten()
        return output

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        """Generate probability predictions for classification tasks.

        Args:
            inputs: Input data of shape (n_samples, n_features).

        Returns:
            Probability array of shape (n_samples, n_classes).
        """
        return self.forward(inputs)

    def evaluate(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the network on test data.

        Computes loss and accuracy on the provided test set.

        Args:
            x_test: Test features of shape (n_samples, n_features).
            y_test: Test labels (integer labels for classification).

        Returns:
            Dictionary containing:
                - loss: Test loss value
                - accuracy: Test accuracy (classification only)
        """
        output = self.forward(x_test)
        output_size = self.layers[-1].output_size
        is_classification = self.loss_name in (
            "cross_entropy",
            "binary_cross_entropy",
        )

        if self.loss_name == "binary_cross_entropy":
            y_encoded = y_test.copy().astype(float)
            if y_encoded.ndim == 1:
                y_encoded = y_encoded.reshape(-1, 1)
        elif self.loss_name == "cross_entropy" and y_test.ndim == 1:
            y_encoded = self._one_hot_encode(y_test, output_size)
        else:
            y_encoded = y_test.copy()
            if y_encoded.ndim == 1:
                y_encoded = y_encoded.reshape(-1, 1)

        loss, _ = self._compute_loss(y_encoded, output)
        results: Dict[str, float] = {"loss": loss}

        if is_classification:
            predictions = self.predict(x_test)
            if y_test.ndim == 1:
                results["accuracy"] = float(np.mean(predictions == y_test))
            else:
                results["accuracy"] = float(
                    np.mean(predictions == np.argmax(y_test, axis=1))
                )

        return results

    def get_training_history(self) -> Dict[str, List[float]]:
        """Return the training history from the last training session.

        Returns:
            Dictionary containing lists of per-epoch metrics.
        """
        return self._training_history

    def summary(self) -> str:
        """Generate a text summary of the network architecture.

        Returns:
            Formatted string describing each layer's configuration
            and the total number of trainable parameters.
        """
        lines = ["Neural Network Summary", "=" * 55]
        lines.append(
            f"{'Layer':<10} {'Shape':<20} {'Activation':<12} {'Params':<10}"
        )
        lines.append("-" * 55)

        total_params = 0
        for i, layer in enumerate(self.layers):
            weight_params = layer.weights.size
            bias_params = layer.biases.size
            layer_params = weight_params + bias_params
            total_params += layer_params
            shape_str = f"({layer.input_size}, {layer.output_size})"
            lines.append(
                f"Dense {i:<4} {shape_str:<20} "
                f"{layer.activation_name:<12} {layer_params:<10}"
            )

        lines.append("-" * 55)
        lines.append(f"Total trainable parameters: {total_params}")
        lines.append("=" * 55)

        return "\n".join(lines)


def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset.

    Creates linearly separable clusters with added Gaussian noise,
    then splits into training and test sets.

    Args:
        n_samples: Total number of samples (default: 1000).
        n_features: Number of input features (default: 20).
        n_classes: Number of target classes (default: 3).
        random_seed: Seed for reproducibility (default: 42).

    Returns:
        Tuple of (x_train, x_test, y_train, y_test).
    """
    np.random.seed(random_seed)

    samples_per_class = n_samples // n_classes
    x_all = []
    y_all = []

    for cls in range(n_classes):
        center = np.random.randn(n_features) * 2
        samples = center + np.random.randn(samples_per_class, n_features)
        x_all.append(samples)
        y_all.append(np.full(samples_per_class, cls))

    x_data = np.vstack(x_all)
    y_data = np.concatenate(y_all)

    indices = np.random.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]

    split = int(0.8 * len(x_data))
    return x_data[:split], x_data[split:], y_data[:split], y_data[split:]


def generate_regression_data(
    n_samples: int = 500,
    n_features: int = 10,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic regression dataset.

    Creates data with a non-linear relationship between features
    and targets, then splits into training and test sets.

    Args:
        n_samples: Total number of samples (default: 500).
        n_features: Number of input features (default: 10).
        random_seed: Seed for reproducibility (default: 42).

    Returns:
        Tuple of (x_train, x_test, y_train, y_test).
    """
    np.random.seed(random_seed)

    x_data = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features) * 2
    y_data = (
        np.sin(x_data @ weights)
        + 0.5 * np.cos(x_data[:, 0] * x_data[:, 1])
        + 0.1 * np.random.randn(n_samples)
    )
    y_data = y_data.reshape(-1, 1)

    split = int(0.8 * n_samples)
    return x_data[:split], x_data[split:], y_data[:split], y_data[split:]


def normalize_features(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize features using training set statistics.

    Applies z-score normalization using mean and standard deviation
    computed from the training set only, preventing data leakage.

    Args:
        x_train: Training features to compute statistics from.
        x_test: Test features to normalize.

    Returns:
        Tuple of (normalized_train, normalized_test).
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    # Avoid division by zero for constant features
    std[std == 0] = 1.0

    return (x_train - mean) / std, (x_test - mean) / std


class NetworkRunner:
    """Orchestrates network creation, training, and evaluation.

    Loads configuration from YAML, builds the network, runs training,
    and reports results.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the runner with configuration.

        Args:
            config_path: Path to YAML configuration file (optional).
        """
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dictionary containing configuration values.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config or {}
        except FileNotFoundError:
            logger.warning(
                "Config file not found: %s, using defaults", config_path
            )
            return {}

    def _setup_logging(self) -> None:
        """Configure logging from configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/app.log")

        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)

    def run_classification(self) -> Dict[str, float]:
        """Run classification experiment using configuration.

        Generates synthetic data, builds and trains a network,
        and evaluates on the test set.

        Returns:
            Dictionary with test loss and accuracy.
        """
        data_config = self.config.get("data", {})
        network_config = self.config.get("network", {})
        training_config = self.config.get("training", {})

        n_samples = data_config.get("n_samples", 1000)
        n_features = data_config.get("n_features", 20)
        n_classes = data_config.get("n_classes", 3)
        seed = data_config.get("random_seed", 42)

        x_train, x_test, y_train, y_test = generate_classification_data(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            random_seed=seed,
        )
        x_train, x_test = normalize_features(x_train, x_test)

        hidden_sizes = network_config.get("hidden_layers", [64, 32])
        hidden_activations = network_config.get(
            "hidden_activations", ["relu", "relu"]
        )
        output_activation = network_config.get(
            "output_activation", "softmax"
        )
        weight_init = network_config.get("weight_init", "he")

        layer_sizes = [n_features] + hidden_sizes + [n_classes]
        activations = hidden_activations + [output_activation]

        network = NeuralNetwork(
            layer_sizes=layer_sizes,
            activations=activations,
            loss="cross_entropy",
            weight_init=weight_init,
        )

        print(network.summary())

        history = network.train(
            x_train,
            y_train,
            epochs=training_config.get("epochs", 100),
            learning_rate=training_config.get("learning_rate", 0.01),
            batch_size=training_config.get("batch_size", 32),
            validation_data=(x_test, y_test),
            verbose=True,
        )

        results = network.evaluate(x_test, y_test)
        logger.info(
            "Classification results - Loss: %.6f, Accuracy: %.4f",
            results["loss"],
            results.get("accuracy", 0.0),
        )

        return results

    def run_regression(self) -> Dict[str, float]:
        """Run regression experiment using configuration.

        Generates synthetic data, builds and trains a network,
        and evaluates on the test set.

        Returns:
            Dictionary with test loss.
        """
        data_config = self.config.get("data", {})
        network_config = self.config.get("network", {})
        training_config = self.config.get("training", {})

        n_samples = data_config.get("n_samples", 500)
        n_features = data_config.get("n_features", 10)
        seed = data_config.get("random_seed", 42)

        x_train, x_test, y_train, y_test = generate_regression_data(
            n_samples=n_samples,
            n_features=n_features,
            random_seed=seed,
        )
        x_train, x_test = normalize_features(x_train, x_test)

        hidden_sizes = network_config.get("hidden_layers", [64, 32])
        hidden_activations = network_config.get(
            "hidden_activations", ["relu", "relu"]
        )
        weight_init = network_config.get("weight_init", "he")

        layer_sizes = [n_features] + hidden_sizes + [1]
        activations = hidden_activations + ["linear"]

        network = NeuralNetwork(
            layer_sizes=layer_sizes,
            activations=activations,
            loss="mse",
            weight_init=weight_init,
        )

        print(network.summary())

        history = network.train(
            x_train,
            y_train,
            epochs=training_config.get("epochs", 100),
            learning_rate=training_config.get("learning_rate", 0.001),
            batch_size=training_config.get("batch_size", 32),
            validation_data=(x_test, y_test),
            verbose=True,
        )

        results = network.evaluate(x_test, y_test)
        logger.info("Regression results - Loss: %.6f", results["loss"])

        return results


def main() -> None:
    """Main entry point for the neural network trainer."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a neural network from scratch using backpropagation"
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="Task type (default: classification)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSON file for results",
    )

    args = parser.parse_args()

    runner = NetworkRunner(
        config_path=Path(args.config) if args.config else None
    )

    if args.task == "classification":
        results = runner.run_classification()
    else:
        results = runner.run_regression()

    print("\nFinal Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
