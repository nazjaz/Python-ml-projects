"""Convolutional Neural Network from Scratch for Image Classification.

This module provides a complete CNN implementation built from scratch using
only NumPy. It includes convolution layers, max pooling layers, ReLU
activation, and fully connected layers with backpropagation for training.
"""

import argparse
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _im2col(
    x: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    stride: int,
    pad: int,
) -> np.ndarray:
    """Reshape image patches into columns for efficient convolution.

    Extracts sliding windows from input and arranges them as columns.
    Used for both forward and backward convolution operations.

    Args:
        x: Input of shape (N, H, W, C).
        kernel_h: Filter height.
        kernel_w: Filter width.
        stride: Stride value.
        pad: Padding amount.

    Returns:
        Column matrix of shape (N * out_h * out_w, kernel_h * kernel_w * C).
    """
    n, h, w, c = x.shape
    out_h = (h + 2 * pad - kernel_h) // stride + 1
    out_w = (w + 2 * pad - kernel_w) // stride + 1

    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")

    col = np.zeros((n, out_h, out_w, kernel_h, kernel_w, c), dtype=x.dtype)
    for i in range(kernel_h):
        i_end = i + stride * out_h
        for j in range(kernel_w):
            j_end = j + stride * out_w
            col[:, :, :, i, j, :] = x[:, i:i_end:stride, j:j_end:stride, :]

    col = col.transpose(0, 1, 2, 4, 5, 3).reshape(
        n * out_h * out_w, kernel_h * kernel_w * c
    )
    return col


def _col2im(
    col: np.ndarray,
    n: int,
    h: int,
    w: int,
    c: int,
    kernel_h: int,
    kernel_w: int,
    stride: int,
    pad: int,
) -> np.ndarray:
    """Reshape columns back into image format for backward pass.

    Args:
        col: Column matrix from backward gradient.
        n, h, w, c: Original input dimensions.
        kernel_h, kernel_w: Filter dimensions.
        stride: Stride value.
        pad: Padding amount.

    Returns:
        Reconstructed gradient image of shape (N, H, W, C).
    """
    out_h = (h + 2 * pad - kernel_h) // stride + 1
    out_w = (w + 2 * pad - kernel_w) // stride + 1

    col_reshaped = col.reshape(
        n, out_h, out_w, kernel_w, kernel_h, c
    ).transpose(0, 1, 2, 4, 3, 5)

    x = np.zeros((n, h + 2 * pad, w + 2 * pad, c), dtype=col.dtype)
    for ph in range(kernel_h):
        for pw in range(kernel_w):
            h_slice = slice(ph, ph + stride * out_h, stride)
            w_slice = slice(pw, pw + stride * out_w, stride)
            x[:, h_slice, w_slice, :] += col_reshaped[:, :, :, ph, pw, :]

    if pad > 0:
        x = x[:, pad : h + pad, pad : w + pad, :]
    return x


class Conv2D:
    """2D convolution layer with configurable filters and stride.

    Applies convolution operation using im2col for efficient computation.
    Supports padding and stores intermediate values for backpropagation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """Initialize convolution layer with He initialization.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output filters.
            kernel_size: Height and width of convolution kernel.
            stride: Stride for spatial dimensions (default: 1).
            padding: Zero-padding amount (default: 0).
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_in)
        self.filters = np.random.randn(
            kernel_size, kernel_size, in_channels, out_channels
        ) * std
        self.biases = np.zeros(out_channels)

        self._input_cache: Optional[np.ndarray] = None
        self._col_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass through convolution.

        Args:
            x: Input of shape (N, H, W, C).

        Returns:
            Output of shape (N, out_H, out_W, out_channels).
        """
        self._input_cache = x
        n, h, w, c = x.shape
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        col = _im2col(
            x, self.kernel_size, self.kernel_size, self.stride, self.padding
        )
        self._col_cache = col

        w_flat = self.filters.reshape(
            self.kernel_size * self.kernel_size * self.in_channels,
            self.out_channels,
        )
        out = col @ w_flat + self.biases
        out = out.reshape(n, out_h, out_w, self.out_channels)
        return out

    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        """Compute backward pass and update filters.

        Args:
            dout: Upstream gradient of shape (N, out_H, out_W, out_channels).
            learning_rate: Step size for gradient descent.

        Returns:
            Gradient with respect to input.
        """
        n, h, w, c = self._input_cache.shape
        dout_flat = dout.reshape(-1, self.out_channels)

        w_flat = self.filters.reshape(
            self.kernel_size * self.kernel_size * self.in_channels,
            self.out_channels,
        )

        dcol = dout_flat @ w_flat.T
        dfilters = self._col_cache.T @ dout_flat
        dbiases = np.sum(dout, axis=(0, 1, 2))

        self.filters -= learning_rate * dfilters.reshape(self.filters.shape)
        self.biases -= learning_rate * dbiases

        dx = _col2im(
            dcol,
            n, h, w, c,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
        )
        return dx


class MaxPool2D:
    """2D max pooling layer for spatial downsampling.

    Reduces spatial dimensions by taking the maximum value within
    each pooling window. Caches max indices for gradient routing.
    """

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None) -> None:
        """Initialize max pooling layer.

        Args:
            pool_size: Height and width of pooling window.
            stride: Stride (defaults to pool_size if None).
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self._input_cache: Optional[np.ndarray] = None
        self._max_mask_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass through max pooling.

        Args:
            x: Input of shape (N, H, W, C).

        Returns:
            Output of shape (N, out_H, out_W, C).
        """
        self._input_cache = x
        n, h, w, c = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        out = np.zeros((n, out_h, out_w, c))
        max_mask = np.zeros_like(x)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                patch = x[:, h_start:h_end, w_start:w_end, :]
                out[:, i, j, :] = np.max(patch, axis=(1, 2))

                patch_flat = patch.reshape(n, -1, c)
                max_idx = np.argmax(patch_flat, axis=1)

                for b in range(n):
                    for ch in range(c):
                        idx = max_idx[b, ch]
                        r, c_ = np.unravel_index(idx, (self.pool_size, self.pool_size))
                        max_mask[b, h_start + r, w_start + c_, ch] = 1

        self._max_mask_cache = max_mask
        return out

    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        """Route gradients to max positions only.

        Max pooling backward propagates gradient only to the position
        that produced the maximum value in forward pass.

        Args:
            dout: Upstream gradient of shape (N, out_H, out_W, C).
            learning_rate: Unused for pooling (no parameters).

        Returns:
            Gradient with respect to input.
        """
        del learning_rate
        dx = np.zeros_like(self._input_cache)
        n, out_h, out_w, c = dout.shape

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                mask = self._max_mask_cache[:, h_start:h_end, w_start:w_end, :]
                dout_broadcast = np.repeat(
                    np.repeat(dout[:, i : i + 1, j : j + 1, :], self.pool_size, axis=1),
                    self.pool_size,
                    axis=2,
                )
                dx[:, h_start:h_end, w_start:w_end, :] += dout_broadcast * mask

        return dx


def _relu_forward(x: np.ndarray) -> np.ndarray:
    """Apply ReLU activation (forward pass)."""
    return np.maximum(0, x)


def _relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """ReLU backward pass: gradient only flows where x > 0."""
    return dout * (x > 0)


class Flatten:
    """Flatten layer to convert spatial tensors to 1D for dense layers."""

    def __init__(self) -> None:
        """Initialize flatten layer."""
        self._original_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Flatten spatial dimensions.

        Args:
            x: Input of shape (N, H, W, C).

        Returns:
            Output of shape (N, H*W*C).
        """
        self._original_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        """Reshape gradient back to original spatial dimensions."""
        del learning_rate
        return dout.reshape(self._original_shape)


class Dense:
    """Fully connected layer with ReLU or softmax activation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
    ) -> None:
        """Initialize dense layer.

        Args:
            input_size: Number of input features.
            output_size: Number of output neurons.
            activation: "relu" or "softmax".
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        std = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * std
        self.biases = np.zeros(output_size)

        self._input_cache: Optional[np.ndarray] = None
        self._z_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute forward pass."""
        self._input_cache = x
        z = x @ self.weights + self.biases
        self._z_cache = z

        if self.activation == "relu":
            return _relu_forward(z)
        if self.activation == "softmax":
            shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(shifted)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return z

    def backward(
        self, dout: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        """Compute backward pass and update weights."""
        n = self._input_cache.shape[0]

        if self.activation == "relu":
            delta = dout * (self._z_cache > 0)
        else:
            delta = dout

        dw = self._input_cache.T @ delta / n
        db = np.mean(delta, axis=0)
        dx = delta @ self.weights.T

        self.weights -= learning_rate * dw
        self.biases -= learning_rate * db
        return dx


class CNN:
    """Convolutional neural network for image classification.

    Architecture: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool ->
    Flatten -> Dense(ReLU) -> Dense(Softmax).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_classes: int,
        conv_config: Optional[List[Dict]] = None,
        dense_units: int = 128,
    ) -> None:
        """Initialize CNN architecture.

        Args:
            input_shape: (height, width, channels) of input images.
            n_classes: Number of output classes.
            conv_config: List of conv layer configs. Default: 2 layers.
            dense_units: Units in hidden dense layer before output.
        """
        if conv_config is None:
            conv_config = [
                {"filters": 32, "kernel_size": 3, "stride": 1, "padding": 1},
                {"filters": 64, "kernel_size": 3, "stride": 1, "padding": 1},
            ]

        self.layers: List = []
        self.activations: List[str] = []
        in_ch = input_shape[2]
        h, w = input_shape[0], input_shape[1]

        for cfg in conv_config:
            conv = Conv2D(
                in_channels=in_ch,
                out_channels=cfg["filters"],
                kernel_size=cfg["kernel_size"],
                stride=cfg.get("stride", 1),
                padding=cfg.get("padding", 0),
            )
            self.layers.append(conv)
            self.activations.append("relu")
            in_ch = cfg["filters"]
            h = (h + 2 * cfg.get("padding", 0) - cfg["kernel_size"]) // cfg.get(
                "stride", 1
            ) + 1
            w = (w + 2 * cfg.get("padding", 0) - cfg["kernel_size"]) // cfg.get(
                "stride", 1
            ) + 1
            self.layers.append(MaxPool2D(pool_size=2, stride=2))
            self.activations.append("linear")
            h = (h - 2) // 2 + 1
            w = (w - 2) // 2 + 1

        self.layers.append(Flatten())
        self.activations.append("linear")

        flat_size = h * w * in_ch
        self.layers.append(Dense(flat_size, dense_units, "relu"))
        self.activations.append("relu")
        self.layers.append(Dense(dense_units, n_classes, "softmax"))
        self.activations.append("softmax")

        self.n_classes = n_classes
        self._history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through all layers."""
        out = x
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                out = layer.forward(out)
                out = _relu_forward(out)
            else:
                out = layer.forward(out)
        return out

    def _compute_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Cross-entropy loss and gradient."""
        n = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / n
        gradient = (y_pred - y_true) / n
        return float(loss), gradient

    def _one_hot(self, labels: np.ndarray) -> np.ndarray:
        """Convert integer labels to one-hot encoding."""
        one_hot = np.zeros((labels.shape[0], self.n_classes))
        one_hot[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
        return one_hot

    def _backward(self, gradient: np.ndarray, learning_rate: float) -> None:
        """Backpropagate through layers.

        Order: loss gradient flows backward through Dense, Flatten,
        MaxPool, ReLU, Conv2D for each block. ReLU uses cached pre-activation.
        """
        conv_idx = len(self._conv_output_cache) - 1
        grad = gradient
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            if isinstance(layer, Dense):
                grad = layer.backward(grad, learning_rate)
            elif isinstance(layer, Flatten):
                grad = layer.backward(grad, learning_rate)
            elif isinstance(layer, MaxPool2D):
                grad = layer.backward(grad, learning_rate)
            elif isinstance(layer, Conv2D):
                grad = _relu_backward(grad, self._conv_output_cache[conv_idx])
                grad = layer.backward(grad, learning_rate)
                conv_idx -= 1

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the CNN with mini-batch gradient descent."""
        n = x_train.shape[0]
        y_enc = self._one_hot(y_train)

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            x_shuf = x_train[idx]
            y_shuf = y_enc[idx]
            epoch_losses: List[float] = []

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_batch = x_shuf[start:end]
                y_batch = y_shuf[start:end]

                out = self._forward_with_cache(x_batch)
                loss, grad = self._compute_loss(y_batch, out)
                self._backward(grad, learning_rate)
                epoch_losses.append(loss)

            avg_loss = float(np.mean(epoch_losses))
            preds = self.predict(x_train)
            acc = float(np.mean(preds == y_train))
            self._history["loss"].append(avg_loss)
            self._history["accuracy"].append(acc)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f} - acc: {acc:.4f}"
                logger.info(msg)
                print(msg)

        return self._history

    def _forward_with_cache(self, x: np.ndarray) -> np.ndarray:
        """Forward pass storing pre-ReLU conv outputs for backpropagation."""
        out = x
        self._conv_output_cache = []
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                out = layer.forward(out)
                self._conv_output_cache.append(out.copy())
                out = _relu_forward(out)
            else:
                out = layer.forward(out)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        out = self.forward(x, training=False)
        return np.argmax(out, axis=1)

    def evaluate(
        self, x_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate on test set."""
        out = self.forward(x_test, training=False)
        y_enc = self._one_hot(y_test)
        loss, _ = self._compute_loss(y_enc, out)
        preds = np.argmax(out, axis=1)
        acc = float(np.mean(preds == y_test))
        return {"loss": loss, "accuracy": acc}


def load_mnist(
    n_train: int = 5000,
    n_test: int = 1000,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST digits for image classification.

    Uses scikit-learn fetch_openml. Falls back to synthetic data
    if fetch fails (e.g., offline).

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (x_train, x_test, y_train, y_test).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    try:
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x_all = mnist.data.astype(np.float32) / 255.0
        y_all = mnist.target.astype(np.int32)

        x_all = x_all.reshape(-1, 28, 28, 1)
        idx = np.random.permutation(len(x_all))
        x_all = x_all[idx]
        y_all = y_all[idx]

        split = len(x_all) - n_test
        x_train = x_all[: min(n_train, split)]
        y_train = y_all[: min(n_train, split)]
        x_test = x_all[split : split + n_test]
        y_test = y_all[split : split + n_test]
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.warning("Could not fetch MNIST: %s. Using synthetic data.", e)
        return _generate_synthetic_images(n_train, n_test, random_seed or 42)


def _generate_synthetic_images(
    n_train: int,
    n_test: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic image-like data for testing."""
    np.random.seed(seed)
    n_classes = 10
    img_h, img_w = 28, 28

    def make_batch(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.random.rand(n_samples, img_h, img_w, 1).astype(np.float32) * 0.5
        y = np.random.randint(0, n_classes, size=n_samples)
        for c in range(n_classes):
            mask = y == c
            center_h = np.random.randint(4, img_h - 4, size=mask.sum())
            center_w = np.random.randint(4, img_w - 4, size=mask.sum())
            for i, (ch, cw) in enumerate(zip(center_h, center_w)):
                idx = np.where(mask)[0][i]
                x[idx, ch - 2 : ch + 2, cw - 2 : cw + 2, 0] += 0.5
        return x, y

    x_train, y_train = make_batch(n_train)
    x_test, y_test = make_batch(n_test)
    return x_train, x_test, y_train, y_test


class CNNRunner:
    """Orchestrates CNN training and evaluation from configuration."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize runner with config."""
        self.config = self._load_config(config_path)
        self._setup_logging()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load YAML configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config not found: %s, using defaults", config_path)
            return {}

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"))
        log_file = log_cfg.get("file", "logs/app.log")
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.setLevel(level)
        logger.addHandler(handler)

    def run(self) -> Dict[str, float]:
        """Run CNN training and evaluation."""
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("training", {})

        n_train = data_cfg.get("n_train", 5000)
        n_test = data_cfg.get("n_test", 1000)
        seed = data_cfg.get("random_seed")

        x_train, x_test, y_train, y_test = load_mnist(
            n_train=n_train, n_test=n_test, random_seed=seed
        )

        input_shape = (28, 28, 1)
        n_classes = 10

        conv_config = model_cfg.get("conv_layers", [
            {"filters": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"filters": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        ])
        dense_units = model_cfg.get("dense_units", 128)

        cnn = CNN(
            input_shape=input_shape,
            n_classes=n_classes,
            conv_config=conv_config,
            dense_units=dense_units,
        )

        cnn.train(
            x_train,
            y_train,
            epochs=train_cfg.get("epochs", 10),
            learning_rate=train_cfg.get("learning_rate", 0.001),
            batch_size=train_cfg.get("batch_size", 32),
            validation_data=(x_test, y_test),
            verbose=True,
        )

        results = cnn.evaluate(x_test, y_test)
        logger.info(
            "Test - loss: %.6f, accuracy: %.4f",
            results["loss"],
            results["accuracy"],
        )
        return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train CNN from scratch for image classification"
    )
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--output", type=str, help="Path to save results JSON")
    args = parser.parse_args()

    runner = CNNRunner(
        config_path=Path(args.config) if args.config else None
    )
    results = runner.run()

    print("\nFinal Results:")
    print("=" * 40)
    for key, val in results.items():
        print(f"  {key}: {val:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
