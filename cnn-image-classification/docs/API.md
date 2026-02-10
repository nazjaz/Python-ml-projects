# CNN Image Classification API Documentation

## Module: src.main

### Classes

#### Conv2D

2D convolution layer with configurable filters, stride, and padding.

**Constructor**: `Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0)`

- `in_channels`: Number of input channels
- `out_channels`: Number of output filters
- `kernel_size`: Kernel height and width
- `stride`: Convolution stride (default: 1)
- `padding`: Zero-padding amount (default: 0)

**Methods**:
- `forward(x)`: Compute convolution. Input shape (N, H, W, C), output (N, out_H, out_W, out_channels)
- `backward(dout, learning_rate)`: Backpropagate gradient and update filters

#### MaxPool2D

2D max pooling layer for spatial downsampling.

**Constructor**: `MaxPool2D(pool_size=2, stride=None)`

- `pool_size`: Pool window size (default: 2)
- `stride`: Stride (defaults to pool_size)

**Methods**:
- `forward(x)`: Max pool over spatial dimensions
- `backward(dout, learning_rate)`: Route gradients to max positions

#### Flatten

Reshapes spatial tensor to 1D for dense layers.

**Methods**:
- `forward(x)`: (N, H, W, C) -> (N, H*W*C)
- `backward(dout, learning_rate)`: Reshape gradient back

#### Dense

Fully connected layer with ReLU or softmax activation.

**Constructor**: `Dense(input_size, output_size, activation="relu")`

**Methods**:
- `forward(x)`: Linear transform and activation
- `backward(dout, learning_rate)`: Backpropagate and update weights

#### CNN

Full CNN model for image classification.

**Constructor**: `CNN(input_shape, n_classes, conv_config=None, dense_units=128)`

- `input_shape`: (height, width, channels)
- `n_classes`: Number of output classes
- `conv_config`: List of conv layer config dicts
- `dense_units`: Hidden dense layer size

**Methods**:
- `forward(x, training)`: Forward pass
- `predict(x)`: Return class labels
- `evaluate(x_test, y_test)`: Compute loss and accuracy
- `train(x_train, y_train, epochs, learning_rate, batch_size, ...)`: Train model

### Functions

#### load_mnist(n_train, n_test, random_seed)

Load MNIST digits. Returns (x_train, x_test, y_train, y_test). Falls back to synthetic data if fetch fails.

#### main()

CLI entry point. Parses --config and --output arguments.
