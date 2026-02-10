# RNN Sequence Prediction API Documentation

## Module: `src.main`

### Classes

#### `LSTM`

Single-layer LSTM network for many-to-one sequence prediction.

**Constructor**: `LSTM(input_dim, hidden_dim)`

- `input_dim`: Dimensionality of input features per time step
- `hidden_dim`: Number of hidden units in the LSTM cell

**Methods**:

- `forward(x)`: Run forward pass over a batch of sequences.
  - `x` shape: `(batch_size, sequence_length, input_dim)`
  - Returns tuple `(h_seq, h_last, c_last)` where:
    - `h_seq`: All hidden states, shape `(batch_size, sequence_length, hidden_dim)`
    - `h_last`: Final hidden state, shape `(batch_size, hidden_dim)`
    - `c_last`: Final cell state, shape `(batch_size, hidden_dim)`
- `backward(dh_last, learning_rate)`: Backpropagate gradient from final hidden state and update parameters.
  - `dh_last` shape: `(batch_size, hidden_dim)`
  - `learning_rate`: Gradient descent step size

#### `LSTMSequenceRegressor`

High-level model for next-step sequence prediction using an LSTM encoder and a dense output layer.

**Constructor**: `LSTMSequenceRegressor(input_dim, hidden_dim, output_dim)`

- `input_dim`: Feature dimension per time step
- `hidden_dim`: Hidden units in the LSTM layer
- `output_dim`: Dimensionality of the prediction (typically 1)

**Methods**:

- `train(x_train, y_train, epochs, learning_rate, batch_size, verbose)`:
  - Trains the model using mini-batch gradient descent with mean squared error loss.
  - Returns a history dictionary with per-epoch training loss.
- `evaluate(x_test, y_test)`: Computes MSE on a held-out test set.
- `predict(x)`: Returns model predictions for input sequences.

#### `RNNRunner`

Orchestrates configuration loading, data generation, training, and evaluation.

**Constructor**: `RNNRunner(config_path: Optional[Path] = None)`

**Methods**:

- `run()`: Loads configuration, generates data, trains the model, evaluates on the test set, and returns a dictionary with `train_loss` and `test_loss`.

### Functions

#### `generate_sine_wave_sequences(n_samples, sequence_length, noise_std, random_seed)`

Generate synthetic sine-wave sequences for supervised learning.

- Returns tuple `(x, y)` where:
  - `x`: Input sequences, shape `(n_samples, sequence_length, 1)`
  - `y`: Scalar targets, shape `(n_samples, 1)` representing the next value after the sequence.

#### `main()`

Command-line entry point. Supports:

- `--config`: Path to configuration YAML file
- `--output`: Path to JSON file for saving results

