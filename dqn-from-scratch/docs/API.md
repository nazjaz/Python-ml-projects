# DQN-from-Scratch API Documentation

## Module: `src.main`

### Classes

#### `SimpleGridworldEnv`

Small deterministic gridworld with discrete states and actions.

- `reset()`: Reset to start state; returns state index.
- `step(action)`: Apply action (0: up, 1: down, 2: left, 3: right). Returns
  `(next_state, reward, done)`.

#### `ReplayBuffer`

Fixed-size experience replay buffer storing transitions:
`(state, action, reward, next_state, done)`.

- Constructor: `ReplayBuffer(capacity, state_dim)`.
- `add(state, action, reward, next_state, done)`.
- `can_sample(batch_size)`: Return True if there are enough elements.
- `sample(batch_size)`: Return mini-batch of transitions.

#### `QNetwork`

Simple fully connected Q-network with one hidden layer and ReLU activation.

- Constructor: `QNetwork(state_dim, n_actions, hidden_dim)`.
- `forward(states)`: Compute Q-values for a batch of states.
- `backward(grad_q, lr)`: Backpropagate gradient of loss w.r.t. Q-values and
  update parameters.
- `copy_from(other)`: Copy parameters from another network instance.

#### `DQNConfig`

Dataclass encapsulating configuration for DQN training:

- `gamma`, `learning_rate`, `batch_size`, `buffer_capacity`,
  `epsilon_start`, `epsilon_end`, `epsilon_decay_steps`,
  `target_update_interval`, `max_episodes`, `max_steps_per_episode`.

#### `DQNAgent`

DQN agent implementing:

- Online Q-network and separate target network.
- Epsilon-greedy action selection.
- Experience replay sampling.
- Periodic target network updates.

Key methods:

- Constructor: `DQNAgent(state_dim, n_actions, hidden_dim, config)`.
- `select_action(state)`: Epsilon-greedy discrete action.
- `store_transition(state, action, reward, next_state, done)`.
- `train_step()`: Sample a batch, compute TD targets, update Q-network,
  periodically update target network, and return loss.

#### `DQNRunner`

High-level runner that wires together environment and agent from YAML
configuration, then trains the agent and reports metrics.

- Constructor: `DQNRunner(config_path: Optional[Path] = None)`.
- `run()`: Train for `max_episodes`, returning a dict with:
  - `final_episode_reward`
  - `average_return_last_50`

### Functions

#### `main()`

Command-line entry point:

- `--config`: Path to YAML configuration file (optional).
- `--output`: Path to JSON results file (optional).

Runs `DQNRunner`, prints final metrics, and optionally writes them to disk.

