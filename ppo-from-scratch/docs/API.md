# API Reference

This document describes the main classes and functions in `src.main` for the PPO (Proximal Policy Optimization) implementation.

## Environment

### SimpleGridworldEnv

Discrete gridworld with start at (0,0), goal at (N-1,N-1), and four actions (up, down, left, right).

- **`__init__(size: int = 5, random_seed: Optional[int] = None)`**  
  Builds an NxN grid; `random_seed` seeds NumPy for reproducibility.

- **`reset() -> int`**  
  Resets to start state; returns state index (0).

- **`step(action: int) -> Tuple[int, float, bool]`**  
  Applies action; returns (next_state_index, reward, done). Reward is +10 at goal, -1 per step.

Attributes: `n_states`, `n_actions` (4).

## Networks

### MLP

Two-layer fully connected network with ReLU.

- **`forward(x: np.ndarray) -> np.ndarray`**  
  Returns output of shape (batch, out_dim).

- **`backward(grad_out: np.ndarray, lr: float) -> np.ndarray`**  
  Backpropagates gradient and updates weights; returns gradient w.r.t. input.

### PolicyNetwork

Policy pi(a|s) with softmax over actions.

- **`forward(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`**  
  Returns (logits, probs, log_probs) for the batch of states.

- **`sample(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`**  
  Samples actions from the policy; returns (actions, log_probs, entropy).

- **`evaluate_actions(states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`**  
  Returns (log_probs, entropy) for the given state-action pairs.

- **`backward_policy(states, actions, grad_log_prob, lr)`**  
  Updates policy parameters using the provided gradient of the log-probability.

### ValueNetwork

State-value function V(s).

- **`forward(states: np.ndarray) -> np.ndarray`**  
  Returns value estimates of shape (batch,) or (batch, 1).

- **`backward(states: np.ndarray, grad_values: np.ndarray, lr: float)`**  
  Updates value network; `grad_values` may be 1d or 2d.

## Advantage Estimation

### compute_gae

```python
compute_gae(rewards, values, dones, next_value, gamma, lam)
```

Computes returns and advantages using Generalized Advantage Estimation.  
Returns: `(returns: np.ndarray, advantages: np.ndarray)`.

### normalize_advantages

```python
normalize_advantages(advantages: np.ndarray) -> np.ndarray
```

Normalizes advantages to zero mean and unit variance (in-place safe; returns the array).

## Configuration and Buffers

### PPOConfig

Dataclass of PPO hyperparameters: `gamma`, `gae_lambda`, `clip_epsilon`, `value_coef`, `entropy_coef`, `kl_target`, `max_grad_norm`, `ppo_epochs`, `batch_size`, `learning_rate`, `rollout_steps`, `max_episodes`, `max_steps_per_episode`.

### RolloutBuffer

Stores one rollout of transitions for PPO.

- **`add(state, action, reward, log_prob, value, done)`**  
  Appends one transition.

- **`clear()`**  
  Clears all stored data.

- **`to_arrays()`**  
  Returns `(states, actions, rewards, log_probs_old, values, dones)` as NumPy arrays.

## Agent and Runner

### PPOAgent

PPO agent with clipped objective and optional KL trust region.

- **`__init__(state_dim, n_actions, hidden_dim, config: PPOConfig)`**  
  Builds policy and value networks and a rollout buffer.

- **`select_action(state: np.ndarray) -> Tuple[int, float, float]`**  
  Returns (action, log_prob, value) for the given state.

- **`store_transition(state, action, reward, log_prob, value, done)`**  
  Adds a transition to the rollout buffer.

- **`update() -> Dict[str, float]`**  
  Computes GAE, normalizes advantages, runs PPO epochs with clipping, updates policy and value networks; clears rollout. Returns dict with `policy_loss`, `value_loss`, `entropy`.

### PPORunner

Runs PPO training from a YAML config file.

- **`__init__(config_path: Optional[Path] = None)`**  
  Loads config (default: project `config.yaml`) and sets up logging.

- **`run() -> Dict[str, float]`**  
  Builds environment and agent from config, runs episodes until `max_episodes`, performs PPO updates when rollout reaches `rollout_steps`. Returns `final_episode_reward` and `average_return_last_50`.

## Entry Point

### main()

Parses CLI arguments `--config` and `--output`, runs `PPORunner(config_path).run()`, prints results and optionally writes JSON to the output path.
