# Proximal Policy Optimization (PPO) from Scratch

This project implements Proximal Policy Optimization (PPO) from scratch using only NumPy. It includes a clipped surrogate objective, trust region constraints via optional KL penalty, Generalized Advantage Estimation (GAE), and multi-epoch minibatch updates on collected rollouts. A small gridworld environment is used for demonstration.

### Project Title and Description

The goal is to provide a clear, framework-free reference implementation of PPO with the clipped objective and trust region behavior commonly used in modern policy gradient methods. The agent learns to navigate a gridworld to reach a goal by maximizing cumulative reward using policy gradient updates with clipping to limit policy change.

**Target Audience**: Practitioners and students who want to understand PPO mechanics in detail without depending on large RL libraries.

### Features

- **Simple gridworld environment**:
  - Discrete states in an NxN grid.
  - Four discrete actions (up, down, left, right).
  - Negative step rewards and positive terminal reward at the goal.
- **Policy and value networks**:
  - Two-layer MLPs with ReLU; policy outputs softmax over actions, value network outputs V(s).
  - Implemented with NumPy only; manual backprop for policy and value updates.
- **PPO clipped objective**:
  - Ratio clipping (e.g. 1 +/- epsilon) to bound policy updates.
  - Surrogate loss as minimum of unclipped and clipped objectives.
- **Trust region**:
  - Optional KL target and gradient handling to keep updates within a trust region.
- **Generalized Advantage Estimation (GAE)**:
  - Lambda-return and advantage computation for variance reduction.
  - Advantage normalization (zero mean, unit variance) before updates.
- **Rollout buffer and multi-epoch updates**:
  - Collects trajectories until a fixed number of steps; then runs multiple PPO epochs over minibatches.
- **Configuration-driven training**:
  - YAML config for environment, agent hyperparameters, and logging.
- **Logging and CLI**:
  - Rotating log file; command-line options for config path and output JSON.

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

#### Step 1: Navigate to project directory

```bash
cd /path/to/Python-ml-projects/ppo-from-scratch
```

#### Step 2: Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Verify installation

```bash
python src/main.py
```

### Configuration

The project reads configuration from `config.yaml` by default.

#### Configuration file structure

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

env:
  grid_size: 5
  random_seed: 0

agent:
  hidden_dim: 64
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  kl_target: 0.01
  max_grad_norm: 0.5
  ppo_epochs: 4
  batch_size: 64
  learning_rate: 0.0003
  rollout_steps: 128
  max_episodes: 200
  max_steps_per_episode: 100
```

- **env**: `grid_size` (grid side length), `random_seed` (NumPy seed).
- **agent**: PPO hyperparameters including discount (`gamma`), GAE lambda, clip range (`clip_epsilon`), value and entropy coefficients, KL target, gradient clipping, PPO epochs, batch size, learning rate, rollout length, and episode limits.

#### Environment variables

Copy `.env.example` to `.env` and configure optional overrides such as `RANDOM_SEED`. No external services or credentials are required.

### Usage

#### Basic usage

```bash
python src/main.py
```

#### With custom config

```bash
python src/main.py --config path/to/config.yaml
```

#### Save results to JSON

```bash
python src/main.py --output results.json
```

The script prints final episode reward and the average return over the last 50 episodes, and optionally writes them to a JSON file.

### Project structure

```text
ppo-from-scratch/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── src/
│   └── main.py
├── tests/
│   └── test_main.py
├── docs/
│   └── API.md
└── logs/
    └── .gitkeep
```

- `src/main.py`: Gridworld environment, policy and value networks, GAE, rollout buffer, PPO agent with clipped objective, training runner, and CLI.
- `config.yaml`: Environment and PPO hyperparameters.
- `tests/test_main.py`: Unit tests for buffer, GAE, networks, agent, and runner.
- `docs/API.md`: API reference for the main classes and functions.

### Testing

Run tests:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Troubleshooting

#### Learning seems slow

- Increase `max_episodes` or `rollout_steps`.
- Tune `learning_rate` and `clip_epsilon`; smaller clip can stabilize but slow learning.
- Increase `hidden_dim` for more capacity.

#### Instability or divergence

- Lower `learning_rate`.
- Reduce `clip_epsilon` to limit policy change.
- Increase `rollout_steps` and `batch_size` for more stable advantage estimates.
- Ensure `max_grad_norm` is set for gradient clipping.

### Contributing

1. Create a feature branch.
2. Follow PEP 8 and the shared project standards.
3. Add or update tests when modifying behavior.
4. Ensure all tests pass before opening a pull request.

### License

This project is part of the Python ML Projects collection. Refer to the main repository license for details.
