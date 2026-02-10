## Deep Q-Network (DQN) from Scratch with Experience Replay and Target Network

This project implements a Deep Q-Network (DQN) agent from scratch using only
NumPy. It includes a simple gridworld environment, fully connected Q-network,
experience replay buffer, epsilon-greedy exploration, and a separate target
network for stable Q-learning.

### Project Title and Description

The goal is to provide a clear, framework-free reference implementation of DQN
as introduced in the deep reinforcement learning literature. The agent learns
to navigate a small gridworld to reach a goal state by maximizing cumulative
reward using Q-learning with function approximation.

**Target Audience**: Practitioners and students who want to understand DQN
mechanics in detail without depending on large RL libraries.

### Features

- **Simple gridworld environment**:
  - Discrete states arranged in a 2D grid.
  - Four discrete actions (up, down, left, right).
  - Negative step rewards and positive terminal reward at the goal.
- **Deep Q-network**:
  - Fully connected neural network with one hidden layer and ReLU activation.
  - Predicts Q-values for all actions in a given state.
- **Experience replay**:
  - Fixed-size replay buffer storing transitions.
  - Random mini-batch sampling for decorrelated updates.
- **Target network**:
  - Separate Q-network used to compute TD targets.
  - Periodically synchronized with the online network for stability.
- **Epsilon-greedy exploration**:
  - Linearly decaying epsilon from `epsilon_start` to `epsilon_end`.
- **Configuration-driven training**:
  - YAML config specifying environment, agent hyperparameters, and training
    duration.
- **Logging and CLI**:
  - Rotating log file.
  - Command-line interface to run training and export metrics.

### Prerequisites

- Python 3.8 or higher
- `pip`

### Installation

#### Step 1: Navigate to project directory

```bash
cd /path/to/Python-ml-projects/dqn-from-scratch
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
  learning_rate: 0.001
  batch_size: 32
  buffer_capacity: 10000
  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_decay_steps: 1000
  target_update_interval: 100
  max_episodes: 300
  max_steps_per_episode: 100
```

### Environment variables

Copy `.env.example` to `.env` and configure optional overrides such as
`RANDOM_SEED`. No external services or credentials are required.

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

The script prints final episode reward and the average return over the last
50 episodes, and optionally writes them to a JSON file.

### Project structure

```text
dqn-from-scratch/
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

- `src/main.py`: Gridworld environment, replay buffer, Q-network, DQN agent,
  training runner, and CLI.
- `config.yaml`: Environment and DQN hyperparameters.
- `tests/test_main.py`: Unit tests for buffer, Q-network, agent, and runner.
- `docs/API.md`: API-level documentation of the main classes and functions.

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

- Increase `max_episodes` or `max_steps_per_episode`.
- Increase `hidden_dim` to allow more capacity in the Q-network.
- Ensure `epsilon` decays slowly enough for sufficient exploration.

#### Instability or divergence

- Lower `learning_rate`.
- Increase `target_update_interval` to stabilize targets.
- Increase `buffer_capacity` or `batch_size` for more diverse updates.

### Contributing

1. Create a feature branch.
2. Follow PEP 8 and the shared project standards.
3. Add or update tests when modifying behavior.
4. Ensure all tests pass before opening a pull request.

### License

This project is part of the Python ML Projects collection. Refer to the main
repository license for details.

