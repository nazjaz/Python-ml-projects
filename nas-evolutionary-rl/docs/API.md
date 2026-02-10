# API Reference

Main classes and functions in `src.main` for Neural Architecture Search (NAS) with evolutionary and RL methods.

## Search space

- **encode_architecture(num_layers, hidden_dim, activation) -> List[int]**  
  Encodes an architecture as three choice indices (num_layers, hidden_dim, activation).

- **decode_architecture(choices: List[int]) -> Tuple[int, int, str]**  
  Decodes choice indices to (num_layers, hidden_dim, activation).

- **random_architecture(random_seed?) -> List[int]**  
  Samples a random architecture (list of three choice indices).

Constants: `NUM_LAYERS_OPTIONS`, `HIDDEN_DIM_OPTIONS`, `ACTIVATION_OPTIONS`, `NUM_CHOICES`.

## Trainable MLP

### TrainableMLP

Variable-depth MLP with configurable activation (relu or tanh).

- **__init__(in_dim, out_dim, num_layers, hidden_dim, activation, random_seed?)**  
  Builds an MLP with (num_layers - 1) hidden layers of size hidden_dim and one output layer.

- **forward(x) -> np.ndarray**  
  Returns logits of shape (batch, out_dim).

- **backward(grad_out, lr)**  
  Backpropagates and updates weights.

## Evaluation

### train_and_evaluate(train_x, train_y, val_x, val_y, choices, epochs?, lr?, batch_size?, random_seed?) -> float

Trains an MLP with the architecture given by `choices` and returns validation accuracy in [0, 1].

## Evolutionary NAS

### EvolutionaryNAS

- **__init__(population_size, num_generations, mutation_prob, tournament_size, train_x, train_y, val_x, val_y, eval_epochs?, random_seed?)**  
  Configures population, generations, mutation probability, tournament size, data, and evaluation epochs.

- **run() -> Tuple[List[int], float, List[Dict]]**  
  Runs the evolutionary search. Returns (best_architecture_choices, best_fitness, history).

## RL-based NAS

### Controller

REINFORCE controller that outputs logits per step and samples architectures.

- **sample(random_seed?) -> List[int]**  
  Samples an architecture and stores log-probabilities for the last trajectory.

- **get_log_prob() -> float**  
  Returns the sum of log-probabilities of the last sampled trajectory.

- **backward_reinforce(reward, lr)**  
  Updates the controller with a REINFORCE step using the given reward.

### RLNAS

- **__init__(num_rollouts, controller_lr, train_x, train_y, val_x, val_y, eval_epochs?, random_seed?)**

- **run() -> Tuple[List[int], float, List[Dict]]**  
  Runs RL-based search. Returns (best_architecture_choices, best_reward, history).

## Data and config

### load_digits_data(train_ratio?, max_samples?, random_seed?) -> Tuple

Returns (train_x, train_y, val_x, val_y). Uses sklearn digits if available; otherwise synthetic data.

### NASConfig

Dataclass: method ("evolution" | "rl"), population_size, num_generations, mutation_prob, tournament_size, num_rollouts, controller_lr, eval_epochs, train_ratio, max_samples, random_seed.

### run_nas(config: NASConfig) -> Dict

Runs NAS with the given config. Returns a dict with method, best_architecture, best_validation_accuracy, and history.

## Entry point

### main()

Parses --config, --output, --method; loads YAML; runs NAS; prints and optionally writes JSON results.
