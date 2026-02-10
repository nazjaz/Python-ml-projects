# API Reference

Federated learning: distributed training, server aggregation, privacy-preserving options.

## Model

### MLP

Two-layer ReLU MLP with get/set_weights for federation.

- **get_weights() -> List[np.ndarray]**  
  Returns [w1, b1, w2, b2] for aggregation.

- **set_weights(weights)**  
  Sets parameters from list of arrays (same order as get_weights).

- **forward(x)**, **backward(grad_out, lr)**  
  Standard forward and SGD update.

## Privacy helpers

### clip_gradient_norm(weights, ref_weights, max_norm) -> List[np.ndarray]

Clips the update (weights - ref_weights) to L2 norm at most max_norm; returns ref_weights + clipped_delta.

### add_gaussian_noise(weights, sigma) -> List[np.ndarray]

Adds N(0, sigma^2) noise to each parameter array.

## Data

### load_data(train_ratio?, max_samples?, random_seed?) -> (train_x, train_y, val_x, val_y)

Loads digits (sklearn) or synthetic data.

### partition_data(x, y, num_clients, iid?, random_seed?) -> List[(x_i, y_i)]

Splits data into num_clients shards. iid=True: random split; iid=False: sort by label then split.

## Clients and server

### FederatedClient

- **__init__(client_id, train_x, train_y, in_dim, hidden_dim, out_dim, local_epochs?, lr?, batch_size?, max_grad_norm?, noise_sigma?, random_seed?)**
- **train_local(global_weights) -> (updated_weights, sample_count)**  
  Trains on local data from global_weights; optionally clips gradient norm and adds Gaussian noise; returns new weights and local sample count.

### aggregate_fedavg(client_weights, client_counts) -> List[np.ndarray]

Weighted average of client weight lists by sample count (FedAvg).

### FederatedServer

- **__init__(in_dim, hidden_dim, out_dim, clients, random_seed?)**
- **run_round() -> Dict**  
  Broadcasts global weights to clients, collects updates, aggregates with FedAvg, updates global model.
- **get_global_weights()**
- **global_model**: the central MLP.

### evaluate_global(server, val_x, val_y) -> (accuracy, mean_CE)

Evaluates the server’s global model on validation data.

## Config and run

### FederatedConfig

num_clients, num_rounds, local_epochs, hidden_dim, learning_rate, batch_size, train_ratio, max_samples, iid, max_grad_norm, noise_sigma, random_seed.

### run_federated(config) -> Dict

Partitions data, builds clients and server, runs num_rounds, returns final_val_accuracy, final_val_ce, num_rounds, num_clients, history.

### main()

CLI: --config, --output. Runs federated learning and prints or writes JSON.
