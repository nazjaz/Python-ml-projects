# API Reference

- `cosine_beta_schedule(T, s=0.008)`: Cosine schedule for betas.
- `MLP`: Two-layer MLP for noise prediction.
- `timestep_embedding(t, dim)`: Sinusoidal timestep embedding.
- `load_data(max_samples, random_seed)`: Load 8x8 images (digits or synthetic).
- `DiffusionConfig`: Configuration dataclass.
- `DiffusionModel`: Implements `q_sample`, `predict_noise`, `train`, `p_sample`, and `sample`.
- `run_diffusion(config)`: Train model and return metrics and generated sample shapes.
- `main()`: CLI entry point.
