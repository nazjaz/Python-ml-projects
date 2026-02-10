# Diffusion Models for Image Generation

This project implements a simplified diffusion model (DDPM-style) for image generation with forward and reverse diffusion processes. Small 8x8 grayscale images (e.g. digits) are flattened; a simple MLP predicts injected noise at each timestep. The model is trained to denoise and can generate images by reversing the diffusion process from pure noise.

### Description

- **Forward diffusion**: Gradually corrupts clean images with Gaussian noise over a fixed number of timesteps.
- **Reverse diffusion**: Learns a noise-prediction network that approximates the reverse process and iteratively denoises noise samples to synthesize images.

### Features

- Cosine beta schedule for forward diffusion.
- Sinusoidal timestep embeddings.
- MLP noise predictor conditioned on noisy image and timestep.
- Training objective: mean squared error between true and predicted noise.
- Sampling by reversing the diffusion process from Gaussian noise.

### Installation

```bash
cd Python-ml-projects/diffusion-image-generation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

See `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

diffusion:
  timesteps: 100
  hidden_dim: 128
  emb_dim: 32
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  max_samples: null
  random_seed: 0
```

### Usage

```bash
python src/main.py
python src/main.py --config config.yaml --output results.json
```

### Project structure

- `src/main.py`: Diffusion model implementation (forward and reverse processes, training, sampling).
- `tests/test_main.py`: Unit tests.
- `docs/API.md`: API reference.

### Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```
