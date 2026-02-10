# MLOps Pipeline: Data Versioning, Model Monitoring, Automated Retraining

A Python implementation of an end-to-end ML system with MLOps pipeline including data versioning (content-addressed versions with metadata), model monitoring (metrics history and drift detection), and automated retraining triggered by performance degradation or input drift.

## Project Title and Description

This project provides a self-contained MLOps pipeline suitable for small to medium ML deployments. Data is versioned by content hash with schema and basic statistics stored as metadata. Model performance and feature distribution are monitored over time; when accuracy drops below a threshold or feature drift exceeds a limit, the pipeline triggers retraining. New models are registered with metadata and evaluation metrics are recorded for audit.

**Target Audience**: Engineers and data scientists building production ML systems who need reproducible data versions, model monitoring, and automated retraining without external MLOps platforms.

## Features

- **Data versioning**: Content-addressed data storage (parquet + metadata). Register datasets, list versions, load by version id. Metadata includes row count, schema, and feature means for drift comparison.
- **Model monitoring**: Persist metrics history (e.g. accuracy per run). Maintain reference feature distribution; compute relative drift for new data. Configurable accuracy and drift thresholds.
- **Automated retraining**: Pipeline checks whether a model exists, current accuracy meets threshold, and drift is within limit. If any check fails, trains a new model on the latest data version, saves to registry, updates reference distribution, and records metrics.
- Config-driven paths, thresholds, and training parameters. Demo mode uses synthetic data; same pipeline can be wired to real data ingestion.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/mlops-pipeline
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows: `venv\Scripts\activate`

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --config config.yaml
```

## Configuration

### Configuration File Structure

Configure via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

paths:
  data_versions: "data_versions"
  model_registry: "model_registry"
  monitoring: "monitoring"

data_versioning:
  enabled: true
  hash_algorithm: "sha256"

monitoring:
  metrics_history_file: "metrics_history.json"
  drift_history_file: "drift_history.json"
  accuracy_min_threshold: 0.70
  drift_threshold: 5.0
  drift_min_samples: 50

training:
  model_type: "logistic"
  random_state: 42
  test_size: 0.2
  max_iter: 500
  C: 1.0

pipeline:
  evaluate_after_retrain: true
```

### Environment Variables

Copy `.env.example` to `.env` and optionally set:

- `RANDOM_SEED`: Override random seed.
- `CONFIG_PATH`: Override config file path (default: config.yaml).

## Usage

### Run Full Pipeline

```bash
python src/main.py --config config.yaml
```

The pipeline will: (1) generate and version synthetic data, (2) check if a model exists and if retraining is needed (no model, low accuracy, or drift), (3) train and register a model if needed, (4) evaluate and record metrics.

### Programmatic Usage

```python
from src.main import run_pipeline, DataVersioner, ModelRegistry, ModelMonitor

summary = run_pipeline("config.yaml")
# summary["data_version"], summary["model_version"], summary["retrained"], summary["accuracy"]
```

### Data Versioning

```python
from src.main import DataVersioner
import pandas as pd

versioner = DataVersioner("data_versions")
df = pd.read_csv("raw_data.csv")
version_id = versioner.register(df)
loaded_df, meta = versioner.load(version_id)
```

### Model Monitoring and Retrain Check

```python
from src.main import ModelMonitor

monitor = ModelMonitor("monitoring", accuracy_min_threshold=0.75, drift_threshold=0.2)
monitor.record_metrics(0.82, data_version="v1", model_version="m1")
should_retrain = monitor.should_retrain(current_accuracy=0.70, no_model_yet=False)
```

## Project Structure

```
mlops-pipeline/
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

- `src/main.py`: DataVersioner, ModelRegistry, ModelMonitor, train_model, run_pipeline, CLI.
- `config.yaml`: Paths, monitoring thresholds, training parameters.
- `data_versions/`, `model_registry/`, `monitoring/`: Created at runtime (gitignored by default).

## Testing

Run tests from the project root:

```bash
cd mlops-pipeline
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Troubleshooting

- **FileNotFoundError: Config file not found**: Run from project directory or pass absolute path to `--config`.
- **FileNotFoundError: Data version not found**: Ensure the version id was produced by `DataVersioner.register` in this project; check `data_versions/versions.json`.
- **Parquet read/write errors**: Ensure `pyarrow` is installed (`pip install pyarrow`).

## Contributing

1. Create a virtual environment and install dependencies.
2. Follow PEP 8 and project docstring/type-hint standards.
3. Add tests for new behavior; run pytest before submitting.

## License

See repository license.
