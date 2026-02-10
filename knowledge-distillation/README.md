# Knowledge Distillation for Model Compression

This project implements knowledge distillation with a teacher-student setup and temperature scaling. A larger teacher MLP is trained on labels; a smaller student MLP is then trained to mimic the teacher's soft outputs (and optionally hard labels) for model compression. Implemented in NumPy only.

### Description

Knowledge distillation transfers knowledge from a teacher model to a smaller student by training the student on the teacher's soft probability distribution (with temperature scaling) in addition to or instead of hard labels. This yields a compact model that can approach the teacher's accuracy.

**Target audience**: Developers and students interested in model compression and distillation without large frameworks.

### Features

- **Teacher-student architecture**: Teacher MLP (larger hidden layer) and student MLP (smaller hidden layer); same input and output dimensions.
- **Temperature scaling**: Softmax with temperature T on teacher and student logits for soft labels; higher T produces softer distributions that carry more information.
- **Distillation loss**: Combined loss = alpha * KL(teacher_soft, student_soft) + (1 - alpha) * CE(student, hard labels). Teacher is fixed during student training.
- **Training pipeline**: Train teacher with cross-entropy; then train student with distillation. Report teacher/student validation accuracy and parameter counts.
- **Config and CLI**: YAML config for hidden sizes, epochs, temperature, alpha, learning rate; optional JSON output.

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd Python-ml-projects/knowledge-distillation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Default: `config.yaml`

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

distillation:
  teacher_hidden: 128
  student_hidden: 32
  teacher_epochs: 10
  student_epochs: 10
  temperature: 4.0
  alpha: 0.7
  learning_rate: 0.01
  batch_size: 32
  train_ratio: 0.8
  max_samples: null
  random_seed: 0
```

- **temperature**: Used in softmax for soft labels (e.g. 4.0).
- **alpha**: Weight for KL term; (1 - alpha) for hard-label CE.
- **teacher_hidden / student_hidden**: Hidden layer sizes; student smaller for compression.

### Usage

```bash
python src/main.py
python src/main.py --config path/to/config.yaml --output results.json
```

Output includes teacher and student validation accuracy and compression ratio (teacher params / student params).

### Project structure

```
knowledge-distillation/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── .gitignore
├── src/main.py
├── tests/test_main.py
├── docs/API.md
└── logs/.gitkeep
```

### Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

### Troubleshooting

- **Student much worse than teacher**: Increase temperature or alpha; train student for more epochs.
- **Slow**: Reduce teacher_epochs/student_epochs or set max_samples in config.

### License

Part of Python ML Projects; see repository license.
