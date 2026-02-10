# API Reference

Knowledge distillation: teacher-student architecture and temperature scaling.

## Functions

### softmax(x, axis=-1, temperature=1.0)

Softmax with optional temperature: `exp(x/T) / sum(exp(x/T))`. Higher T yields softer distributions.

### kl_divergence(p, q, axis=-1)

KL(p || q) along the given axis. Inputs are probability vectors; returns per-element KL.

## MLP

Two-layer ReLU MLP used for both teacher and student.

- **__init__(in_dim, hidden_dim, out_dim, random_seed?)**
- **forward(x) -> logits**
- **backward(grad_out, lr)**  
  Updates weights; call after forward.

## Training

### train_teacher(teacher, train_x, train_y, epochs, lr, batch_size, random_seed?) -> List[float]

Trains the teacher with cross-entropy on hard labels. Returns mean loss per epoch.

### train_student_with_distillation(teacher, student, train_x, train_y, temperature, alpha, epochs, lr, batch_size, random_seed?) -> List[float]

Trains the student with combined loss: `alpha * KL(soft_teacher, soft_student) + (1-alpha) * CE(student, hard)`. Teacher logits and student logits are scaled by `temperature` for the soft targets. Teacher is not updated. Returns mean loss per epoch.

### evaluate(model, x, y) -> (accuracy, mean_CE)

Returns validation accuracy and mean cross-entropy.

### load_data(train_ratio?, max_samples?, random_seed?) -> (train_x, train_y, val_x, val_y)

Loads digits (sklearn) or synthetic data.

## Config and run

### DistillationConfig

Dataclass: teacher_hidden, student_hidden, teacher_epochs, student_epochs, temperature, alpha, learning_rate, batch_size, train_ratio, max_samples, random_seed.

### run_distillation(config) -> Dict

Trains teacher then student with distillation. Returns dict with teacher_val_accuracy, teacher_val_ce, student_val_accuracy, student_val_ce, teacher_params, student_params, compression_ratio.

### main()

CLI: --config, --output. Runs distillation and prints or writes JSON results.
