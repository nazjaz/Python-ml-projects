# Word2Vec Embeddings with Skip-gram and CBOW

A Python implementation of Word2Vec from scratch with skip-gram and CBOW (Continuous Bag of Words) architectures. This is the forty-third project in the ML learning series, focusing on understanding word embeddings, neural network training, and distributed word representations.

## Project Title and Description

The Word2Vec Embeddings tool provides a complete implementation of Word2Vec from scratch, including both skip-gram and CBOW architectures, neural network training with backpropagation, negative sampling, and word similarity calculation. It helps users understand how word embeddings are learned, how neural networks are trained for NLP tasks, and how distributed representations capture semantic relationships.

This tool solves the problem of learning word embeddings fundamentals by providing a clear, educational implementation without relying on external NLP libraries. It demonstrates skip-gram (predict context from center word) and CBOW (predict center from context) architectures, neural network training, and negative sampling from scratch.

**Target Audience**: Beginners learning machine learning, students studying natural language processing, and anyone who needs to understand Word2Vec, word embeddings, and neural network training from scratch.

## Features

- Skip-gram architecture implementation
- CBOW architecture implementation
- Neural network training with backpropagation
- Negative sampling for efficient training
- Vocabulary building and text preprocessing
- Word vector extraction
- Most similar words calculation
- Embedding saving and loading
- Configurable hyperparameters
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for text files and CSV files

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/word2vec-embeddings
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python src/main.py --input sample.txt --output embeddings.vec --architecture skipgram
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

model:
  architecture: "skipgram"
  vector_size: 100
  window: 5
  min_count: 1
  negative: 5
  alpha: 0.025
  min_alpha: 0.0001
  epochs: 5
  random_state: null
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `model.architecture`: "skipgram" or "cbow" (default: "skipgram")
- `model.vector_size`: Dimension of word vectors (default: 100)
- `model.window`: Maximum distance between current and predicted word (default: 5)
- `model.min_count`: Minimum word count (default: 1)
- `model.negative`: Number of negative samples (default: 5)
- `model.alpha`: Initial learning rate (default: 0.025)
- `model.min_alpha`: Minimum learning rate (default: 0.0001)
- `model.epochs`: Number of training epochs (default: 5)
- `model.random_state`: Random seed (default: null)

## Usage

### Basic Usage

```python
from src.main import Word2Vec

sentences = [
    ["hello", "world"],
    ["hello", "test"],
    ["world", "test"],
    ["this", "is", "a", "test"],
]

# Skip-gram
w2v = Word2Vec(architecture="skipgram", vector_size=100, epochs=5)
w2v.fit(sentences)

# Get word vector
vector = w2v.get_word_vector("hello")
print(f"Vector shape: {vector.shape}")

# Find similar words
similar = w2v.most_similar("hello", topn=5)
for word, score in similar:
    print(f"{word}: {score:.4f}")
```

### Skip-gram Architecture

```python
from src.main import Word2Vec

sentences = [
    ["the", "quick", "brown", "fox"],
    ["jumps", "over", "the", "lazy", "dog"],
]

w2v = Word2Vec(
    architecture="skipgram",
    vector_size=100,
    window=5,
    epochs=10,
    negative=5,
)
w2v.fit(sentences)

embeddings = w2v.get_embeddings()
print(f"Vocabulary size: {len(embeddings)}")
```

### CBOW Architecture

```python
from src.main import Word2Vec

sentences = [
    ["the", "quick", "brown", "fox"],
    ["jumps", "over", "the", "lazy", "dog"],
]

w2v = Word2Vec(
    architecture="cbow",
    vector_size=100,
    window=5,
    epochs=10,
    negative=5,
)
w2v.fit(sentences)

vector = w2v.get_word_vector("fox")
```

### Most Similar Words

```python
from src.main import Word2Vec

sentences = [
    ["king", "queen", "prince", "princess"],
    ["man", "woman", "boy", "girl"],
    ["king", "man", "queen", "woman"],
]

w2v = Word2Vec(vector_size=50, epochs=10)
w2v.fit(sentences)

similar = w2v.most_similar("king", topn=5)
print("Most similar to 'king':")
for word, score in similar:
    print(f"  {word}: {score:.4f}")
```

### Save Embeddings

```python
from src.main import Word2Vec

sentences = [
    ["hello", "world"],
    ["hello", "test"],
]

w2v = Word2Vec(vector_size=100, epochs=5)
w2v.fit(sentences)

w2v.save_embeddings("embeddings.vec")
```

### Complete Example

```python
from src.main import Word2Vec

# Prepare sentences
sentences = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["the", "dog", "is", "lazy"],
    ["the", "fox", "is", "quick"],
    ["brown", "fox", "jumps"],
]

# Train skip-gram model
w2v_skipgram = Word2Vec(
    architecture="skipgram",
    vector_size=100,
    window=5,
    min_count=1,
    negative=5,
    epochs=10,
)
w2v_skipgram.fit(sentences)

print(f"Vocabulary size: {len(w2v_skipgram.vocabulary_)}")

# Get embeddings
embeddings = w2v_skipgram.get_embeddings()
print(f"Embedding dimension: {len(embeddings['the'])}")

# Find similar words
similar = w2v_skipgram.most_similar("fox", topn=5)
print("\nMost similar to 'fox':")
for word, score in similar:
    print(f"  {word}: {score:.4f}")

# Save embeddings
w2v_skipgram.save_embeddings("skipgram_embeddings.vec")
```

### Command-Line Usage

Skip-gram with text file:

```bash
python src/main.py --input text.txt --output embeddings.vec --architecture skipgram
```

CBOW with CSV file:

```bash
python src/main.py --input data.csv --text-column text --output embeddings.vec --architecture cbow
```

With custom parameters:

```bash
python src/main.py --input text.txt --output embeddings.vec --architecture skipgram --vector-size 200 --window 10 --epochs 10
```

Find similar words:

```bash
python src/main.py --input text.txt --output embeddings.vec --similar-word hello --topn 10
```

## Project Structure

```
word2vec-embeddings/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Configuration file
├── .gitignore               # Git ignore rules
├── src/
│   └── main.py              # Main implementation
├── tests/
│   └── test_main.py         # Unit tests
├── docs/
│   └── API.md               # API documentation
└── logs/
    └── .gitkeep             # Log directory
```

### File Descriptions

- `src/main.py`: Core implementation with `Word2Vec` class
- `config.yaml`: Configuration file for model settings
- `tests/test_main.py`: Comprehensive unit tests
- `docs/API.md`: Detailed API documentation
- `logs/`: Directory for application logs

## Testing

### Run All Tests

```bash
pytest tests/
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

Tests cover:
- Model initialization
- Skip-gram architecture
- CBOW architecture
- Vocabulary building
- Training pair generation
- Negative sampling
- Word vector extraction
- Most similar words
- Error handling

## Understanding Word2Vec

### Skip-gram Architecture

**Concept:**
- Predicts context words from center word
- Input: center word
- Output: context words
- Better for rare words

**Process:**
1. For each center word in sentence
2. Predict surrounding context words
3. Train neural network to maximize probability

**Neural Network:**
- Input layer: one-hot encoded center word
- Hidden layer: word embedding (W1)
- Output layer: context word predictions (W2)

### CBOW Architecture

**Concept:**
- Predicts center word from context words
- Input: context words
- Output: center word
- Faster training

**Process:**
1. For each center word in sentence
2. Use surrounding context words as input
3. Predict center word
4. Train neural network to maximize probability

**Neural Network:**
- Input layer: average of context word embeddings
- Hidden layer: combined context representation
- Output layer: center word prediction

### Negative Sampling

**Purpose:**
- Efficient training alternative to softmax
- Samples negative examples instead of computing all
- Reduces computational complexity

**Process:**
1. For each positive (center, context) pair
2. Sample N negative words
3. Train to distinguish positive from negatives

### Training Process

**Steps:**
1. Initialize word embeddings randomly
2. For each training pair:
   - Forward pass: compute prediction
   - Calculate error
   - Backpropagation: update weights
3. Repeat for multiple epochs
4. Learning rate decays over time

**Loss Function:**
- Binary cross-entropy for positive/negative classification
- Sigmoid activation for probability

### Word Embeddings

**Properties:**
- Dense vector representations
- Capture semantic relationships
- Similar words have similar vectors
- Can perform arithmetic (king - man + woman ≈ queen)

**Use Cases:**
- Text classification
- Machine translation
- Information retrieval
- Semantic similarity

## Troubleshooting

### Common Issues

**Issue**: Poor word similarity results

**Solution**: 
- Increase epochs
- Increase vector_size
- Use more training data
- Adjust window size
- Increase negative samples

**Issue**: Slow training

**Solution**: 
- Reduce vector_size
- Reduce window size
- Reduce negative samples
- Use smaller vocabulary (increase min_count)

**Issue**: Out of memory

**Solution**: 
- Reduce vector_size
- Increase min_count to reduce vocabulary
- Process data in batches (future enhancement)

**Issue**: Embeddings not capturing semantics

**Solution**: 
- Need more training data
- Increase epochs
- Use larger vector_size
- Ensure diverse training data

### Error Messages

- `ValueError: Architecture must be 'skipgram' or 'cbow'`: Use valid architecture
- `ValueError: Model must be fitted before getting word vectors`: Call `fit()` first
- `ValueError: --text-column required for CSV files`: Specify text column for CSV

## Best Practices

1. **Data quality**: Use clean, diverse text data
2. **Vector size**: 100-300 dimensions work well
3. **Window size**: 5-10 for most tasks
4. **Epochs**: 5-20 depending on data size
5. **Negative samples**: 5-20 for good results
6. **Min count**: Filter rare words (min_count=2-5)
7. **Architecture**: Skip-gram for rare words, CBOW for speed
8. **Learning rate**: Start with 0.025, decays automatically

## Real-World Applications

- Text classification
- Sentiment analysis
- Machine translation
- Information retrieval
- Question answering
- Educational purposes for learning word embeddings

## Contributing

### Development Setup

1. Clone the repository
2. Create virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Install development dependencies: `pip install pytest pytest-cov`
5. Run tests: `pytest tests/`

### Code Style Guidelines

- Follow PEP 8 strictly
- Maximum line length: 88 characters
- Use type hints for all functions
- Include docstrings for all public functions and classes
- Write tests for all new functionality

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit pull request with clear description

## License

This project is part of the Python ML Projects collection. See LICENSE file in parent directory for details.
