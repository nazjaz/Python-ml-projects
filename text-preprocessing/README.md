# Text Preprocessing Tool

A Python tool for performing text preprocessing including tokenization, lowercasing, stop word removal, and stemming. This is the fourteenth project in the ML learning series, focusing on preparing text data for natural language processing and machine learning.

## Project Title and Description

The Text Preprocessing Tool provides comprehensive preprocessing capabilities for text data. It supports tokenization, lowercasing, stop word removal, and stemming operations that are essential for preparing text data for NLP tasks and machine learning models.

This tool solves the problem of preparing raw text data for analysis by providing automated preprocessing operations. It handles common text preprocessing challenges like case normalization, removing common words, and reducing words to their root forms, which improves the quality of text features for ML models.

**Target Audience**: Beginners learning natural language processing, data scientists working with text data, and anyone who needs to preprocess text for ML models.

## Features

- Tokenization of text into words
- Text lowercasing
- Stop word removal with customizable stop word lists
- Stemming using Porter-like algorithm
- Batch processing of text columns in DataFrames
- Preprocessing statistics and analysis
- Configurable preprocessing steps
- Save preprocessed data to CSV
- Configurable via YAML
- Comprehensive logging

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/text-preprocessing
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
python src/main.py --input sample.csv --text-column text
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
preprocessing:
  lowercase: true
  remove_stopwords: true
  stemming: true
  custom_stop_words: []

logging:
  level: "INFO"
  file: "logs/app.log"
```

### Configuration Parameters

- `lowercase`: Convert text to lowercase (default: true)
- `remove_stopwords`: Remove stop words (default: true)
- `stemming`: Apply stemming (default: true)
- `custom_stop_words`: List of custom stop words to add
- `logging.level`: Logging verbosity level
- `logging.file`: Path to log file

## Usage

### Basic Usage

```python
from src.main import TextPreprocessor

preprocessor = TextPreprocessor()

# Preprocess single text
text = "The quick brown fox jumps over the lazy dog"
processed = preprocessor.preprocess_text(text)
print(processed)  # ['quick', 'brown', 'fox', 'jump', 'lazi', 'dog']
```

### Command-Line Usage

Preprocess text column in CSV:

```bash
python src/main.py --input data.csv --text-column text --output processed_data.csv
```

Disable specific preprocessing steps:

```bash
python src/main.py --input data.csv --text-column text --no-stemming
```

### Complete Example

```python
import pandas as pd
from src.main import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Load data
df = pd.read_csv("reviews.csv")

# Preprocess text column
df_processed = preprocessor.preprocess_dataframe(
    df,
    text_column="review_text",
    output_column="review_processed"
)

# Get preprocessing statistics
stats = preprocessor.get_preprocessing_stats(df["review_text"].tolist())
print(f"Words removed: {stats['words_removed']}")
print(f"Reduction: {stats['reduction_percentage']:.2f}%")

# Save processed data
df_processed.to_csv("processed_reviews.csv", index=False)
```

### Custom Stop Words

```python
from src.main import TextPreprocessor

preprocessor = TextPreprocessor()

# Add custom stop words
preprocessor.add_stop_words(["custom", "words", "to", "remove"])

# Remove words from stop words list
preprocessor.remove_stop_words_from_set(["not", "no"])

# Get current stop words
stop_words = preprocessor.get_stop_words()
```

### Individual Preprocessing Steps

```python
from src.main import TextPreprocessor

preprocessor = TextPreprocessor()

text = "The QUICK Brown Fox Jumps Over The Lazy Dog"

# Tokenize
tokens = preprocessor.tokenize(text)
print(tokens)  # ['the', 'quick', 'brown', 'fox', ...]

# Lowercase
lowered = preprocessor.lowercase_text(text)
print(lowered)  # "the quick brown fox jumps over the lazy dog"

# Remove stop words
filtered = preprocessor.remove_stop_words(tokens)
print(filtered)  # ['quick', 'brown', 'fox', 'jumps', ...]

# Stem
stemmed = preprocessor.stem(filtered)
print(stemmed)  # ['quick', 'brown', 'fox', 'jump', ...]
```

## Project Structure

```
text-preprocessing/
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

- `src/main.py`: Core implementation with `TextPreprocessor` class
- `config.yaml`: Configuration file for preprocessing parameters
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
- Tokenization functionality
- Lowercasing functionality
- Stop word removal
- Stemming functionality
- DataFrame preprocessing
- Preprocessing statistics
- Error handling

## Troubleshooting

### Common Issues

**Issue**: Text not being lowercased

**Solution**: Check that `lowercase: true` is set in config or pass `lowercase=True` explicitly.

**Issue**: Stop words not being removed

**Solution**: Verify stop words list and ensure `remove_stopwords: true` is set.

**Issue**: Stemming produces unexpected results

**Solution**: The simple stemming algorithm may not handle all cases. Consider using more advanced libraries for production use.

### Error Messages

- `ValueError: Column 'X' not found`: Check column name spelling
- `FileNotFoundError`: Ensure input file exists

## Preprocessing Techniques

### Tokenization

- **Purpose**: Split text into individual words
- **Method**: Regular expression-based word extraction
- **Use for**: Breaking text into analyzable units

### Lowercasing

- **Purpose**: Normalize text case
- **Method**: Convert all characters to lowercase
- **Use for**: Case-insensitive text analysis

### Stop Word Removal

- **Purpose**: Remove common words that don't carry meaning
- **Method**: Filter tokens against stop word list
- **Use for**: Reducing noise and focusing on meaningful words

### Stemming

- **Purpose**: Reduce words to their root forms
- **Method**: Simple Porter-like algorithm
- **Use for**: Normalizing word variations

## Best Practices

1. **Choose preprocessing steps**: Not all steps are needed for every task
2. **Customize stop words**: Add domain-specific stop words
3. **Validate preprocessing**: Check results to ensure quality
4. **Preserve original**: Keep original text for comparison
5. **Consider context**: Some preprocessing may lose important information

## Real-World Applications

- Text classification preprocessing
- Sentiment analysis data preparation
- Document similarity preprocessing
- Search engine text processing
- Chatbot text preprocessing
- NLP pipeline preparation

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
