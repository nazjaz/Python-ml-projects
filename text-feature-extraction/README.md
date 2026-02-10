# Text Feature Extraction using TF-IDF, Bag-of-Words, and N-grams

A Python implementation of text feature extraction from scratch including TF-IDF (Term Frequency-Inverse Document Frequency), bag-of-words, and n-gram representations. This is the forty-second project in the ML learning series, focusing on understanding text feature extraction techniques for natural language processing and machine learning.

## Project Title and Description

The Text Feature Extraction tool provides complete implementations of three major text feature extraction methods from scratch: bag-of-words, TF-IDF, and n-gram representations. It includes text preprocessing (tokenization, lowercasing, punctuation removal, stopword removal), vocabulary building, and feature importance calculation. It helps users understand how text is converted to numerical features for machine learning models.

This tool solves the problem of converting text data to numerical features for machine learning by providing clear, educational implementations without relying on external NLP libraries. It demonstrates bag-of-words, TF-IDF, and n-gram feature extraction from scratch.

**Target Audience**: Beginners learning machine learning, students studying natural language processing, and anyone who needs to understand text feature extraction from scratch.

## Features

- Bag-of-words feature extraction
- TF-IDF feature extraction
- N-gram representations (unigrams, bigrams, trigrams, and custom ranges)
- Text preprocessing (tokenization, lowercasing, punctuation removal)
- Stopword removal (with default English stopwords)
- Vocabulary filtering (min_df, max_df, max_features)
- Feature importance calculation
- Support for both methods simultaneously
- Command-line interface
- Configurable via YAML
- Comprehensive logging
- Input validation and error handling
- Support for CSV input files

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Step 1: Navigate to Project Directory

```bash
cd /path/to/Python-ml-projects/text-feature-extraction
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
python src/main.py --input sample.csv --text-column text --method tfidf --output features.csv
```

## Configuration

### Configuration File Structure

The tool is configured via `config.yaml`:

```yaml
logging:
  level: "INFO"
  file: "logs/app.log"

features:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: false
  min_df: 1
  max_df: 1.0
  max_features: null
  ngram_min: 1
  ngram_max: 1
```

### Configuration Parameters

- `logging.level`: Logging verbosity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file`: Path to log file
- `features.lowercase`: Convert text to lowercase (default: true)
- `features.remove_punctuation`: Remove punctuation (default: true)
- `features.remove_stopwords`: Remove stopwords (default: false)
- `features.min_df`: Minimum document frequency (default: 1)
- `features.max_df`: Maximum document frequency, 0-1 (default: 1.0)
- `features.max_features`: Maximum number of features (default: null, unlimited)
- `features.ngram_min`: Minimum n-gram size (default: 1)
- `features.ngram_max`: Maximum n-gram size (default: 1)

## Usage

### Basic Usage

```python
from src.main import TextFeatureExtractor

documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
]

# TF-IDF
extractor = TextFeatureExtractor()
tfidf_matrix = extractor.fit_transform_tfidf(documents)
print(f"TF-IDF shape: {tfidf_matrix.shape}")

# Bag-of-words
bow_matrix = extractor.fit_transform_bag_of_words(documents)
print(f"Bag-of-words shape: {bow_matrix.shape}")
```

### TF-IDF Feature Extraction

```python
from src.main import TextFeatureExtractor

documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
]

extractor = TextFeatureExtractor(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=False,
    min_df=1,
    max_df=1.0,
)

# Fit and transform
tfidf_matrix = extractor.fit_transform_tfidf(documents)

# Get feature names
feature_names = extractor.get_feature_names()
print(f"Features: {feature_names[:10]}")

# Get feature importance
importance = extractor.get_feature_importance(documents, method="tfidf")
print(f"Top features: {list(importance.items())[:5]}")
```

### Bag-of-Words Feature Extraction

```python
from src.main import TextFeatureExtractor

documents = [
    "This is the first document",
    "This document is the second document",
]

extractor = TextFeatureExtractor()
bow_matrix = extractor.fit_transform_bag_of_words(documents)

print(f"Bag-of-words shape: {bow_matrix.shape}")
print(f"Vocabulary size: {len(extractor.vocabulary_)}")
```

### N-gram Representations

```python
from src.main import TextFeatureExtractor

documents = [
    "This is a test document",
    "This is another test document",
]

# Unigrams only (default)
extractor = TextFeatureExtractor()
matrix = extractor.fit_transform_tfidf(documents, ngram_range=(1, 1))

# Bigrams
matrix = extractor.fit_transform_tfidf(documents, ngram_range=(2, 2))

# Unigrams and bigrams
matrix = extractor.fit_transform_tfidf(documents, ngram_range=(1, 2))

# Trigrams
matrix = extractor.fit_transform_tfidf(documents, ngram_range=(3, 3))
```

### With Text Preprocessing

```python
from src.main import TextFeatureExtractor

documents = [
    "This is THE FIRST document!",
    "This document is the second document.",
]

# With lowercase and punctuation removal
extractor = TextFeatureExtractor(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=False,
)
matrix = extractor.fit_transform_tfidf(documents)

# With stopword removal
extractor = TextFeatureExtractor(remove_stopwords=True)
matrix = extractor.fit_transform_tfidf(documents)
```

### Vocabulary Filtering

```python
from src.main import TextFeatureExtractor

documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
]

# Minimum document frequency
extractor = TextFeatureExtractor(min_df=2)
matrix = extractor.fit_transform_tfidf(documents)

# Maximum document frequency (as fraction)
extractor = TextFeatureExtractor(max_df=0.5)
matrix = extractor.fit_transform_tfidf(documents)

# Maximum number of features
extractor = TextFeatureExtractor(max_features=10)
matrix = extractor.fit_transform_tfidf(documents)
```

### Complete Example

```python
from src.main import TextFeatureExtractor
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
documents = df["text"].astype(str).tolist()

# Initialize extractor
extractor = TextFeatureExtractor(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
    min_df=2,
    max_features=1000,
)

# Extract TF-IDF features
tfidf_matrix = extractor.fit_transform_tfidf(documents, ngram_range=(1, 2))

# Create feature dataframe
feature_names = extractor.get_feature_names()
tfidf_df = pd.DataFrame(
    tfidf_matrix,
    columns=[f"tfidf_{name}" for name in feature_names]
)

# Get feature importance
importance = extractor.get_feature_importance(documents, method="tfidf")
importance_df = pd.DataFrame({
    "feature": list(importance.keys()),
    "importance": list(importance.values()),
})

print(f"Extracted {len(feature_names)} features")
print(f"Top 10 features:")
print(importance_df.head(10))
```

### Command-Line Usage

TF-IDF extraction:

```bash
python src/main.py --input data.csv --text-column text --method tfidf --output features.csv
```

Bag-of-words extraction:

```bash
python src/main.py --input data.csv --text-column text --method bow --output features.csv
```

Both methods:

```bash
python src/main.py --input data.csv --text-column text --method both --output features.csv
```

With n-grams:

```bash
python src/main.py --input data.csv --text-column text --method tfidf --ngram-min 1 --ngram-max 2 --output features.csv
```

With stopword removal:

```bash
python src/main.py --input data.csv --text-column text --method tfidf --remove-stopwords --output features.csv
```

With feature limits:

```bash
python src/main.py --input data.csv --text-column text --method tfidf --max-features 1000 --min-df 2 --output features.csv
```

Save feature importance:

```bash
python src/main.py --input data.csv --text-column text --method tfidf --output features.csv --output-importance importance.csv
```

## Project Structure

```
text-feature-extraction/
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

- `src/main.py`: Core implementation with `TextFeatureExtractor` class
- `config.yaml`: Configuration file for feature extraction settings
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
- Text preprocessing
- Bag-of-words extraction
- TF-IDF extraction
- N-gram creation
- Vocabulary filtering
- Feature importance
- Error handling

## Understanding Text Feature Extraction

### Bag-of-Words

**Concept:**
- Represents text as frequency of words
- Ignores word order
- Simple and effective

**Process:**
1. Build vocabulary from all documents
2. Count word occurrences in each document
3. Create feature vector with word counts

**Use Cases:**
- Text classification
- Sentiment analysis
- Document similarity

### TF-IDF (Term Frequency-Inverse Document Frequency)

**Concept:**
- Weights words by importance
- Rare words get higher weights
- Common words get lower weights

**Formula:**
- `TF(t, d) = count(t, d) / total_words(d)`
- `IDF(t) = log(N / (df(t) + 1)) + 1`
- `TF-IDF(t, d) = TF(t, d) * IDF(t)`

Where:
- `t`: term (word)
- `d`: document
- `N`: total number of documents
- `df(t)`: document frequency of term t

**Use Cases:**
- Information retrieval
- Text classification
- Document ranking

### N-grams

**Concept:**
- Sequences of n consecutive words
- Captures word order and context
- Unigrams (1-gram), bigrams (2-gram), trigrams (3-gram)

**Examples:**
- Unigram: "hello", "world"
- Bigram: "hello world", "world test"
- Trigram: "hello world test"

**Use Cases:**
- Capturing phrases
- Context-aware features
- Language modeling

### Text Preprocessing

**Steps:**
1. **Lowercasing**: Convert to lowercase
2. **Tokenization**: Split into words
3. **Punctuation removal**: Remove special characters
4. **Stopword removal**: Remove common words

**Benefits:**
- Normalizes text
- Reduces vocabulary size
- Improves feature quality

## Troubleshooting

### Common Issues

**Issue**: Too many features

**Solution**: 
- Use max_features parameter
- Increase min_df
- Decrease max_df
- Remove stopwords

**Issue**: Memory issues with large datasets

**Solution**: 
- Use max_features to limit vocabulary
- Process in batches
- Use sparse matrices (future enhancement)

**Issue**: Poor feature quality

**Solution**: 
- Adjust preprocessing (lowercase, punctuation)
- Remove stopwords
- Use n-grams
- Tune min_df and max_df

### Error Messages

- `ValueError: Model must be fitted before transformation`: Call `fit_*` before `transform_*`
- `ValueError: Text column 'X' not found`: Check column name in CSV

## Best Practices

1. **Preprocessing**: Always lowercase and remove punctuation
2. **Stopwords**: Remove for most tasks, keep for some (e.g., sentiment)
3. **N-grams**: Use unigrams + bigrams for better performance
4. **Vocabulary filtering**: Use min_df to remove rare words
5. **Feature limits**: Use max_features for large datasets
6. **TF-IDF vs BOW**: Use TF-IDF for most tasks, BOW for simple cases
7. **Feature importance**: Analyze to understand important terms

## Real-World Applications

- Text classification
- Sentiment analysis
- Spam detection
- Document clustering
- Information retrieval
- Educational purposes for learning NLP

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
