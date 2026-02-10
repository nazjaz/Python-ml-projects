# Text Feature Extraction API Documentation

## TextFeatureExtractor Class

### `TextFeatureExtractor(lowercase=True, remove_punctuation=True, remove_stopwords=False, stopwords=None, min_df=1, max_df=1.0, max_features=None)`

Text Feature Extractor with TF-IDF, Bag-of-Words, and N-grams.

#### Parameters

- `lowercase` (bool): Convert to lowercase (default: True).
- `remove_punctuation` (bool): Remove punctuation (default: True).
- `remove_stopwords` (bool): Remove stopwords (default: False).
- `stopwords` (list, optional): Custom stopwords list (default: None).
- `min_df` (int): Minimum document frequency (default: 1).
- `max_df` (float): Maximum document frequency, 0-1 (default: 1.0).
- `max_features` (int, optional): Maximum number of features (default: None).

#### Attributes

- `vocabulary_` (dict): Vocabulary mapping terms to indices.
- `idf_` (ndarray): IDF values for TF-IDF.
- `feature_names_` (list): List of feature names.

### Methods

#### `fit_bag_of_words(documents, ngram_range=(1, 1))`

Fit bag-of-words model.

**Parameters:**
- `documents` (list): List of documents (strings).
- `ngram_range` (tuple): Range of n-gram sizes (min_n, max_n).

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
extractor = TextFeatureExtractor()
extractor.fit_bag_of_words(documents, ngram_range=(1, 2))
```

#### `transform_bag_of_words(documents)`

Transform documents to bag-of-words representation.

**Parameters:**
- `documents` (list): List of documents.

**Returns:**
- `ndarray`: Bag-of-words matrix (n_documents, n_features).

**Example:**
```python
matrix = extractor.transform_bag_of_words(documents)
```

#### `fit_transform_bag_of_words(documents, ngram_range=(1, 1))`

Fit and transform documents to bag-of-words.

**Parameters:**
- `documents` (list): List of documents.
- `ngram_range` (tuple): Range of n-gram sizes.

**Returns:**
- `ndarray`: Bag-of-words matrix.

**Example:**
```python
matrix = extractor.fit_transform_bag_of_words(documents)
```

#### `fit_tfidf(documents, ngram_range=(1, 1))`

Fit TF-IDF model.

**Parameters:**
- `documents` (list): List of documents.
- `ngram_range` (tuple): Range of n-gram sizes.

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
extractor = TextFeatureExtractor()
extractor.fit_tfidf(documents, ngram_range=(1, 2))
```

#### `transform_tfidf(documents)`

Transform documents to TF-IDF representation.

**Parameters:**
- `documents` (list): List of documents.

**Returns:**
- `ndarray`: TF-IDF matrix (n_documents, n_features).

**Example:**
```python
matrix = extractor.transform_tfidf(documents)
```

#### `fit_transform_tfidf(documents, ngram_range=(1, 1))`

Fit and transform documents to TF-IDF.

**Parameters:**
- `documents` (list): List of documents.
- `ngram_range` (tuple): Range of n-gram sizes.

**Returns:**
- `ndarray`: TF-IDF matrix.

**Example:**
```python
matrix = extractor.fit_transform_tfidf(documents)
```

#### `get_feature_names()`

Get feature names.

**Returns:**
- `list`: List of feature names.

**Example:**
```python
feature_names = extractor.get_feature_names()
```

#### `get_feature_importance(documents, method='tfidf')`

Get feature importance scores.

**Parameters:**
- `documents` (list): List of documents.
- `method` (str): Method to use ('tfidf' or 'bow') (default: 'tfidf').

**Returns:**
- `dict`: Dictionary mapping features to importance scores.

**Example:**
```python
importance = extractor.get_feature_importance(documents, method="tfidf")
```

## Usage Examples

### TF-IDF Feature Extraction

```python
from src.main import TextFeatureExtractor

documents = [
    "This is the first document",
    "This document is the second document",
]

extractor = TextFeatureExtractor()
tfidf_matrix = extractor.fit_transform_tfidf(documents)
feature_names = extractor.get_feature_names()
```

### Bag-of-Words Feature Extraction

```python
extractor = TextFeatureExtractor()
bow_matrix = extractor.fit_transform_bag_of_words(documents)
```

### N-gram Representations

```python
# Unigrams and bigrams
extractor = TextFeatureExtractor()
matrix = extractor.fit_transform_tfidf(documents, ngram_range=(1, 2))

# Trigrams only
matrix = extractor.fit_transform_tfidf(documents, ngram_range=(3, 3))
```

### With Preprocessing

```python
extractor = TextFeatureExtractor(
    lowercase=True,
    remove_punctuation=True,
    remove_stopwords=True,
)
matrix = extractor.fit_transform_tfidf(documents)
```

### Vocabulary Filtering

```python
extractor = TextFeatureExtractor(
    min_df=2,
    max_df=0.5,
    max_features=1000,
)
matrix = extractor.fit_transform_tfidf(documents)
```
