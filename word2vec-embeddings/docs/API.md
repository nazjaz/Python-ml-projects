# Word2Vec API Documentation

## Word2Vec Class

### `Word2Vec(architecture='skipgram', vector_size=100, window=5, min_count=1, negative=5, alpha=0.025, min_alpha=0.0001, epochs=5, random_state=None)`

Word2Vec implementation with skip-gram and CBOW architectures.

#### Parameters

- `architecture` (str): "skipgram" or "cbow" (default: "skipgram").
- `vector_size` (int): Dimension of word vectors (default: 100).
- `window` (int): Maximum distance between current and predicted word (default: 5).
- `min_count` (int): Minimum word count (default: 1).
- `negative` (int): Number of negative samples (default: 5).
- `alpha` (float): Initial learning rate (default: 0.025).
- `min_alpha` (float): Minimum learning rate (default: 0.0001).
- `epochs` (int): Number of training epochs (default: 5).
- `random_state` (int, optional): Random seed (default: None).

#### Attributes

- `vocabulary_` (dict): Vocabulary mapping words to indices.
- `word_counts_` (Counter): Word frequency counts.
- `w1_` (ndarray): Input-to-hidden weight matrix (word embeddings).
- `w2_` (ndarray): Hidden-to-output weight matrix.
- `word_index_` (dict): Word to index mapping.
- `index_word_` (dict): Index to word mapping.

### Methods

#### `fit(sentences, verbose=True)`

Fit Word2Vec model.

**Parameters:**
- `sentences` (list): List of tokenized sentences (list of word lists).
- `verbose` (bool): Whether to print progress (default: True).

**Returns:**
- `self`: Returns self for method chaining.

**Example:**
```python
sentences = [["hello", "world"], ["hello", "test"]]
w2v = Word2Vec(architecture="skipgram")
w2v.fit(sentences)
```

#### `get_word_vector(word)`

Get word vector for a word.

**Parameters:**
- `word` (str): Word to get vector for.

**Returns:**
- `ndarray` or `None`: Word vector or None if word not in vocabulary.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
vector = w2v.get_word_vector("hello")
```

#### `get_embeddings()`

Get all word embeddings.

**Returns:**
- `dict`: Dictionary mapping words to vectors.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
embeddings = w2v.get_embeddings()
```

#### `most_similar(word, topn=10)`

Find most similar words.

**Parameters:**
- `word` (str): Input word.
- `topn` (int): Number of similar words to return (default: 10).

**Returns:**
- `list`: List of (word, similarity_score) tuples.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
similar = w2v.most_similar("hello", topn=5)
```

#### `save_embeddings(filepath)`

Save word embeddings to file.

**Parameters:**
- `filepath` (str): Path to save embeddings.

**Raises:**
- `ValueError`: If model not fitted.

**Example:**
```python
w2v.save_embeddings("embeddings.vec")
```

## Usage Examples

### Skip-gram Architecture

```python
from src.main import Word2Vec

sentences = [
    ["the", "quick", "brown", "fox"],
    ["jumps", "over", "the", "lazy", "dog"],
]

w2v = Word2Vec(architecture="skipgram", vector_size=100, epochs=10)
w2v.fit(sentences)

vector = w2v.get_word_vector("fox")
similar = w2v.most_similar("fox", topn=5)
```

### CBOW Architecture

```python
w2v = Word2Vec(architecture="cbow", vector_size=100, epochs=10)
w2v.fit(sentences)

embeddings = w2v.get_embeddings()
```

### Custom Parameters

```python
w2v = Word2Vec(
    architecture="skipgram",
    vector_size=200,
    window=10,
    min_count=2,
    negative=10,
    epochs=20,
)
w2v.fit(sentences)
```

### Get Embeddings

```python
w2v = Word2Vec(vector_size=100)
w2v.fit(sentences)

embeddings = w2v.get_embeddings()
for word, vector in embeddings.items():
    print(f"{word}: {vector.shape}")
```

### Find Similar Words

```python
w2v = Word2Vec(vector_size=100)
w2v.fit(sentences)

similar = w2v.most_similar("king", topn=10)
for word, score in similar:
    print(f"{word}: {score:.4f}")
```

### Save and Load Embeddings

```python
w2v = Word2Vec(vector_size=100)
w2v.fit(sentences)

w2v.save_embeddings("embeddings.vec")
```
