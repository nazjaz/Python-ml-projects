# Text Preprocessing API Documentation

## Classes

### TextPreprocessor

Main class for text preprocessing operations.

#### Methods

##### `__init__(config_path: str = "config.yaml")`

Initialize TextPreprocessor with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
preprocessor = TextPreprocessor()
```

##### `tokenize(text: str) -> List[str]`

Tokenize text into words.

**Parameters:**
- `text` (str): Input text string

**Returns:**
- `List[str]`: List of tokens (words)

**Example:**
```python
tokens = preprocessor.tokenize("The quick brown fox")
# Returns: ['the', 'quick', 'brown', 'fox']
```

##### `lowercase_text(text: str) -> str`

Convert text to lowercase.

**Parameters:**
- `text` (str): Input text string

**Returns:**
- `str`: Lowercased text

**Example:**
```python
lowered = preprocessor.lowercase_text("The QUICK Brown")
# Returns: "the quick brown"
```

##### `remove_stop_words(tokens: List[str], custom_stop_words: Optional[List[str]] = None) -> List[str]`

Remove stop words from token list.

**Parameters:**
- `tokens` (List[str]): List of tokens
- `custom_stop_words` (Optional[List[str]]): Custom list of stop words to use (optional)

**Returns:**
- `List[str]`: List of tokens with stop words removed

**Example:**
```python
tokens = ["the", "quick", "brown", "fox", "and", "the", "dog"]
filtered = preprocessor.remove_stop_words(tokens)
# Returns: ['quick', 'brown', 'fox', 'dog']
```

##### `stem(tokens: List[str]) -> List[str]`

Stem tokens using simple Porter-like algorithm.

**Parameters:**
- `tokens` (List[str]): List of tokens

**Returns:**
- `List[str]`: List of stemmed tokens

**Example:**
```python
tokens = ["running", "jumps", "quickly"]
stemmed = preprocessor.stem(tokens)
# Returns: ['run', 'jump', 'quick']
```

##### `preprocess_text(text: str, lowercase: Optional[bool] = None, remove_stopwords: Optional[bool] = None, stemming: Optional[bool] = None, return_tokens: bool = True) -> str | List[str]`

Apply all preprocessing steps to text.

**Parameters:**
- `text` (str): Input text string
- `lowercase` (Optional[bool]): Whether to lowercase (default from config)
- `remove_stopwords` (Optional[bool]): Whether to remove stop words (default from config)
- `stemming` (Optional[bool]): Whether to apply stemming (default from config)
- `return_tokens` (bool): Whether to return tokens or joined string

**Returns:**
- `str | List[str]`: Preprocessed text as string or list of tokens

**Example:**
```python
text = "The quick brown fox jumps over the lazy dog"
processed = preprocessor.preprocess_text(text)
# Returns: ['quick', 'brown', 'fox', 'jump', 'lazi', 'dog']
```

##### `preprocess_dataframe(df: pd.DataFrame, text_column: str, lowercase: Optional[bool] = None, remove_stopwords: Optional[bool] = None, stemming: Optional[bool] = None, output_column: Optional[str] = None) -> pd.DataFrame`

Preprocess text column in DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `text_column` (str): Name of text column to preprocess
- `lowercase` (Optional[bool]): Whether to lowercase (default from config)
- `remove_stopwords` (Optional[bool]): Whether to remove stop words (default from config)
- `stemming` (Optional[bool]): Whether to apply stemming (default from config)
- `output_column` (Optional[str]): Name of output column (default: text_column + '_processed')

**Returns:**
- `pd.DataFrame`: DataFrame with preprocessed text column

**Raises:**
- `ValueError`: If text_column not found in DataFrame

**Example:**
```python
df = pd.DataFrame({"text": ["The quick brown fox", "A lazy dog"]})
df_processed = preprocessor.preprocess_dataframe(df, text_column="text")
```

##### `add_stop_words(words: List[str]) -> None`

Add custom stop words to the stop words set.

**Parameters:**
- `words` (List[str]): List of words to add as stop words

**Example:**
```python
preprocessor.add_stop_words(["custom", "words"])
```

##### `remove_stop_words_from_set(words: List[str]) -> None`

Remove words from the stop words set.

**Parameters:**
- `words` (List[str]): List of words to remove from stop words

**Example:**
```python
preprocessor.remove_stop_words_from_set(["not", "no"])
```

##### `get_stop_words() -> set`

Get current set of stop words.

**Returns:**
- `set`: Set of stop words

**Example:**
```python
stop_words = preprocessor.get_stop_words()
```

##### `get_preprocessing_stats(texts: List[str]) -> Dict[str, any]`

Get statistics about preprocessing.

**Parameters:**
- `texts` (List[str]): List of input texts

**Returns:**
- `Dict[str, any]`: Dictionary with preprocessing statistics

**Example:**
```python
texts = ["The quick brown fox", "A lazy dog"]
stats = preprocessor.get_preprocessing_stats(texts)
print(f"Reduction: {stats['reduction_percentage']:.2f}%")
```

## Configuration

### Configuration File Format

The tool uses YAML configuration files with the following structure:

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

- `lowercase` (bool): Convert text to lowercase
- `remove_stopwords` (bool): Remove stop words
- `stemming` (bool): Apply stemming
- `custom_stop_words` (List[str]): List of custom stop words to add
- `logging.level` (str): Logging verbosity level
- `logging.file` (str): Path to log file

## Examples

### Basic Usage

```python
from src.main import TextPreprocessor

preprocessor = TextPreprocessor()
text = "The quick brown fox jumps over the lazy dog"
processed = preprocessor.preprocess_text(text)
```

### Complete Workflow

```python
import pandas as pd
from src.main import TextPreprocessor

preprocessor = TextPreprocessor()

# Load data
df = pd.read_csv("reviews.csv")

# Preprocess
df_processed = preprocessor.preprocess_dataframe(
    df,
    text_column="review_text",
    output_column="review_processed"
)

# Get statistics
stats = preprocessor.get_preprocessing_stats(df["review_text"].tolist())

# Save
df_processed.to_csv("processed_reviews.csv", index=False)
```

### Custom Stop Words

```python
preprocessor = TextPreprocessor()
preprocessor.add_stop_words(["custom", "words"])
preprocessor.remove_stop_words_from_set(["not"])
```

### Individual Steps

```python
preprocessor = TextPreprocessor()

text = "The QUICK Brown Fox"
tokens = preprocessor.tokenize(text)
lowered = preprocessor.lowercase_text(text)
filtered = preprocessor.remove_stop_words(tokens)
stemmed = preprocessor.stem(filtered)
```
