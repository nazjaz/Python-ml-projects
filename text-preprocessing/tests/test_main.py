"""Unit tests for text preprocessing implementation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import TextPreprocessor


class TestTextPreprocessor:
    """Test TextPreprocessor functionality."""

    def create_temp_csv(self, content: str) -> str:
        """Create temporary CSV file for testing.

        Args:
            content: CSV content as string.

        Returns:
            Path to temporary CSV file.
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        temp_file.write(content)
        temp_file.close()
        return temp_file.name

    def create_temp_config(self, config_dict: dict) -> str:
        """Create temporary config file for testing.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Path to temporary config file.
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def test_initialization_with_default_config(self):
        """Test initialization with default config file."""
        preprocessor = TextPreprocessor()
        assert preprocessor.lowercase is True
        assert preprocessor.remove_stopwords is True

    def test_tokenize(self):
        """Test tokenization."""
        preprocessor = TextPreprocessor()
        text = "The quick brown fox"
        tokens = preprocessor.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) == 4
        assert "the" in tokens

    def test_lowercase_text(self):
        """Test lowercasing."""
        preprocessor = TextPreprocessor()
        text = "The QUICK Brown Fox"
        lowered = preprocessor.lowercase_text(text)

        assert lowered == "the quick brown fox"

    def test_remove_stop_words(self):
        """Test stop word removal."""
        preprocessor = TextPreprocessor()
        tokens = ["the", "quick", "brown", "fox", "and", "the", "dog"]
        filtered = preprocessor.remove_stop_words(tokens)

        assert "the" not in filtered
        assert "and" not in filtered
        assert "quick" in filtered
        assert "brown" in filtered

    def test_stem(self):
        """Test stemming."""
        preprocessor = TextPreprocessor()
        tokens = ["running", "jumps", "quickly", "dogs"]
        stemmed = preprocessor.stem(tokens)

        assert isinstance(stemmed, list)
        assert len(stemmed) == len(tokens)

    def test_preprocess_text_full(self):
        """Test full text preprocessing."""
        preprocessor = TextPreprocessor()
        text = "The quick brown fox jumps over the lazy dog"
        processed = preprocessor.preprocess_text(text, return_tokens=True)

        assert isinstance(processed, list)
        assert "the" not in processed
        assert len(processed) > 0

    def test_preprocess_text_no_stopwords(self):
        """Test preprocessing without stop word removal."""
        preprocessor = TextPreprocessor()
        text = "The quick brown fox"
        processed = preprocessor.preprocess_text(
            text, remove_stopwords=False, return_tokens=True
        )

        assert "the" in processed

    def test_preprocess_text_no_stemming(self):
        """Test preprocessing without stemming."""
        preprocessor = TextPreprocessor()
        text = "The quick brown fox jumps"
        processed = preprocessor.preprocess_text(
            text, stemming=False, return_tokens=True
        )

        assert isinstance(processed, list)

    def test_preprocess_text_return_string(self):
        """Test preprocessing returning string."""
        preprocessor = TextPreprocessor()
        text = "The quick brown fox"
        processed = preprocessor.preprocess_text(text, return_tokens=False)

        assert isinstance(processed, str)
        assert " " in processed or len(processed) == 0

    def test_preprocess_dataframe(self):
        """Test DataFrame preprocessing."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "text": [
                    "The quick brown fox",
                    "A lazy dog",
                    "The cat and the hat",
                ],
            }
        )

        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df, text_column="text")

        assert "text_processed" in df_processed.columns
        assert len(df_processed) == len(df)

    def test_preprocess_dataframe_custom_output_column(self):
        """Test DataFrame preprocessing with custom output column."""
        df = pd.DataFrame({"text": ["The quick brown fox"]})
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(
            df, text_column="text", output_column="processed"
        )

        assert "processed" in df_processed.columns

    def test_preprocess_dataframe_invalid_column(self):
        """Test that invalid column raises error."""
        df = pd.DataFrame({"text": ["The quick brown fox"]})
        preprocessor = TextPreprocessor()

        with pytest.raises(ValueError, match="not found"):
            preprocessor.preprocess_dataframe(df, text_column="invalid")

    def test_add_stop_words(self):
        """Test adding custom stop words."""
        preprocessor = TextPreprocessor()
        initial_count = len(preprocessor.stop_words)

        preprocessor.add_stop_words(["custom", "words"])
        assert len(preprocessor.stop_words) == initial_count + 2
        assert "custom" in preprocessor.stop_words

    def test_remove_stop_words_from_set(self):
        """Test removing words from stop words set."""
        preprocessor = TextPreprocessor()
        initial_count = len(preprocessor.stop_words)

        preprocessor.remove_stop_words_from_set(["the", "a"])
        assert len(preprocessor.stop_words) < initial_count
        assert "the" not in preprocessor.stop_words

    def test_get_stop_words(self):
        """Test getting stop words."""
        preprocessor = TextPreprocessor()
        stop_words = preprocessor.get_stop_words()

        assert isinstance(stop_words, set)
        assert len(stop_words) > 0

    def test_get_preprocessing_stats(self):
        """Test getting preprocessing statistics."""
        preprocessor = TextPreprocessor()
        texts = [
            "The quick brown fox",
            "A lazy dog",
            "The cat and the hat",
        ]

        stats = preprocessor.get_preprocessing_stats(texts)

        assert "total_texts" in stats
        assert "total_words_before" in stats
        assert "total_words_after" in stats
        assert stats["total_texts"] == 3

    def test_config_file_not_found(self):
        """Test that missing config file raises error."""
        with pytest.raises(FileNotFoundError):
            TextPreprocessor(config_path="nonexistent.yaml")

    def test_custom_stop_words_from_config(self):
        """Test custom stop words from config."""
        config = {
            "preprocessing": {
                "lowercase": True,
                "remove_stopwords": True,
                "stemming": True,
                "custom_stop_words": ["custom", "word"],
            },
            "logging": {"level": "INFO", "file": "logs/test.log"},
        }

        config_path = self.create_temp_config(config)
        try:
            preprocessor = TextPreprocessor(config_path=config_path)
            assert "custom" in preprocessor.stop_words
            assert "word" in preprocessor.stop_words
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
