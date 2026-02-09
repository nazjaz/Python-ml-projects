"""Text Preprocessing Tool.

This module provides functionality to perform text preprocessing including
tokenization, lowercasing, stop word removal, and stemming.
"""

import logging
import logging.handlers
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Performs preprocessing operations on text data."""

    # Common English stop words
    DEFAULT_STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "the", "this", "but", "they",
        "have", "had", "what", "said", "each", "which", "their", "time",
        "if", "up", "out", "many", "then", "them", "these", "so", "some",
        "her", "would", "make", "like", "into", "him", "has", "two", "more",
        "very", "after", "words", "long", "about", "other", "many", "first",
        "well", "water", "been", "call", "who", "oil", "sit", "now", "find",
        "down", "day", "did", "get", "come", "made", "may", "part"
    }

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize TextPreprocessor with configuration.

        Args:
            config_path: Path to configuration YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_parameters()
        self.stop_words: set = self.DEFAULT_STOP_WORDS.copy()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dictionary containing configuration settings.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if not config:
                raise ValueError("Configuration file is empty")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in configuration file: {e}")
            raise

    def _setup_logging(self) -> None:
        """Configure logging based on configuration settings."""
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_file = self.config.get("logging", {}).get("file", "logs/app.log")
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - " "%(message)s"
        )

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=10485760, backupCount=5
                ),
                logging.StreamHandler(),
            ],
        )

    def _initialize_parameters(self) -> None:
        """Initialize algorithm parameters from configuration."""
        preprocess_config = self.config.get("preprocessing", {})
        self.lowercase = preprocess_config.get("lowercase", True)
        self.remove_stopwords = preprocess_config.get("remove_stopwords", True)
        self.stemming = preprocess_config.get("stemming", True)
        self.custom_stop_words = preprocess_config.get("custom_stop_words", [])

        if self.custom_stop_words:
            self.stop_words.update(self.custom_stop_words)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Input text string.

        Returns:
            List of tokens (words).
        """
        if not isinstance(text, str):
            text = str(text)

        tokens = re.findall(r"\b\w+\b", text.lower() if self.lowercase else text)
        logger.debug(f"Tokenized text into {len(tokens)} tokens")
        return tokens

    def lowercase_text(self, text: str) -> str:
        """Convert text to lowercase.

        Args:
            text: Input text string.

        Returns:
            Lowercased text.
        """
        if not isinstance(text, str):
            text = str(text)

        return text.lower()

    def remove_stop_words(self, tokens: List[str], custom_stop_words: Optional[List[str]] = None) -> List[str]:
        """Remove stop words from token list.

        Args:
            tokens: List of tokens.
            custom_stop_words: Custom list of stop words to use (optional).

        Returns:
            List of tokens with stop words removed.
        """
        stop_words_set = set(custom_stop_words) if custom_stop_words else self.stop_words
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words_set]
        logger.debug(f"Removed {len(tokens) - len(filtered_tokens)} stop words")
        return filtered_tokens

    def _stem_word(self, word: str) -> str:
        """Stem a single word using Porter-like algorithm.

        Args:
            word: Input word.

        Returns:
            Stemmed word.
        """
        if len(word) <= 2:
            return word

        word = word.lower()

        if word.endswith("ies"):
            word = word[:-3] + "i"
        elif word.endswith("es"):
            if len(word) > 3 and word[-4] not in "aeiou":
                word = word[:-2]
        elif word.endswith("s"):
            if len(word) > 3:
                word = word[:-1]

        if word.endswith("ed"):
            if len(word) > 4:
                word = word[:-2]
        elif word.endswith("ing"):
            if len(word) > 5:
                word = word[:-3]
        elif word.endswith("ly"):
            if len(word) > 4:
                word = word[:-2]

        if word.endswith("er"):
            if len(word) > 3:
                word = word[:-2]
        elif word.endswith("est"):
            if len(word) > 4:
                word = word[:-3]

        return word

    def stem(self, tokens: List[str]) -> List[str]:
        """Stem tokens using simple Porter-like algorithm.

        Args:
            tokens: List of tokens.

        Returns:
            List of stemmed tokens.
        """
        stemmed = [self._stem_word(token) for token in tokens]
        logger.debug(f"Stemmed {len(tokens)} tokens")
        return stemmed

    def preprocess_text(
        self,
        text: str,
        lowercase: Optional[bool] = None,
        remove_stopwords: Optional[bool] = None,
        stemming: Optional[bool] = None,
        return_tokens: bool = True,
    ) -> str | List[str]:
        """Apply all preprocessing steps to text.

        Args:
            text: Input text string.
            lowercase: Whether to lowercase (default from config).
            remove_stopwords: Whether to remove stop words (default from config).
            stemming: Whether to apply stemming (default from config).
            return_tokens: Whether to return tokens or joined string.

        Returns:
            Preprocessed text as string or list of tokens.
        """
        if not isinstance(text, str):
            text = str(text)

        lowercase = lowercase if lowercase is not None else self.lowercase
        remove_stopwords = (
            remove_stopwords if remove_stopwords is not None else self.remove_stopwords
        )
        stemming = stemming if stemming is not None else self.stemming

        if lowercase:
            text = self.lowercase_text(text)

        tokens = self.tokenize(text)

        if remove_stopwords:
            tokens = self.remove_stop_words(tokens)

        if stemming:
            tokens = self.stem(tokens)

        if return_tokens:
            return tokens
        else:
            return " ".join(tokens)

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        lowercase: Optional[bool] = None,
        remove_stopwords: Optional[bool] = None,
        stemming: Optional[bool] = None,
        output_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Preprocess text column in DataFrame.

        Args:
            df: Input DataFrame.
            text_column: Name of text column to preprocess.
            lowercase: Whether to lowercase (default from config).
            remove_stopwords: Whether to remove stop words (default from config).
            stemming: Whether to apply stemming (default from config).
            output_column: Name of output column (default: text_column + '_processed').

        Returns:
            DataFrame with preprocessed text column.

        Raises:
            ValueError: If text_column not found in DataFrame.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        if output_column is None:
            output_column = f"{text_column}_processed"

        result = df.copy()

        result[output_column] = result[text_column].apply(
            lambda x: self.preprocess_text(
                x,
                lowercase=lowercase,
                remove_stopwords=remove_stopwords,
                stemming=stemming,
                return_tokens=False,
            )
        )

        logger.info(
            f"Preprocessed {len(result)} texts in column '{text_column}'"
        )

        return result

    def add_stop_words(self, words: List[str]) -> None:
        """Add custom stop words to the stop words set.

        Args:
            words: List of words to add as stop words.
        """
        self.stop_words.update(word.lower() for word in words)
        logger.info(f"Added {len(words)} stop words")

    def remove_stop_words_from_set(self, words: List[str]) -> None:
        """Remove words from the stop words set.

        Args:
            words: List of words to remove from stop words.
        """
        for word in words:
            self.stop_words.discard(word.lower())
        logger.info(f"Removed {len(words)} stop words")

    def get_stop_words(self) -> set:
        """Get current set of stop words.

        Returns:
            Set of stop words.
        """
        return self.stop_words.copy()

    def get_preprocessing_stats(self, texts: List[str]) -> Dict[str, any]:
        """Get statistics about preprocessing.

        Args:
            texts: List of input texts.

        Returns:
            Dictionary with preprocessing statistics.
        """
        total_chars = sum(len(str(text)) for text in texts)
        total_words = sum(len(self.tokenize(str(text))) for text in texts)

        preprocessed_texts = [
            self.preprocess_text(text, return_tokens=True) for text in texts
        ]
        total_words_after = sum(len(tokens) for tokens in preprocessed_texts)

        return {
            "total_texts": len(texts),
            "total_characters": total_chars,
            "total_words_before": total_words,
            "total_words_after": total_words_after,
            "words_removed": total_words - total_words_after,
            "reduction_percentage": (
                (total_words - total_words_after) / total_words * 100
                if total_words > 0
                else 0
            ),
        }


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Text Preprocessing Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="Name of text column to preprocess",
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default=None,
        help="Name of output column (default: text_column + '_processed')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercasing",
    )
    parser.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Disable stop word removal",
    )
    parser.add_argument(
        "--no-stemming",
        action="store_true",
        help="Disable stemming",
    )

    args = parser.parse_args()

    preprocessor = TextPreprocessor(config_path=args.config)

    try:
        df = pd.read_csv(args.input)

        print("\n=== Text Preprocessing ===")
        print(f"Input file: {args.input}")
        print(f"Text column: {args.text_column}")
        print(f"Number of texts: {len(df)}")

        lowercase = not args.no_lowercase
        remove_stopwords = not args.no_stopwords
        stemming = not args.no_stemming

        print(f"\nPreprocessing options:")
        print(f"  Lowercase: {lowercase}")
        print(f"  Remove stop words: {remove_stopwords}")
        print(f"  Stemming: {stemming}")

        df_processed = preprocessor.preprocess_dataframe(
            df,
            text_column=args.text_column,
            output_column=args.output_column,
            lowercase=lowercase,
            remove_stopwords=remove_stopwords,
            stemming=stemming,
        )

        print("\n=== Preprocessing Statistics ===")
        stats = preprocessor.get_preprocessing_stats(df[args.text_column].tolist())
        print(f"Total texts: {stats['total_texts']}")
        print(f"Total words before: {stats['total_words_before']}")
        print(f"Total words after: {stats['total_words_after']}")
        print(f"Words removed: {stats['words_removed']}")
        print(f"Reduction: {stats['reduction_percentage']:.2f}%")

        if args.output:
            df_processed.to_csv(args.output, index=False)
            print(f"\nProcessed data saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        raise


if __name__ == "__main__":
    main()
