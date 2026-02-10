"""Text Feature Extraction using TF-IDF, Bag-of-Words, and N-grams.

This module provides functionality for text feature extraction including
TF-IDF, bag-of-words, and n-gram representations.
"""

import json
import logging
import logging.handlers
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Text Feature Extractor with TF-IDF, Bag-of-Words, and N-grams."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = False,
        stopwords: Optional[List[str]] = None,
        min_df: int = 1,
        max_df: float = 1.0,
        max_features: Optional[int] = None,
    ) -> None:
        """Initialize Text Feature Extractor.

        Args:
            lowercase: Convert to lowercase (default: True).
            remove_punctuation: Remove punctuation (default: True).
            remove_stopwords: Remove stopwords (default: False).
            stopwords: Custom stopwords list (default: None).
            min_df: Minimum document frequency (default: 1).
            max_df: Maximum document frequency (0-1) (default: 1.0).
            max_features: Maximum number of features (default: None).
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(stopwords) if stopwords else self._get_default_stopwords()
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

        self.vocabulary_: Optional[Dict[str, int]] = None
        self.idf_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def _get_default_stopwords(self) -> set:
        """Get default English stopwords.

        Returns:
            Set of stopwords.
        """
        return {
            "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on", "that",
            "the", "to", "was", "were", "will", "with", "the", "this", "but",
            "they", "have", "had", "what", "said", "each", "which", "their",
            "time", "if", "up", "out", "many", "then", "them", "these", "so",
        }

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: tokenize, lowercase, remove punctuation/stopwords.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        return tokens

    def _create_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Create n-grams from tokens.

        Args:
            tokens: List of tokens.
            n: N-gram size.

        Returns:
            List of n-grams.
        """
        if n == 1:
            return tokens

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams.append(ngram)

        return ngrams

    def _build_vocabulary(
        self, documents: List[str], ngram_range: Tuple[int, int] = (1, 1)
    ) -> Dict[str, int]:
        """Build vocabulary from documents.

        Args:
            documents: List of documents.
            ngram_range: Range of n-gram sizes (min_n, max_n).

        Returns:
            Vocabulary dictionary mapping terms to indices.
        """
        term_counts = Counter()
        doc_counts = defaultdict(int)

        for doc in documents:
            doc_tokens = self._preprocess_text(doc)
            doc_terms = set()

            for n in range(ngram_range[0], ngram_range[1] + 1):
                ngrams = self._create_ngrams(doc_tokens, n)
                for ngram in ngrams:
                    term_counts[ngram] += 1
                    doc_terms.add(ngram)

            for term in doc_terms:
                doc_counts[term] += 1

        n_docs = len(documents)
        min_doc_freq = max(1, int(self.min_df * n_docs)) if self.min_df < 1 else self.min_df
        max_doc_freq = int(self.max_df * n_docs) if self.max_df < 1 else n_docs

        vocabulary = {}
        idx = 0

        for term, count in term_counts.most_common():
            doc_freq = doc_counts[term]

            if doc_freq < min_doc_freq:
                continue
            if doc_freq > max_doc_freq:
                continue

            vocabulary[term] = idx
            idx += 1

            if self.max_features and len(vocabulary) >= self.max_features:
                break

        return vocabulary

    def fit_bag_of_words(
        self, documents: List[str], ngram_range: Tuple[int, int] = (1, 1)
    ) -> "TextFeatureExtractor":
        """Fit bag-of-words model.

        Args:
            documents: List of documents.
            ngram_range: Range of n-gram sizes (min_n, max_n).

        Returns:
            Self for method chaining.
        """
        self.vocabulary_ = self._build_vocabulary(documents, ngram_range)
        self.feature_names_ = [term for term, _ in sorted(self.vocabulary_.items(), key=lambda x: x[1])]

        logger.info(f"Bag-of-words fitted: vocabulary size={len(self.vocabulary_)}")
        return self

    def transform_bag_of_words(self, documents: List[str]) -> np.ndarray:
        """Transform documents to bag-of-words representation.

        Args:
            documents: List of documents.

        Returns:
            Bag-of-words matrix (n_documents, n_features).
        """
        if self.vocabulary_ is None:
            raise ValueError("Model must be fitted before transformation")

        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        bow_matrix = np.zeros((n_docs, n_features), dtype=float)

        for i, doc in enumerate(documents):
            tokens = self._preprocess_text(doc)
            doc_terms = []

            for n in range(1, 4):
                ngrams = self._create_ngrams(tokens, n)
                doc_terms.extend(ngrams)

            for term in doc_terms:
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    bow_matrix[i, idx] += 1

        return bow_matrix

    def fit_transform_bag_of_words(
        self, documents: List[str], ngram_range: Tuple[int, int] = (1, 1)
    ) -> np.ndarray:
        """Fit and transform documents to bag-of-words.

        Args:
            documents: List of documents.
            ngram_range: Range of n-gram sizes (min_n, max_n).

        Returns:
            Bag-of-words matrix.
        """
        return self.fit_bag_of_words(documents, ngram_range).transform_bag_of_words(documents)

    def fit_tfidf(
        self, documents: List[str], ngram_range: Tuple[int, int] = (1, 1)
    ) -> "TextFeatureExtractor":
        """Fit TF-IDF model.

        Args:
            documents: List of documents.
            ngram_range: Range of n-gram sizes (min_n, max_n).

        Returns:
            Self for method chaining.
        """
        self.vocabulary_ = self._build_vocabulary(documents, ngram_range)
        self.feature_names_ = [term for term, _ in sorted(self.vocabulary_.items(), key=lambda x: x[1])]

        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        doc_freq = np.zeros(n_features, dtype=int)

        for doc in documents:
            tokens = self._preprocess_text(doc)
            doc_terms = set()

            for n in range(ngram_range[0], ngram_range[1] + 1):
                ngrams = self._create_ngrams(tokens, n)
                for ngram in ngrams:
                    if ngram in self.vocabulary_:
                        doc_terms.add(self.vocabulary_[ngram])

            for idx in doc_terms:
                doc_freq[idx] += 1

        self.idf_ = np.log(n_docs / (doc_freq + 1)) + 1

        logger.info(f"TF-IDF fitted: vocabulary size={len(self.vocabulary_)}, n_documents={n_docs}")
        return self

    def transform_tfidf(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF representation.

        Args:
            documents: List of documents.

        Returns:
            TF-IDF matrix (n_documents, n_features).
        """
        if self.vocabulary_ is None or self.idf_ is None:
            raise ValueError("Model must be fitted before transformation")

        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        tf_matrix = np.zeros((n_docs, n_features), dtype=float)

        for i, doc in enumerate(documents):
            tokens = self._preprocess_text(doc)
            doc_terms = []

            for n in range(1, 4):
                ngrams = self._create_ngrams(tokens, n)
                doc_terms.extend(ngrams)

            term_counts = Counter(doc_terms)
            total_terms = len(doc_terms)

            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    tf = count / total_terms if total_terms > 0 else 0
                    tf_matrix[i, idx] = tf

        tfidf_matrix = tf_matrix * self.idf_

        return tfidf_matrix

    def fit_transform_tfidf(
        self, documents: List[str], ngram_range: Tuple[int, int] = (1, 1)
    ) -> np.ndarray:
        """Fit and transform documents to TF-IDF.

        Args:
            documents: List of documents.
            ngram_range: Range of n-gram sizes (min_n, max_n).

        Returns:
            TF-IDF matrix.
        """
        return self.fit_tfidf(documents, ngram_range).transform_tfidf(documents)

    def get_feature_names(self) -> List[str]:
        """Get feature names.

        Returns:
            List of feature names.
        """
        if self.feature_names_ is None:
            raise ValueError("Model must be fitted before getting feature names")
        return self.feature_names_

    def get_feature_importance(
        self, documents: List[str], method: str = "tfidf"
    ) -> Dict[str, float]:
        """Get feature importance scores.

        Args:
            documents: List of documents.
            method: Method to use ('tfidf' or 'bow') (default: 'tfidf').

        Returns:
            Dictionary mapping features to importance scores.
        """
        if method == "tfidf":
            matrix = self.transform_tfidf(documents)
            scores = np.mean(matrix, axis=0)
        else:
            matrix = self.transform_bag_of_words(documents)
            scores = np.mean(matrix, axis=0)

        feature_names = self.get_feature_names()
        importance = dict(zip(feature_names, scores))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Text Feature Extraction")
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
        help="Path to CSV file with text data",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="Name of text column",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save extracted features CSV",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bow", "tfidf", "both"],
        default="tfidf",
        help="Feature extraction method (default: tfidf)",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum n-gram size (default: 1)",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=1,
        help="Maximum n-gram size (default: 1)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features (default: None)",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency (default: 1)",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=1.0,
        help="Maximum document frequency (0-1) (default: 1.0)",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords",
    )
    parser.add_argument(
        "--output-importance",
        type=str,
        default=None,
        help="Path to save feature importance CSV",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        feature_config = config.get("features", {})

        df = pd.read_csv(args.input)
        print(f"\n=== Text Feature Extraction ===")
        print(f"Data shape: {df.shape}")

        if args.text_column not in df.columns:
            raise ValueError(f"Text column '{args.text_column}' not found in data")

        documents = df[args.text_column].astype(str).tolist()
        print(f"Number of documents: {len(documents)}")

        extractor = TextFeatureExtractor(
            lowercase=feature_config.get("lowercase", True),
            remove_punctuation=feature_config.get("remove_punctuation", True),
            remove_stopwords=args.remove_stopwords or feature_config.get("remove_stopwords", False),
            min_df=args.min_df or feature_config.get("min_df", 1),
            max_df=args.max_df or feature_config.get("max_df", 1.0),
            max_features=args.max_features or feature_config.get("max_features"),
        )

        ngram_range = (args.ngram_min, args.ngram_max)

        print(f"\nExtracting features...")
        print(f"Method: {args.method}")
        print(f"N-gram range: {ngram_range}")
        print(f"Max features: {args.max_features or 'unlimited'}")

        if args.method in ["bow", "both"]:
            print("\nCreating bag-of-words features...")
            bow_matrix = extractor.fit_transform_bag_of_words(documents, ngram_range)
            print(f"Bag-of-words shape: {bow_matrix.shape}")

            bow_df = pd.DataFrame(
                bow_matrix,
                columns=[f"bow_{name}" for name in extractor.get_feature_names()],
            )

        if args.method in ["tfidf", "both"]:
            print("Creating TF-IDF features...")
            tfidf_matrix = extractor.fit_transform_tfidf(documents, ngram_range)
            print(f"TF-IDF shape: {tfidf_matrix.shape}")

            tfidf_df = pd.DataFrame(
                tfidf_matrix,
                columns=[f"tfidf_{name}" for name in extractor.get_feature_names()],
            )

        print(f"\n=== Feature Extraction Results ===")
        print(f"Vocabulary size: {len(extractor.vocabulary_)}")
        print(f"Feature names (first 10): {extractor.get_feature_names()[:10]}")

        if args.method == "both":
            result_df = pd.concat([df, bow_df, tfidf_df], axis=1)
        elif args.method == "bow":
            result_df = pd.concat([df, bow_df], axis=1)
        else:
            result_df = pd.concat([df, tfidf_df], axis=1)

        result_df.to_csv(args.output, index=False)
        print(f"\nExtracted features saved to: {args.output}")

        if args.output_importance:
            importance = extractor.get_feature_importance(documents, method=args.method if args.method != "both" else "tfidf")
            importance_df = pd.DataFrame({
                "feature": list(importance.keys()),
                "importance": list(importance.values()),
            })
            importance_df.to_csv(args.output_importance, index=False)
            print(f"Feature importance saved to: {args.output_importance}")

    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        raise


if __name__ == "__main__":
    main()
