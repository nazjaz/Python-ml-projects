"""Word2Vec Embeddings with Skip-gram and CBOW Architectures.

This module provides functionality to implement Word2Vec from scratch with
skip-gram and CBOW architectures.
"""

import json
import logging
import logging.handlers
import random
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


class Word2Vec:
    """Word2Vec implementation with skip-gram and CBOW architectures."""

    def __init__(
        self,
        architecture: str = "skipgram",
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        negative: int = 5,
        alpha: float = 0.025,
        min_alpha: float = 0.0001,
        epochs: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize Word2Vec.

        Args:
            architecture: "skipgram" or "cbow" (default: "skipgram").
            vector_size: Dimension of word vectors (default: 100).
            window: Maximum distance between current and predicted word (default: 5).
            min_count: Minimum word count (default: 1).
            negative: Number of negative samples (default: 5).
            alpha: Initial learning rate (default: 0.025).
            min_alpha: Minimum learning rate (default: 0.0001).
            epochs: Number of training epochs (default: 5).
            random_state: Random seed (default: None).
        """
        self.architecture = architecture.lower()
        if self.architecture not in ["skipgram", "cbow"]:
            raise ValueError("Architecture must be 'skipgram' or 'cbow'")

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.epochs = epochs
        self.random_state = random_state

        self.vocabulary_: Optional[Dict[str, int]] = None
        self.word_counts_: Optional[Counter] = None
        self.w1_: Optional[np.ndarray] = None
        self.w2_: Optional[np.ndarray] = None
        self.word_index_: Optional[Dict[str, int]] = None
        self.index_word_: Optional[Dict[int, str]] = None

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: lowercase and tokenize.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        tokens = text.split()
        return tokens

    def _build_vocabulary(self, sentences: List[List[str]]) -> Tuple[Dict[str, int], Counter]:
        """Build vocabulary from sentences.

        Args:
            sentences: List of tokenized sentences.

        Returns:
            Tuple of (vocabulary dict, word counts).
        """
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)

        vocabulary = {}
        word_index = {}
        index_word = {}
        idx = 0

        for word, count in word_counts.items():
            if count >= self.min_count:
                vocabulary[word] = idx
                word_index[word] = idx
                index_word[idx] = word
                idx += 1

        return vocabulary, word_counts

    def _generate_training_pairs_skipgram(
        self, sentence: List[str]
    ) -> List[Tuple[int, int]]:
        """Generate training pairs for skip-gram.

        Args:
            sentence: Tokenized sentence.

        Returns:
            List of (center_word_idx, context_word_idx) pairs.
        """
        pairs = []
        sentence_indices = [
            self.vocabulary_[word] for word in sentence if word in self.vocabulary_
        ]

        for i, center_idx in enumerate(sentence_indices):
            start = max(0, i - self.window)
            end = min(len(sentence_indices), i + self.window + 1)

            for j in range(start, end):
                if j != i:
                    context_idx = sentence_indices[j]
                    pairs.append((center_idx, context_idx))

        return pairs

    def _generate_training_pairs_cbow(
        self, sentence: List[str]
    ) -> List[Tuple[List[int], int]]:
        """Generate training pairs for CBOW.

        Args:
            sentence: Tokenized sentence.

        Returns:
            List of (context_word_indices, center_word_idx) pairs.
        """
        pairs = []
        sentence_indices = [
            self.vocabulary_[word] for word in sentence if word in self.vocabulary_
        ]

        for i, center_idx in enumerate(sentence_indices):
            context_indices = []
            start = max(0, i - self.window)
            end = min(len(sentence_indices), i + self.window + 1)

            for j in range(start, end):
                if j != i:
                    context_indices.append(sentence_indices[j])

            if len(context_indices) > 0:
                pairs.append((context_indices, center_idx))

        return pairs

    def _negative_sampling(self, target_idx: int, n_samples: int) -> List[int]:
        """Generate negative samples.

        Args:
            target_idx: Target word index.
            n_samples: Number of negative samples.

        Returns:
            List of negative sample indices.
        """
        negative_samples = []
        vocab_size = len(self.vocabulary_)

        while len(negative_samples) < n_samples:
            sample = random.randint(0, vocab_size - 1)
            if sample != target_idx and sample not in negative_samples:
                negative_samples.append(sample)

        return negative_samples

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid function.

        Args:
            x: Input values.

        Returns:
            Sigmoid values.
        """
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _train_skipgram_pair(
        self, center_idx: int, context_idx: int, learning_rate: float
    ) -> None:
        """Train skip-gram on a single pair.

        Args:
            center_idx: Center word index.
            context_idx: Context word index.
            learning_rate: Current learning rate.
        """
        center_vector = self.w1_[center_idx]
        context_vector = self.w2_[context_idx]

        score = np.dot(center_vector, context_vector)
        prediction = self._sigmoid(score)

        error = 1.0 - prediction

        grad_w1 = error * context_vector
        grad_w2 = error * center_vector

        self.w1_[center_idx] += learning_rate * grad_w1
        self.w2_[context_idx] += learning_rate * grad_w2

        negative_samples = self._negative_sampling(context_idx, self.negative)

        for neg_idx in negative_samples:
            neg_vector = self.w2_[neg_idx]
            neg_score = np.dot(center_vector, neg_vector)
            neg_prediction = self._sigmoid(neg_score)

            neg_error = 0.0 - neg_prediction

            neg_grad_w1 = neg_error * neg_vector
            neg_grad_w2 = neg_error * center_vector

            self.w1_[center_idx] += learning_rate * neg_grad_w1
            self.w2_[neg_idx] += learning_rate * neg_grad_w2

    def _train_cbow_pair(
        self, context_indices: List[int], center_idx: int, learning_rate: float
    ) -> None:
        """Train CBOW on a single pair.

        Args:
            context_indices: Context word indices.
            center_idx: Center word index.
            learning_rate: Current learning rate.
        """
        context_vectors = self.w1_[context_indices]
        context_sum = np.sum(context_vectors, axis=0) / len(context_indices)

        center_vector = self.w2_[center_idx]

        score = np.dot(context_sum, center_vector)
        prediction = self._sigmoid(score)

        error = 1.0 - prediction

        grad_w2 = error * context_sum
        grad_w1_avg = error * center_vector / len(context_indices)

        self.w2_[center_idx] += learning_rate * grad_w2

        for ctx_idx in context_indices:
            self.w1_[ctx_idx] += learning_rate * grad_w1_avg

        negative_samples = self._negative_sampling(center_idx, self.negative)

        for neg_idx in negative_samples:
            neg_vector = self.w2_[neg_idx]
            neg_score = np.dot(context_sum, neg_vector)
            neg_prediction = self._sigmoid(neg_score)

            neg_error = 0.0 - neg_prediction

            neg_grad_w2 = neg_error * context_sum
            neg_grad_w1_avg = neg_error * neg_vector / len(context_indices)

            self.w2_[neg_idx] += learning_rate * neg_grad_w2

            for ctx_idx in context_indices:
                self.w1_[ctx_idx] += learning_rate * neg_grad_w1_avg

    def fit(self, sentences: List[List[str]], verbose: bool = True) -> "Word2Vec":
        """Fit Word2Vec model.

        Args:
            sentences: List of tokenized sentences.
            verbose: Whether to print progress (default: True).

        Returns:
            Self for method chaining.
        """
        if verbose:
            logger.info(f"Building vocabulary...")

        self.vocabulary_, self.word_counts_ = self._build_vocabulary(sentences)
        vocab_size = len(self.vocabulary_)

        self.word_index_ = self.vocabulary_
        self.index_word_ = {idx: word for word, idx in self.vocabulary_.items()}

        if verbose:
            logger.info(f"Vocabulary size: {vocab_size}")

        self.w1_ = np.random.uniform(
            -0.5 / self.vector_size, 0.5 / self.vector_size, (vocab_size, self.vector_size)
        )
        self.w2_ = np.random.uniform(
            -0.5 / self.vector_size, 0.5 / self.vector_size, (vocab_size, self.vector_size)
        )

        total_pairs = 0
        for epoch in range(self.epochs):
            if verbose:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            learning_rate = self.alpha * (1 - epoch / self.epochs)
            learning_rate = max(learning_rate, self.min_alpha)

            epoch_pairs = 0

            for sentence in sentences:
                if self.architecture == "skipgram":
                    pairs = self._generate_training_pairs_skipgram(sentence)
                    for center_idx, context_idx in pairs:
                        self._train_skipgram_pair(center_idx, context_idx, learning_rate)
                        epoch_pairs += 1
                else:
                    pairs = self._generate_training_pairs_cbow(sentence)
                    for context_indices, center_idx in pairs:
                        self._train_cbow_pair(context_indices, center_idx, learning_rate)
                        epoch_pairs += 1

            total_pairs += epoch_pairs

            if verbose:
                logger.info(f"  Processed {epoch_pairs} training pairs")

        if verbose:
            logger.info(f"Training completed. Total pairs: {total_pairs}")

        return self

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector for a word.

        Args:
            word: Word to get vector for.

        Returns:
            Word vector or None if word not in vocabulary.
        """
        if self.w1_ is None:
            raise ValueError("Model must be fitted before getting word vectors")

        if word not in self.vocabulary_:
            return None

        idx = self.vocabulary_[word]
        return self.w1_[idx]

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all word embeddings.

        Returns:
            Dictionary mapping words to vectors.
        """
        if self.w1_ is None:
            raise ValueError("Model must be fitted before getting embeddings")

        embeddings = {}
        for word, idx in self.vocabulary_.items():
            embeddings[word] = self.w1_[idx]

        return embeddings

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words.

        Args:
            word: Input word.
            topn: Number of similar words to return (default: 10).

        Returns:
            List of (word, similarity_score) tuples.
        """
        if self.w1_ is None:
            raise ValueError("Model must be fitted before finding similar words")

        if word not in self.vocabulary_:
            return []

        word_vector = self.get_word_vector(word)
        vocab_size = len(self.vocabulary_)

        similarities = []
        for other_word, other_idx in self.vocabulary_.items():
            if other_word == word:
                continue

            other_vector = self.w1_[other_idx]
            similarity = np.dot(word_vector, other_vector) / (
                np.linalg.norm(word_vector) * np.linalg.norm(other_vector) + 1e-8
            )
            similarities.append((other_word, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def save_embeddings(self, filepath: str) -> None:
        """Save word embeddings to file.

        Args:
            filepath: Path to save embeddings.
        """
        if self.w1_ is None:
            raise ValueError("Model must be fitted before saving embeddings")

        embeddings = self.get_embeddings()
        with open(filepath, "w") as f:
            f.write(f"{len(embeddings)} {self.vector_size}\n")
            for word, vector in embeddings.items():
                vector_str = " ".join(str(x) for x in vector)
                f.write(f"{word} {vector_str}\n")

        logger.info(f"Embeddings saved to: {filepath}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Word2Vec Embeddings")
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
        help="Path to text file or CSV file with text data",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Name of text column (if CSV file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save word embeddings",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["skipgram", "cbow"],
        default=None,
        help="Architecture: skipgram or cbow (default: from config)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=None,
        help="Dimension of word vectors (default: from config)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Context window size (default: from config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=None,
        help="Minimum word count (default: from config)",
    )
    parser.add_argument(
        "--negative",
        type=int,
        default=None,
        help="Number of negative samples (default: from config)",
    )
    parser.add_argument(
        "--similar-word",
        type=str,
        default=None,
        help="Find similar words to this word",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Number of similar words to return (default: 10)",
    )

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})

        architecture = (
            args.architecture
            if args.architecture is not None
            else model_config.get("architecture", "skipgram")
        )
        vector_size = (
            args.vector_size
            if args.vector_size is not None
            else model_config.get("vector_size", 100)
        )
        window = (
            args.window if args.window is not None else model_config.get("window", 5)
        )
        epochs = (
            args.epochs if args.epochs is not None else model_config.get("epochs", 5)
        )
        min_count = (
            args.min_count
            if args.min_count is not None
            else model_config.get("min_count", 1)
        )
        negative = (
            args.negative
            if args.negative is not None
            else model_config.get("negative", 5)
        )

        print(f"\n=== Word2Vec Embeddings ===")
        print(f"Architecture: {architecture}")
        print(f"Vector size: {vector_size}")
        print(f"Window: {window}")
        print(f"Epochs: {epochs}")
        print(f"Min count: {min_count}")
        print(f"Negative samples: {negative}")

        if args.input.endswith(".csv"):
            df = pd.read_csv(args.input)
            if args.text_column is None:
                raise ValueError("--text-column required for CSV files")

            if args.text_column not in df.columns:
                raise ValueError(f"Text column '{args.text_column}' not found")

            sentences = []
            for text in df[args.text_column].astype(str):
                tokens = text.lower().split()
                sentences.append(tokens)
        else:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()

            sentences = []
            for line in text.split("\n"):
                if line.strip():
                    tokens = line.lower().split()
                    sentences.append(tokens)

        print(f"\nNumber of sentences: {len(sentences)}")
        total_words = sum(len(s) for s in sentences)
        print(f"Total words: {total_words}")

        w2v = Word2Vec(
            architecture=architecture,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            negative=negative,
            epochs=epochs,
        )

        print(f"\nTraining Word2Vec...")
        w2v.fit(sentences, verbose=True)

        print(f"\n=== Training Results ===")
        print(f"Vocabulary size: {len(w2v.vocabulary_)}")
        print(f"Embedding dimension: {vector_size}")

        if args.similar_word:
            print(f"\nMost similar words to '{args.similar_word}':")
            similar = w2v.most_similar(args.similar_word, topn=args.topn)
            for word, score in similar:
                print(f"  {word}: {score:.4f}")

        w2v.save_embeddings(args.output)
        print(f"\nEmbeddings saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error in Word2Vec training: {e}")
        raise


if __name__ == "__main__":
    main()
