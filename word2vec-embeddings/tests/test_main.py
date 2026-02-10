"""Unit tests for Word2Vec implementation."""

import numpy as np
import pytest

from src.main import Word2Vec


class TestWord2Vec:
    """Test Word2Vec functionality."""

    def test_initialization_skipgram(self):
        """Test skip-gram initialization."""
        w2v = Word2Vec(architecture="skipgram", vector_size=50)
        assert w2v.architecture == "skipgram"
        assert w2v.vector_size == 50

    def test_initialization_cbow(self):
        """Test CBOW initialization."""
        w2v = Word2Vec(architecture="cbow", vector_size=50)
        assert w2v.architecture == "cbow"
        assert w2v.vector_size == 50

    def test_invalid_architecture(self):
        """Test invalid architecture error."""
        with pytest.raises(ValueError, match="skipgram or cbow"):
            Word2Vec(architecture="invalid")

    def test_preprocess_text(self):
        """Test text preprocessing."""
        w2v = Word2Vec()
        tokens = w2v._preprocess_text("Hello World Test")
        assert tokens == ["hello", "world", "test"]

    def test_build_vocabulary(self):
        """Test vocabulary building."""
        sentences = [
            ["hello", "world"],
            ["hello", "test"],
            ["world", "test"],
        ]
        w2v = Word2Vec(min_count=1)
        vocabulary, word_counts = w2v._build_vocabulary(sentences)

        assert len(vocabulary) > 0
        assert "hello" in vocabulary
        assert "world" in vocabulary

    def test_build_vocabulary_min_count(self):
        """Test vocabulary building with min_count."""
        sentences = [
            ["hello", "world"],
            ["hello", "test"],
            ["rare", "word"],
        ]
        w2v = Word2Vec(min_count=2)
        vocabulary, word_counts = w2v._build_vocabulary(sentences)

        assert "rare" not in vocabulary
        assert "word" not in vocabulary

    def test_generate_training_pairs_skipgram(self):
        """Test skip-gram training pair generation."""
        sentences = [["hello", "world", "test"]]
        w2v = Word2Vec(architecture="skipgram", window=1)
        w2v.vocabulary_, _ = w2v._build_vocabulary(sentences)

        pairs = w2v._generate_training_pairs_skipgram(sentences[0])
        assert len(pairs) > 0
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs)

    def test_generate_training_pairs_cbow(self):
        """Test CBOW training pair generation."""
        sentences = [["hello", "world", "test"]]
        w2v = Word2Vec(architecture="cbow", window=1)
        w2v.vocabulary_, _ = w2v._build_vocabulary(sentences)

        pairs = w2v._generate_training_pairs_cbow(sentences[0])
        assert len(pairs) > 0
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs)

    def test_negative_sampling(self):
        """Test negative sampling."""
        sentences = [["hello", "world", "test"]]
        w2v = Word2Vec()
        w2v.vocabulary_, _ = w2v._build_vocabulary(sentences)

        target_idx = 0
        negative_samples = w2v._negative_sampling(target_idx, 2)

        assert len(negative_samples) == 2
        assert target_idx not in negative_samples

    def test_sigmoid(self):
        """Test sigmoid function."""
        w2v = Word2Vec()
        x = np.array([0, 1, -1, 10, -10])
        sigmoid_values = w2v._sigmoid(x)

        assert np.all(sigmoid_values >= 0)
        assert np.all(sigmoid_values <= 1)

    def test_fit_skipgram(self):
        """Test skip-gram fitting."""
        sentences = [
            ["hello", "world"],
            ["hello", "test"],
            ["world", "test"],
        ]
        w2v = Word2Vec(architecture="skipgram", vector_size=10, epochs=1)
        w2v.fit(sentences, verbose=False)

        assert w2v.vocabulary_ is not None
        assert w2v.w1_ is not None
        assert w2v.w2_ is not None
        assert len(w2v.vocabulary_) > 0

    def test_fit_cbow(self):
        """Test CBOW fitting."""
        sentences = [
            ["hello", "world"],
            ["hello", "test"],
            ["world", "test"],
        ]
        w2v = Word2Vec(architecture="cbow", vector_size=10, epochs=1)
        w2v.fit(sentences, verbose=False)

        assert w2v.vocabulary_ is not None
        assert w2v.w1_ is not None
        assert w2v.w2_ is not None

    def test_get_word_vector(self):
        """Test getting word vector."""
        sentences = [["hello", "world", "test"]]
        w2v = Word2Vec(vector_size=10, epochs=1)
        w2v.fit(sentences, verbose=False)

        vector = w2v.get_word_vector("hello")
        assert vector is not None
        assert len(vector) == 10

    def test_get_word_vector_not_in_vocab(self):
        """Test getting vector for word not in vocabulary."""
        sentences = [["hello", "world"]]
        w2v = Word2Vec(vector_size=10, epochs=1)
        w2v.fit(sentences, verbose=False)

        vector = w2v.get_word_vector("unknown")
        assert vector is None

    def test_get_embeddings(self):
        """Test getting all embeddings."""
        sentences = [["hello", "world", "test"]]
        w2v = Word2Vec(vector_size=10, epochs=1)
        w2v.fit(sentences, verbose=False)

        embeddings = w2v.get_embeddings()
        assert isinstance(embeddings, dict)
        assert len(embeddings) == len(w2v.vocabulary_)
        assert all(len(vec) == 10 for vec in embeddings.values())

    def test_most_similar(self):
        """Test finding most similar words."""
        sentences = [
            ["hello", "world"],
            ["hello", "test"],
            ["world", "test"],
            ["hello", "world", "test"],
        ]
        w2v = Word2Vec(vector_size=10, epochs=2)
        w2v.fit(sentences, verbose=False)

        similar = w2v.most_similar("hello", topn=5)
        assert len(similar) <= 5
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similar)

    def test_most_similar_not_in_vocab(self):
        """Test finding similar words for word not in vocabulary."""
        sentences = [["hello", "world"]]
        w2v = Word2Vec(vector_size=10, epochs=1)
        w2v.fit(sentences, verbose=False)

        similar = w2v.most_similar("unknown")
        assert similar == []

    def test_get_word_vector_before_fit(self):
        """Test error when getting vector before fitting."""
        w2v = Word2Vec()
        with pytest.raises(ValueError, match="must be fitted"):
            w2v.get_word_vector("hello")

    def test_get_embeddings_before_fit(self):
        """Test error when getting embeddings before fitting."""
        w2v = Word2Vec()
        with pytest.raises(ValueError, match="must be fitted"):
            w2v.get_embeddings()

    def test_most_similar_before_fit(self):
        """Test error when finding similar words before fitting."""
        w2v = Word2Vec()
        with pytest.raises(ValueError, match="must be fitted"):
            w2v.most_similar("hello")
