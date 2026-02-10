"""Unit tests for Text Feature Extraction implementation."""

import numpy as np
import pytest

from src.main import TextFeatureExtractor


class TestTextFeatureExtractor:
    """Test Text Feature Extractor functionality."""

    def test_initialization(self):
        """Test initialization."""
        extractor = TextFeatureExtractor()
        assert extractor.lowercase is True
        assert extractor.remove_punctuation is True
        assert extractor.remove_stopwords is False

    def test_preprocess_text(self):
        """Test text preprocessing."""
        extractor = TextFeatureExtractor()
        tokens = extractor._preprocess_text("Hello World! This is a test.")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_preprocess_text_lowercase(self):
        """Test lowercase preprocessing."""
        extractor = TextFeatureExtractor(lowercase=True)
        tokens = extractor._preprocess_text("Hello World")
        assert all(token.islower() for token in tokens)

    def test_preprocess_text_remove_punctuation(self):
        """Test punctuation removal."""
        extractor = TextFeatureExtractor(remove_punctuation=True)
        tokens = extractor._preprocess_text("Hello, World!")
        assert "Hello" in tokens or "hello" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_preprocess_text_remove_stopwords(self):
        """Test stopword removal."""
        extractor = TextFeatureExtractor(remove_stopwords=True)
        tokens = extractor._preprocess_text("This is a test")
        assert "is" not in tokens
        assert "a" not in tokens
        assert "test" in tokens

    def test_create_ngrams(self):
        """Test n-gram creation."""
        extractor = TextFeatureExtractor()
        tokens = ["hello", "world", "test"]
        
        unigrams = extractor._create_ngrams(tokens, 1)
        assert unigrams == tokens
        
        bigrams = extractor._create_ngrams(tokens, 2)
        assert len(bigrams) == 2
        assert "hello world" in bigrams
        assert "world test" in bigrams

    def test_fit_bag_of_words(self):
        """Test bag-of-words fitting."""
        documents = [
            "This is the first document",
            "This document is the second document",
            "And this is the third one",
        ]
        extractor = TextFeatureExtractor()
        extractor.fit_bag_of_words(documents)

        assert extractor.vocabulary_ is not None
        assert len(extractor.vocabulary_) > 0

    def test_transform_bag_of_words(self):
        """Test bag-of-words transformation."""
        documents = [
            "This is the first document",
            "This document is the second document",
        ]
        extractor = TextFeatureExtractor()
        extractor.fit_bag_of_words(documents)
        matrix = extractor.transform_bag_of_words(documents)

        assert matrix.shape[0] == len(documents)
        assert matrix.shape[1] == len(extractor.vocabulary_)
        assert np.all(matrix >= 0)

    def test_fit_transform_bag_of_words(self):
        """Test fit and transform bag-of-words."""
        documents = [
            "This is the first document",
            "This document is the second document",
        ]
        extractor = TextFeatureExtractor()
        matrix = extractor.fit_transform_bag_of_words(documents)

        assert matrix.shape[0] == len(documents)
        assert matrix.shape[1] > 0

    def test_fit_tfidf(self):
        """Test TF-IDF fitting."""
        documents = [
            "This is the first document",
            "This document is the second document",
            "And this is the third one",
        ]
        extractor = TextFeatureExtractor()
        extractor.fit_tfidf(documents)

        assert extractor.vocabulary_ is not None
        assert extractor.idf_ is not None
        assert len(extractor.idf_) == len(extractor.vocabulary_)

    def test_transform_tfidf(self):
        """Test TF-IDF transformation."""
        documents = [
            "This is the first document",
            "This document is the second document",
        ]
        extractor = TextFeatureExtractor()
        extractor.fit_tfidf(documents)
        matrix = extractor.transform_tfidf(documents)

        assert matrix.shape[0] == len(documents)
        assert matrix.shape[1] == len(extractor.vocabulary_)
        assert np.all(matrix >= 0)

    def test_fit_transform_tfidf(self):
        """Test fit and transform TF-IDF."""
        documents = [
            "This is the first document",
            "This document is the second document",
        ]
        extractor = TextFeatureExtractor()
        matrix = extractor.fit_transform_tfidf(documents)

        assert matrix.shape[0] == len(documents)
        assert matrix.shape[1] > 0

    def test_ngram_range(self):
        """Test n-gram range."""
        documents = [
            "This is a test",
            "This is another test",
        ]
        extractor = TextFeatureExtractor()
        matrix = extractor.fit_transform_tfidf(documents, ngram_range=(1, 2))

        assert matrix.shape[1] > 0

    def test_max_features(self):
        """Test max features limit."""
        documents = [
            "This is the first document",
            "This document is the second document",
            "And this is the third one",
        ]
        extractor = TextFeatureExtractor(max_features=5)
        extractor.fit_bag_of_words(documents)

        assert len(extractor.vocabulary_) <= 5

    def test_min_df(self):
        """Test minimum document frequency."""
        documents = [
            "This is the first document",
            "This document is the second document",
            "And this is the third one",
        ]
        extractor = TextFeatureExtractor(min_df=2)
        extractor.fit_bag_of_words(documents)

        assert len(extractor.vocabulary_) > 0

    def test_get_feature_names(self):
        """Test getting feature names."""
        documents = [
            "This is the first document",
            "This document is the second document",
        ]
        extractor = TextFeatureExtractor()
        extractor.fit_bag_of_words(documents)

        feature_names = extractor.get_feature_names()
        assert len(feature_names) == len(extractor.vocabulary_)

    def test_get_feature_importance(self):
        """Test feature importance."""
        documents = [
            "This is the first document",
            "This document is the second document",
        ]
        extractor = TextFeatureExtractor()
        extractor.fit_tfidf(documents)

        importance = extractor.get_feature_importance(documents, method="tfidf")
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_transform_before_fit_error(self):
        """Test error when transforming before fitting."""
        extractor = TextFeatureExtractor()
        documents = ["test document"]

        with pytest.raises(ValueError, match="must be fitted"):
            extractor.transform_bag_of_words(documents)

        with pytest.raises(ValueError, match="must be fitted"):
            extractor.transform_tfidf(documents)
