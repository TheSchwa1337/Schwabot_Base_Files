"""
Tests for WordFitnessTracker
===========================

Verifies the functionality of word fitness tracking and profit score evolution.
"""

import pytest
import yaml
import time
from core.lantern.word_fitness_tracker import WordFitnessTracker


@pytest.fixture
def temp_yaml_path(tmp_path):
    """Create a temporary YAML file for testing"""
    yaml_path = tmp_path / "test_words.yaml"  # noqa: F841
    return str(yaml_path)


@pytest.fixture
def sample_words():
    """Sample word data for testing"""
    return [
        {
            "word": "bullish",
            "pos": "adj",
            "sentiment": "positive",
            "vector_bias": "up",
            "entropy": "low",
            "profit_score": 0.0,
            "usage_count": 0
        },
        {
            "word": "bearish",
            "pos": "adj",
            "sentiment": "negative",
            "vector_bias": "down",
            "entropy": "low",
            "profit_score": 0.0,
            "usage_count": 0
        },
        {
            "word": "volatile",
            "pos": "adj",
            "sentiment": "neutral",
            "vector_bias": "sideways",
            "entropy": "high",
            "profit_score": 0.0,
            "usage_count": 0
        }
    ]


@pytest.fixture
def tracker(temp_yaml_path, sample_words):
    """Create a WordFitnessTracker instance with sample data"""
    # Write sample data to YAML
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(sample_words, f)

    return WordFitnessTracker(yaml_path=temp_yaml_path)


def test_initialization(tracker, sample_words):
    """Test tracker initialization and data loading"""
    assert len(tracker.words) == len(sample_words)
    for word in tracker.words:
        assert "word" in word
        assert "profit_score" in word
        assert "usage_count" in word


def test_update_profit_scores(tracker):
    """Test profit score updates"""
    # Update scores for "bullish" and "bearish"
    used_words = ["bullish", "bearish"]
    delta_profit = 0.5

    tracker.update_profit_scores(used_words, delta_profit)

    # Check updates
    for word in tracker.words:
        if word["word"] in used_words:
            assert word["profit_score"] == delta_profit
            assert word["usage_count"] == 1
        else:
            assert word["profit_score"] == 0.0
            assert word["usage_count"] == 0


def test_decay_factor(tracker):
    """Test time-based decay of profit scores"""
    # Set initial scores
    for word in tracker.words:
        word["profit_score"] = 1.0

    # Update one word
    tracker.update_profit_scores(["bullish"], 0.5)

    # Wait for decay
    time.sleep(1)

    # Update again
    tracker.update_profit_scores(["bearish"], 0.5)

    # Check decay
    for word in tracker.words:
        if word["word"] == "bullish":
            assert word["profit_score"] > 0.5  # Decayed but still positive
        elif word["word"] == "bearish":
            assert word["profit_score"] == 0.5  # Fresh update
        else:
            assert word["profit_score"] < 1.0  # Decayed


def test_get_top_words(tracker):
    """Test getting top words by profit score"""
    # Set different scores
    for word in tracker.words:
        if word["word"] == "bullish":
            word["profit_score"] = 0.8
        elif word["word"] == "bearish":
            word["profit_score"] = 0.5
        else:
            word["profit_score"] = 0.3

    top_words = tracker.get_top_words(top_n=2)
    assert len(top_words) == 2
    assert top_words[0]["word"] == "bullish"
    assert top_words[1]["word"] == "bearish"


def test_get_word_stats(tracker):
    """Test getting statistics for a specific word"""
    stats = tracker.get_word_stats("bullish")
    assert stats is not None
    assert stats["profit_score"] == 0.0
    assert stats["usage_count"] == 0
    assert stats["pos"] == "adj"
    assert stats["sentiment"] == "positive"

    # Test non-existent word
    assert tracker.get_word_stats("nonexistent") is None


def test_get_lexicon_stats(tracker):
    """Test getting overall lexicon statistics"""
    stats = tracker.get_lexicon_stats()

    assert stats["total_words"] == 3
    assert stats["avg_profit_score"] == 0.0
    assert stats["total_usage"] == 0

    # Check distributions
    assert "adj" in stats["pos_distribution"]
    assert "positive" in stats["sentiment_distribution"]
    assert "up" in stats["vector_bias_distribution"]


def test_add_remove_word(tracker):
    """Test adding and removing words"""
    # Add new word
    tracker.add_new_word(
        word="trending",
        pos="verb",
        sentiment="positive",
        vector_bias="up",
        entropy="medium"
    )

    assert len(tracker.words) == 4
    assert any(w["word"] == "trending" for w in tracker.words)

    # Remove word
    assert tracker.remove_word("trending")
    assert len(tracker.words) == 3
    assert not any(w["word"] == "trending" for w in tracker.words)


def test_reset_scores(tracker):
    """Test resetting all scores"""
    # Set some scores
    for word in tracker.words:
        word["profit_score"] = 1.0
        word["usage_count"] = 5

    tracker.reset_all_scores()

    for word in tracker.words:
        assert word["profit_score"] == 0.0
        assert word["usage_count"] == 0