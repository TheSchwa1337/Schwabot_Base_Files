"""
Tests for LexiconEngine
=====================

Verifies the implementation of the formal mathematical framework
for the Lantern Profit Lexicon Engine.
"""

import pytest  # noqa: F401
import yaml  # noqa: F401
import time  # noqa: F401
from datetime import datetime  # noqa: F821
import numpy as np  # noqa: F401
from core.lantern.lexicon_engine import (  # noqa: F401
    LexiconEngine,
    WordState,
    PartOfSpeech,
    EntropyClass,
    VectorBias
)


@pytest.fixture
def temp_yaml_path(tmp_path):
    """Create a temporary YAML file for testing"""
    yaml_path = tmp_path / "test_lexicon.yaml"  # noqa: F841
    return str(yaml_path)


@pytest.fixture
def sample_words():
    """Sample word data for testing"""
    return [
        {
            "word": "bullish",
            "pos": "adj",
            "sentiment": 0.8,
            "entropy": "low",
            "vector_bias": "long",
            "profit_score": 0.0,
            "usage_count": 0,
            "volatility_score": 0.5
        },
        {
            "word": "bearish",
            "pos": "adj",
            "sentiment": -0.8,
            "entropy": "low",
            "vector_bias": "short",
            "profit_score": 0.0,
            "usage_count": 0,
            "volatility_score": 0.5
        },
        {
            "word": "volatile",
            "pos": "adj",
            "sentiment": 0.0,
            "entropy": "high",
            "vector_bias": "warning",
            "profit_score": 0.0,
            "usage_count": 0,
            "volatility_score": 0.8
        },
        {
            "word": "trend",
            "pos": "noun",
            "sentiment": 0.5,
            "entropy": "medium",
            "vector_bias": "long",
            "profit_score": 0.0,
            "usage_count": 0,
            "volatility_score": 0.3
        },
        {
            "word": "break",
            "pos": "verb",
            "sentiment": -0.3,
            "entropy": "medium",
            "vector_bias": "short",
            "profit_score": 0.0,
            "usage_count": 0,
            "volatility_score": 0.6
        }
    ]


@pytest.fixture
def engine(temp_yaml_path, sample_words):
    """Create a LexiconEngine instance with sample data"""
    # Write sample data to YAML
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(sample_words, f)

    return LexiconEngine(yaml_path=temp_yaml_path)


def test_initialization(engine, sample_words):
    """Test engine initialization and data loading"""
    assert len(engine.lexicon) == len(sample_words)
    for word, state in engine.lexicon.items():
        assert isinstance(state, WordState)
        assert state.word == word
        assert isinstance(state.pos, PartOfSpeech)
        assert isinstance(state.entropy, EntropyClass)
        assert isinstance(state.vector_bias, VectorBias)
        assert -1 <= state.sentiment <= 1
        assert state.profit_fitness >= 0
        assert state.usage_count >= 0
        assert 0 <= state.volatility_score <= 1


def test_fitness_update(engine):
    """Test fitness update dynamics"""
    # Update fitness for "bullish" and "bearish"
    used_words = ["bullish", "bearish"]
    profit_delta = 0.5

    engine.update_fitness(used_words, profit_delta)

    # Check updates
    for word, state in engine.lexicon.items():
        if word in used_words:
            assert state.profit_fitness == profit_delta * engine.beta
            assert state.usage_count == 1
        else:
            assert state.profit_fitness == 0.0
            assert state.usage_count == 0


def test_decay_dynamics(engine):
    """Test time-based decay of fitness scores"""
    # Set initial scores
    for state in engine.lexicon.values():
        state.profit_fitness = 1.0

    # Update one word
    engine.update_fitness(["bullish"], 0.5)

    # Wait for decay
    time.sleep(1)

    # Update again
    engine.update_fitness(["bearish"], 0.5)

    # Check decay
    for word, state in engine.lexicon.items():
        if word == "bullish":
            assert state.profit_fitness > 0.5 * engine.beta  # Decayed but still positive
        elif word == "bearish":
            assert state.profit_fitness == 0.5 * engine.beta  # Fresh update
        else:
            assert state.profit_fitness < 1.0  # Decayed


def test_price_hash_generation(engine):
    """Test price hash generation and mapping"""
    price = 50000.0
    timestamp = datetime.now()  # noqa: F821

    # Generate hash
    price_hash = engine.generate_price_hash(price, timestamp)
    assert len(price_hash) == 64  # SHA-256 produces 64 hex chars

    # Map to index
    seed_index = engine.hash_to_seed_index(price_hash)
    assert 0 <= seed_index < len(engine.lexicon)


def test_story_generation(engine):
    """Test story generation mechanics"""
    # Get top words
    top_words = engine.get_top_profit_words(k=2)
    assert len(top_words) == 2

    # Generate story
    story_length = 5
    story = engine.generate_story(0, top_words, story_length)

    assert len(story) == story_length
    assert all(word in engine.lexicon for word in story)

    # Check story history
    assert len(engine.story_history) == 1
    assert engine.story_history[0] == story


def test_narrative_entropy(engine):
    """Test narrative entropy calculation"""
    # Test with uniform distribution
    uniform_story = ["bullish", "bearish", "volatile", "trend", "break"]
    uniform_entropy = engine.calculate_narrative_entropy(uniform_story)
    assert uniform_entropy > 0

    # Test with repeated words
    repeated_story = ["bullish", "bullish", "bullish", "bullish", "bullish"]
    repeated_entropy = engine.calculate_narrative_entropy(repeated_story)
    assert repeated_entropy == 0  # Zero entropy for uniform distribution

    # Test empty story
    assert engine.calculate_narrative_entropy([]) == 0.0


def test_profitability_estimation(engine):
    """Test story profitability estimation"""
    # Generate test story
    story = ["bullish", "trend", "break", "volatile", "bearish"]

    # Calculate profitability
    profitability = engine.estimate_story_profitability(story)
    assert 0 <= profitability <= 1

    # Test empty story
    assert engine.estimate_story_profitability([]) == 0.0


def test_system_state(engine):
    """Test system state tracking"""
    # Update some fitness scores
    engine.update_fitness(["bullish", "bearish"], 0.5)

    # Get system state
    state = engine.get_system_state()

    assert 'fitness_vector' in state
    assert 'story_history' in state
    assert 'cumulative_profit' in state
    assert 'last_update' in state

    assert len(state['fitness_vector']) == len(engine.lexicon)
    assert state['cumulative_profit'] == 0.5


def test_lexicon_statistics(engine):
    """Test lexicon statistics calculation"""
    stats = engine.get_lexicon_stats()

    assert stats['total_words'] == len(engine.lexicon)
    assert stats['avg_fitness'] >= 0
    assert stats['total_usage'] >= 0

    # Check distributions
    assert 'adj' in stats['pos_distribution']
    assert 'low' in stats['entropy_distribution']
    assert 'long' in stats['vector_bias_distribution']

    # Verify distribution percentages
    for dist in [stats['pos_distribution'],
                 stats['entropy_distribution'],
                 stats['vector_bias_distribution']]:
        assert abs(sum(dist.values()) - 1.0) < 1e-10  # Should sum to 1


def test_mathematical_invariants(engine):
    """Test mathematical invariants"""
    # Test probability conservation
    story = engine.generate_story(0, engine.get_top_profit_words(), 10)
    word_counts = {}
    for word in story:
        word_counts[word] = word_counts.get(word, 0) + 1
    probs = [count / len(story) for count in word_counts.values()]
    assert abs(sum(probs) - 1.0) < 1e-10

    # Test fitness positivity
    engine.update_fitness(["bullish"], -0.5)  # Try negative update
    assert engine.lexicon["bullish"].profit_fitness >= 0

    # Test entropy bounds
    story = ["bullish"] * 10  # Maximum entropy
    entropy = engine.calculate_narrative_entropy(story)
    assert 0 <= entropy <= np.log(len(story))

    # Test action completeness
    vector_biases = {state.vector_bias for state in engine.lexicon.values()}
    assert set(VectorBias) == vector_biases