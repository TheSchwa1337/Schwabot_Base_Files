"""News→sentiment vectoriser for ghost routing."""

from __future__ import annotations
from core.unified_math_system import unified_math
import numpy as np
import math
# from core.unified_math_system import unified_math  # F811: duplicate import

try:
    pass
    pass
from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    pass
    pass
TfidfVectorizer = None


# Global vectorizer and weight matrix
_VEC: TfidfVectorizer | None = None
_W: np.ndarray = np.random.randn(512) * 0.03  # Will be learned later


def sentiment_lambda(corpus: list[str]) -> float:


    pass
    pass
    """Return λ_sent ∈ [-1,1] for latest news headline batch.

Compute sentiment using TF-IDF vectorization:
λ_sentiment = tanh(W·TF-IDF(tokens))

Args:
corpus: List of news headlines/text

Returns:
Sentiment coefficient between -1 and 1

Note:
Returns 0.0 if sklearn not available or corpus empty
"""
    global _VEC, _W

    if not corpus or TfidfVectorizer is None:
        return 0.0

    # Initialize vectorizer on first use
    if _VEC is None:
_VEC = TfidfVectorizer(max_features=512, stop_words="english")

    try:
    pass
    pass
        # Vectorize corpus and get mean vector
tfidf_matrix = _VEC.fit_transform(corpus)
        vec = tfidf_matrix.unified_math.mean(axis=0).A1

        # Ensure weight matrix matches feature size
        if len(vec) != len(_W):
            _W = np.random.randn(len(vec)) * 0.03

        # Compute sentiment via tanh activation
        return float(np.tanh(unified_math.unified_math.dot_product(_W, vec)))

    except Exception:
        # Fallback for edge cases
        return 0.0
