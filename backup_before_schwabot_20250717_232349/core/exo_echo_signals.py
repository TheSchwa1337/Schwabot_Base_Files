"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXO Echo Signals - Global Sentiment-Aware Memory-Trading Mesh
============================================================

Acts as a signal ingestion layer from:
- Twitter API
- Google News API
- Reddit + Webhooks
- Social sentiment sources

Each signal gets hashed, ranked, and time-stamped, then injected into
Lantern Core for ghost memory pattern matching and trade execution.

System Flow:
Twitter/Web/News/Reddit → exo_echo_signals.py → hash_string + classify_intent()
→ lantern_core.process_external_echo() → strategy_mapper.queue_trade()
"""

import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import requests
from urllib.parse import urlparse

# Import core dependencies with lazy loading
try:
from core.lantern_core import LanternCore
from core.hash_config_manager import generate_hash_from_string
LANTERN_AVAILABLE = True
except ImportError as e:
logging.warning(f"Core dependencies not available: {e}")
LANTERN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Signal source types
class SignalSource(Enum):
"""Class for Schwabot trading functionality."""
"""External signal sources."""
TWITTER = "twitter"
GOOGLE_NEWS = "google_news"
REDDIT = "reddit"
HACKER_NEWS = "hacker_news"
COINBASE_SENTIMENT = "coinbase_sentiment"
GLASSNODE = "glassnode"
WEBHOOK = "webhook"
MANUAL = "manual"

# Signal intent classification
class SignalIntent(Enum):
"""Class for Schwabot trading functionality."""
"""Signal intent classification."""
PANIC = "panic"
PUMP = "pump"
GHOST_RETURN = "ghost_return"
MASS_FEAR = "mass_fear"
FOMO = "fomo"
DUMP = "dump"
NEUTRAL = "neutral"
BULLISH = "bullish"
BEARISH = "bearish"

# Asset symbols for mapping
CRYPTO_SYMBOLS = {
'BTC': ['bitcoin', 'btc', 'bitcoin', 'satoshi'],
'ETH': ['ethereum', 'eth', 'ether'],
'XRP': ['ripple', 'xrp', 'xrp'],
'ADA': ['cardano', 'ada'],
'DOT': ['polkadot', 'dot'],
'LINK': ['chainlink', 'link'],
'SOL': ['solana', 'sol'],
'USDC': ['usdc', 'usd coin'],
'USDT': ['tether', 'usdt'],
'BNB': ['binance coin', 'bnb'],
'MATIC': ['polygon', 'matic'],
'AVAX': ['avalanche', 'avax'],
'UNI': ['uniswap', 'uni'],
'ATOM': ['cosmos', 'atom'],
'LTC': ['litecoin', 'ltc']
}

# Ghost pattern keywords for soulprint matching
GHOST_PATTERNS = {
'ghost_return': [
'return', 'back', 'again', 'reappear', 'resurface', 'comeback',
'bounce back', 'recovery', 'rebound', 'resurrection'
],
'mass_fear': [
'panic', 'fear', 'dump', 'crash', 'sell off', 'mass selling',
'capitulation', 'bloodbath', 'red', 'bear market'
],
'fomo': [
'moon', 'pump', 'bull run', 'fomo', 'buying', 'accumulation',
'green', 'bull market', 'rally', 'surge'
]
}


@dataclass
class EchoSignal:
"""Class for Schwabot trading functionality."""
"""External echo signal data structure."""
source: SignalSource
content: str
symbol: Optional[str]
intent: SignalIntent
priority: float
timestamp: datetime
hash_value: str
metadata: Dict[str, Any] = field(default_factory=dict)
processed: bool = False


@dataclass
class SignalProcessor:
"""Class for Schwabot trading functionality."""
"""Signal processing configuration."""
min_priority: float = 0.3
max_signals_per_minute: int = 100
enable_twitter: bool = True
enable_news: bool = True
enable_reddit: bool = True
enable_webhooks: bool = True
ghost_pattern_threshold: float = 0.7


class EXOEchoSignals:
"""Class for Schwabot trading functionality."""
"""
EXO Echo Signals - Global sentiment-aware signal ingestion and processing.

Processes external signals from various sources and converts them into
hashed, ranked, and time-stamped signals for Lantern Core integration.
"""

def __init__(
self,
config: Optional[Dict[str, Any]] = None,
lantern_core: Optional[LanternCore] = None
):
"""
Initialize EXO Echo Signals processor.

Args:
config: Configuration dictionary
lantern_core: Lantern Core instance for signal processing
"""
self.config = config or self._default_config()
self.lantern_core = lantern_core

# Signal processing state
self.processor = SignalProcessor(**self.config.get('processor', {}))
self.signal_queue: List[EchoSignal] = []
self.processed_signals: List[EchoSignal] = []
self.signal_history: Dict[str, List[EchoSignal]] = {}

# API configurations
self.twitter_config = self.config.get('twitter', {})
self.news_config = self.config.get('news', {})
self.reddit_config = self.config.get('reddit', {})

# Rate limiting
self.last_signal_time = datetime.utcnow()
self.signals_this_minute = 0

# Integration metrics
self.metrics = {
'total_signals_processed': 0,
'signals_by_source': {},
'signals_by_intent': {},
'high_priority_signals': 0,
'ghost_pattern_matches': 0,
'lantern_integrations': 0
}

logger.info("EXO Echo Signals initialized")
print(f"[EXO] Echo Signals initialized with {len(CRYPTO_SYMBOLS)} crypto symbols")

def _default_config(self) -> Dict[str, Any]:
"""Default configuration."""
return {
'processor': {
'min_priority': 0.3,
'max_signals_per_minute': 100,
'enable_twitter': True,
'enable_news': True,
'enable_reddit': True,
'enable_webhooks': True,
'ghost_pattern_threshold': 0.7
},
'twitter': {
'enabled': False,
'bearer_token': '',
'search_queries': ['btc', 'ethereum', 'crypto', 'bitcoin'],
'max_results': 10
},
'news': {
'enabled': False,
'api_key': '',
'sources': ['google', 'newsapi'],
'keywords': ['bitcoin', 'crypto', 'blockchain']
},
'reddit': {
'enabled': False,
'client_id': '',
'client_secret': '',
'subreddits': ['cryptocurrency', 'bitcoin', 'ethereum']
},
'webhooks': {
'enabled': True,
'endpoint': '/api/echo',
'secret_key': ''
}
}

def process_external_signal(
self,
content: str,
source: SignalSource,
metadata: Optional[Dict[str, Any]] = None
) -> Optional[EchoSignal]:
"""
Process an external signal and convert it to an EchoSignal.

Args:
content: Signal content (tweet, news article, reddit post, etc.)
source: Source of the signal
metadata: Additional metadata

Returns:
Processed EchoSignal or None if below threshold
"""
try:
# Rate limiting check
if not self._check_rate_limit():
logger.warning("Rate limit exceeded, skipping signal")
return None

# Classify signal intent
intent = self._classify_intent(content)

# Extract symbol
symbol = self._extract_symbol(content)

# Calculate priority
priority = self._calculate_priority(content, intent, source, metadata)

# Check if signal meets minimum priority threshold
if priority < self.processor.min_priority:
logger.debug(f"Signal below threshold: {priority:.3f}")
return None

# Generate hash
hash_value = self._generate_signal_hash(content, source, intent, symbol)

# Create echo signal
echo_signal = EchoSignal(
source=source,
content=content,
symbol=symbol,
intent=intent,
priority=priority,
timestamp=datetime.utcnow(),
hash_value=hash_value,
metadata=metadata or {}
)

# Add to queue
self.signal_queue.append(echo_signal)

# Update metrics
self._update_metrics(echo_signal)

# Process with Lantern Core if available
if self.lantern_core and LANTERN_AVAILABLE:
self._process_with_lantern(echo_signal)

logger.info(f"Processed {source.value} signal: {intent.value} for {symbol} (priority: {priority:.3f})")
print(f"[EXO] {source.value.upper()} → {intent.value} → {symbol} (priority: {priority:.3f})")

return echo_signal

except Exception as e:
logger.error(f"Error processing external signal: {e}")
return None

def _classify_intent(self, content: str) -> SignalIntent:
"""
Classify the intent of a signal based on content analysis.

Args:
content: Signal content

Returns:
Classified intent
"""
content_lower = content.lower()

# Count keyword matches for each intent
intent_scores = {}

for intent, keywords in GHOST_PATTERNS.items():
score = sum(1 for keyword in keywords if keyword in content_lower)
intent_scores[intent] = score

# Additional sentiment analysis
panic_words = ['crash', 'dump', 'sell', 'panic', 'fear', 'bear']
pump_words = ['moon', 'pump', 'buy', 'bull', 'rally', 'surge']

panic_score = sum(1 for word in panic_words if word in content_lower)
pump_score = sum(1 for word in pump_words if word in content_lower)

# Determine primary intent
if intent_scores.get('mass_fear', 0) > 2 or panic_score > 2:
return SignalIntent.MASS_FEAR
elif intent_scores.get('fomo', 0) > 2 or pump_score > 2:
return SignalIntent.FOMO
elif intent_scores.get('ghost_return', 0) > 1:
return SignalIntent.GHOST_RETURN
elif panic_score > pump_score:
return SignalIntent.PANIC
elif pump_score > panic_score:
return SignalIntent.PUMP
else:
return SignalIntent.NEUTRAL

def _extract_symbol(self, content: str) -> Optional[str]:
"""
Extract cryptocurrency symbol from content.

Args:
content: Signal content

Returns:
Extracted symbol or None
"""
content_lower = content.lower()

for symbol, keywords in CRYPTO_SYMBOLS.items():
for keyword in keywords:
if keyword in content_lower:
return symbol

return None

def _calculate_priority(
self,
content: str,
intent: SignalIntent,
source: SignalSource,
metadata: Optional[Dict[str, Any]]
) -> float:
"""
Calculate signal priority based on various factors.

Args:
content: Signal content
intent: Signal intent
source: Signal source
metadata: Additional metadata

Returns:
Priority score between 0 and 1
"""
priority = 0.5  # Base priority

# Intent-based adjustments
intent_weights = {
SignalIntent.MASS_FEAR: 0.9,
SignalIntent.GHOST_RETURN: 0.85,
SignalIntent.FOMO: 0.8,
SignalIntent.PANIC: 0.75,
SignalIntent.PUMP: 0.7,
SignalIntent.DUMP: 0.65,
SignalIntent.BULLISH: 0.6,
SignalIntent.BEARISH: 0.6,
SignalIntent.NEUTRAL: 0.3
}

# Apply intent weight
intent_weight = intent_weights.get(intent, 0.5)
priority = priority * intent_weight

# Source-based adjustments
source_weights = {
SignalSource.TWITTER: 0.8,
SignalSource.GOOGLE_NEWS: 0.9,
SignalSource.REDDIT: 0.7,
SignalSource.HACKER_NEWS: 0.6,
SignalSource.COINBASE_SENTIMENT: 0.95,
SignalSource.GLASSNODE: 0.9,
SignalSource.WEBHOOK: 0.8,
SignalSource.MANUAL: 1.0
}

# Apply source weight
source_weight = source_weights.get(source, 0.5)
priority = priority * source_weight

# Content length adjustment (longer content = more detailed)
content_length = len(content)
if content_length > 200:
priority *= 1.2
elif content_length < 50:
priority *= 0.8

# Ghost pattern detection
if self._detect_ghost_patterns(content):
priority *= 1.3

# Metadata adjustments
if metadata:
# Engagement metrics (if available)
if 'engagement' in metadata:
engagement = metadata['engagement']
if engagement > 1000:
priority *= 1.2
elif engagement > 100:
priority *= 1.1

# Sentiment score (if available)
if 'sentiment_score' in metadata:
sentiment = metadata['sentiment_score']
if abs(sentiment) > 0.7:
priority *= 1.1

return min(priority, 1.0)

def _detect_ghost_patterns(self, content: str) -> bool:
"""
Detect ghost patterns in content that match soulprint memory.

Args:
content: Signal content

Returns:
True if ghost patterns detected
"""
content_lower = content.lower()

for pattern_name, keywords in GHOST_PATTERNS.items():
matches = sum(1 for keyword in keywords if keyword in content_lower)
if matches >= 2:  # At least 2 keyword matches
return True

return False

def _generate_signal_hash(
self,
content: str,
source: SignalSource,
intent: SignalIntent,
symbol: Optional[str]
) -> str:
"""
Generate unique hash for signal.

Args:
content: Signal content
source: Signal source
intent: Signal intent
symbol: Cryptocurrency symbol

Returns:
Unique hash string
"""
hash_input = f"{content[:100]}:{source.value}:{intent.value}:{symbol or 'unknown'}:{datetime.utcnow().isoformat()}"

if hasattr(self, 'hash_registry') and self.hash_registry:
return self.hash_registry.hash_string(hash_input)
else:
return hashlib.sha256(hash_input.encode()).hexdigest()

def _check_rate_limit(self) -> bool:
"""
Check if we're within rate limits.

Returns:
True if within limits, False otherwise
"""
current_time = datetime.utcnow()

# Reset counter if a minute has passed
if (current_time - self.last_signal_time).total_seconds() > 60:
self.signals_this_minute = 0
self.last_signal_time = current_time

# Check if we're over the limit
if self.signals_this_minute >= self.processor.max_signals_per_minute:
return False

self.signals_this_minute += 1
return True

def _update_metrics(self, signal: EchoSignal) -> None:
"""Update processing metrics."""
self.metrics['total_signals_processed'] += 1

# Update source metrics
source = signal.source.value
self.metrics['signals_by_source'][source] = self.metrics['signals_by_source'].get(source, 0) + 1

# Update intent metrics
intent = signal.intent.value
self.metrics['signals_by_intent'][intent] = self.metrics['signals_by_intent'].get(intent, 0) + 1

# Update high priority signals
if signal.priority > 0.8:
self.metrics['high_priority_signals'] += 1

# Update ghost pattern matches
if self._detect_ghost_patterns(signal.content):
self.metrics['ghost_pattern_matches'] += 1

def _process_with_lantern(self, signal: EchoSignal) -> None:
"""
Process signal with Lantern Core for ghost memory matching.

Args:
signal: Echo signal to process
"""
try:
if not self.lantern_core:
return

# Create external echo data for Lantern Core
external_echo = {
'source': signal.source.value,
'signal': signal.intent.value,
'mapped_asset': signal.symbol,
'soulprint_hint': self._get_soulprint_hint(signal),
'priority': signal.priority,
'timestamp': signal.timestamp.isoformat(),
'hash_value': signal.hash_value,
'content': signal.content[:200]  # Truncate for storage
}

# Process with Lantern Core
result = self.lantern_core.process_external_echo(external_echo)

if result and result.get('should_trigger', False):
logger.info(f"🎯 Lantern Core triggered ghost reentry: {signal.symbol} ({signal.intent.value})")
print(f"[EXO] Lantern Core triggered ghost reentry for {signal.symbol} (score: {result.get('trigger_score', 0):.3f})")

# Mark as processed
signal.processed = True
self.metrics['lantern_integrations'] += 1

# Add to processed signals
self.processed_signals.append(signal)

elif result:
logger.debug(f"External echo processed but not triggered: {signal.symbol}")
signal.processed = True
self.processed_signals.append(signal)

except Exception as e:
logger.error(f"Error processing with Lantern Core: {e}")

def _get_soulprint_hint(self, signal: EchoSignal) -> str:
"""
Get soulprint hint based on signal content and intent.

Args:
signal: Echo signal

Returns:
Soulprint hint string
"""
if signal.intent == SignalIntent.GHOST_RETURN:
return "ghost_return"
elif signal.intent == SignalIntent.MASS_FEAR:
return "mass_fear"
elif signal.intent == SignalIntent.FOMO:
return "fomo_pump"
else:
return "general_sentiment"

async def process_twitter_signals(self, query: str = "btc OR ethereum OR crypto") -> List[EchoSignal]:
"""
Process Twitter signals (placeholder for API integration).

Args:
query: Twitter search query

Returns:
List of processed echo signals
"""
if not self.twitter_config.get('enabled', False):
logger.info("Twitter processing disabled")
return []

# Placeholder for Twitter API integration
# In a real implementation, this would use tweepy or direct API calls
logger.info(f"Processing Twitter signals for query: {query}")

# Simulate some Twitter signals
mock_tweets = [
"Bitcoin just dropped 15%! Mass panic selling happening! #BTC #crypto",
"Ethereum looking bullish again, time to buy the dip! #ETH #moon",
"Crypto market is dead, everyone is selling their bags #dump #panic"
]

signals = []
for tweet in mock_tweets:
signal = self.process_external_signal(
content=tweet,
source=SignalSource.TWITTER,
metadata={'engagement': 500}
)
if signal:
signals.append(signal)

return signals

async def process_news_signals(self, keywords: List[str] = None) -> List[EchoSignal]:
"""
Process news signals (placeholder for API integration).

Args:
keywords: News search keywords

Returns:
List of processed echo signals
"""
if not self.news_config.get('enabled', False):
logger.info("News processing disabled")
return []

keywords = keywords or ['bitcoin', 'crypto', 'blockchain']
logger.info(f"Processing news signals for keywords: {keywords}")

# Placeholder for news API integration
mock_articles = [
"Bitcoin crashes 20% as market fears intensify",
"Ethereum shows signs of recovery after recent dip",
"Crypto market sees massive sell-off across all major coins"
]

signals = []
for article in mock_articles:
signal = self.process_external_signal(
content=article,
source=SignalSource.GOOGLE_NEWS,
metadata={'sentiment_score': -0.8}
)
if signal:
signals.append(signal)

return signals

async def process_reddit_signals(self, subreddits: List[str] = None) -> List[EchoSignal]:
"""
Process Reddit signals (placeholder for API integration).

Args:
subreddits: Subreddits to monitor

Returns:
List of processed echo signals
"""
if not self.reddit_config.get('enabled', False):
logger.info("Reddit processing disabled")
return []

subreddits = subreddits or ['cryptocurrency', 'bitcoin', 'ethereum']
logger.info(f"Processing Reddit signals for subreddits: {subreddits}")

# Placeholder for Reddit API integration
mock_posts = [
"Just sold all my BTC at a loss, this market is dead",
"ETH looking ready for a massive pump!",
"Anyone else feeling the FOMO right now?"
]

signals = []
for post in mock_posts:
signal = self.process_external_signal(
content=post,
source=SignalSource.REDDIT,
metadata={'upvotes': 100}
)
if signal:
signals.append(signal)

return signals

def get_metrics(self) -> Dict[str, Any]:
"""
Get processing metrics.

Returns:
Dictionary of metrics
"""
return {
**self.metrics,
'queue_size': len(self.signal_queue),
'processed_count': len(self.processed_signals),
'rate_limit_status': {
'signals_this_minute': self.signals_this_minute,
'max_per_minute': self.processor.max_signals_per_minute
}
}

def export_signals(self) -> Dict[str, Any]:
"""
Export processed signals for persistence.

Returns:
Dictionary containing signal data
"""
return {
'signals': [
{
'source': signal.source.value,
'content': signal.content,
'symbol': signal.symbol,
'intent': signal.intent.value,
'priority': signal.priority,
'timestamp': signal.timestamp.isoformat(),
'hash_value': signal.hash_value,
'metadata': signal.metadata,
'processed': signal.processed
}
for signal in self.processed_signals
],
'metrics': self.metrics,
'export_timestamp': datetime.utcnow().isoformat()
}

def import_signals(self, data: Dict[str, Any]) -> None:
"""
Import signals from persistence.

Args:
data: Dictionary containing signal data
"""
try:
if 'signals' in data:
for signal_data in data['signals']:
try:
signal = EchoSignal(
source=SignalSource(signal_data['source']),
content=signal_data['content'],
symbol=signal_data['symbol'],
intent=SignalIntent(signal_data['intent']),
priority=signal_data['priority'],
timestamp=datetime.fromisoformat(signal_data['timestamp']),
hash_value=signal_data['hash_value'],
metadata=signal_data.get('metadata', {}),
processed=signal_data.get('processed', False)
)
self.processed_signals.append(signal)
except Exception as e:
logger.warning(f"Failed to import signal: {e}")
continue

if 'metrics' in data:
# Update metrics but preserve current counts
for key, value in data['metrics'].items():
if key in self.metrics:
if isinstance(value, dict):
self.metrics[key].update(value)
else:
self.metrics[key] = value

logger.info(f"Imported {len(data.get('signals', []))} signals")

except Exception as e:
logger.error(f"Error importing signals: {e}")


# Global instance for easy access
exo_echo_signals = EXOEchoSignals()

# =========================
# Bridge & Backfill Section
# =========================

def process_external_echo_signal(
content: str,
source: str,
metadata: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
"""
Process external echo signal (bridge function).

Args:
content: Signal content
source: Signal source
metadata: Additional metadata

Returns:
Processed signal data or None
"""
try:
# Create a new instance if global is not available
if not hasattr(exo_echo_signals, 'process_external_signal'):
local_exo = EXOEchoSignals()
else:
local_exo = exo_echo_signals

signal_source = SignalSource(source)
signal = local_exo.process_external_signal(content, signal_source, metadata)

if signal:
return {
'source': signal.source.value,
'symbol': signal.symbol,
'intent': signal.intent.value,
'priority': signal.priority,
'hash_value': signal.hash_value,
'timestamp': signal.timestamp.isoformat(),
'processed': signal.processed
}

return None

except Exception as e:
logger.error(f"Error processing external echo signal: {e}")
return None


def classify_signal_intent(content: str) -> str:
"""
Classify signal intent (bridge function).

Args:
content: Signal content

Returns:
Intent classification
"""
try:
# Create a new instance if global is not available
if not hasattr(exo_echo_signals, '_classify_intent'):
local_exo = EXOEchoSignals()
else:
local_exo = exo_echo_signals

intent = local_exo._classify_intent(content)
return intent.value
except Exception as e:
logger.error(f"Error classifying signal intent: {e}")
return "neutral"


def extract_crypto_symbol(content: str) -> Optional[str]:
"""
Extract cryptocurrency symbol (bridge function).

Args:
content: Signal content

Returns:
Extracted symbol or None
"""
try:
# Create a new instance if global is not available
if not hasattr(exo_echo_signals, '_extract_symbol'):
local_exo = EXOEchoSignals()
else:
local_exo = exo_echo_signals

return local_exo._extract_symbol(content)
except Exception as e:
logger.error(f"Error extracting crypto symbol: {e}")
return None


# Configuration constants
EXO_CONFIG_CONSTANTS = {
'DEFAULT_MIN_PRIORITY': 0.3,
'DEFAULT_MAX_SIGNALS_PER_MINUTE': 100,
'GHOST_PATTERN_THRESHOLD': 0.7,
'SUPPORTED_SOURCES': [source.value for source in SignalSource],
'SUPPORTED_INTENTS': [intent.value for intent in SignalIntent],
'CRYPTO_SYMBOLS': list(CRYPTO_SYMBOLS.keys())
}