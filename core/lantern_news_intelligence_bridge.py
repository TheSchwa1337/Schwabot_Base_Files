"""
Lantern News Intelligence Bridge
==============================
Integrates news intelligence with Lantern Core for mathematical correlation
between news events, BTC hash structures, and profit-optimized lexicon analysis.

Key Features:
- News lexicon analysis through Lantern's word fitness system
- BTC hash correlation with news timing and sentiment
- Syntax structure analysis for hash pattern recognition
- Profit allocation mapping based on news-driven word evolution
- Backlog management to prevent over-saturation
- Integration with BTC processor controls
"""

import asyncio
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import logging

from .news_intelligence_engine import NewsIntelligenceEngine, NewsItem, MarketContext
from .lantern.lexicon_engine import LexiconEngine, WordState, VectorBias, EntropyClass
from .lantern.profit_story_engine import ProfitStoryEngine
from .lantern.word_fitness_tracker import WordFitnessTracker
from .btc_processor_controller import BTCProcessorController
from .hash_recollection import HashRecollectionSystem
from .memory_agent import MemoryAgent
from .sustainment_integration_hooks import SustainmentHooks

logger = logging.getLogger(__name__)

@dataclass
class NewsHashCorrelation:
    """Correlation between news event and BTC hash structure"""
    news_hash: str
    btc_hash: str
    timestamp: datetime
    correlation_score: float
    sentiment_score: float
    lexicon_words: List[str]
    hash_similarity: float
    profit_potential: float

@dataclass
class NewsLexiconEvent:
    """News event processed through Lantern lexicon analysis"""
    news_item: NewsItem
    lexicon_words: List[str]
    profit_story: List[str]
    hash_correlation: NewsHashCorrelation
    vector_bias: VectorBias
    entropy_class: EntropyClass
    processing_priority: float


class LanternNewsIntelligenceBridge:
    """Bridge between news intelligence and Lantern Core mathematical framework"""
    
    def __init__(self, 
                 news_engine: Optional[NewsIntelligenceEngine] = None,
                 btc_controller: Optional[BTCProcessorController] = None,
                 config_path: str = "config/lantern_news_config.yaml"):
        
        # Core components
        self.news_engine = news_engine or NewsIntelligenceEngine()
        self.btc_controller = btc_controller or BTCProcessorController()
        self.lexicon_engine = LexiconEngine()
        self.profit_story_engine = ProfitStoryEngine()
        self.word_fitness_tracker = WordFitnessTracker()
        self.hash_system = HashRecollectionSystem()
        self.memory_agent = MemoryAgent()
        self.hooks = SustainmentHooks()
        
        # Processing state
        self.news_backlog: List[NewsLexiconEvent] = []
        self.correlation_cache: Dict[str, NewsHashCorrelation] = {}
        self.processing_active = True
        
        # Configuration
        self.config = {
            "max_backlog_size": 500,
            "correlation_threshold": 0.3,
            "processing_batch_size": 10,
            "lexicon_update_interval": 300,  # 5 minutes
            "hash_correlation_window": 3600,  # 1 hour
            "profit_weight_factor": 0.7,
            "entropy_threshold": 0.6,
            "auto_throttle_enabled": True
        }
        
        # Performance tracking
        self.metrics = {
            "news_processed": 0,
            "correlations_found": 0,
            "profit_updates": 0,
            "backlog_overflows": 0,
            "last_update": datetime.now()
        }
    
    async def initialize(self):
        """Initialize all components"""
        await self.news_engine.initialize()
        logger.info("Lantern News Intelligence Bridge initialized")
    
    async def process_news_through_lantern(self, news_items: List[NewsItem]) -> List[NewsLexiconEvent]:
        """Process news items through Lantern's lexicon and hash correlation system"""
        
        # Check backlog capacity
        if len(self.news_backlog) > self.config["max_backlog_size"]:
            await self._manage_backlog_overflow()
        
        processed_events = []
        
        for news_item in news_items:
            try:
                # Extract lexicon words from news content
                lexicon_words = await self._extract_lexicon_words(news_item)
                
                if not lexicon_words:
                    continue  # Skip items with no relevant lexicon words
                
                # Generate profit story from news content
                profit_story = await self._generate_news_profit_story(news_item, lexicon_words)
                
                # Calculate hash correlation with BTC
                hash_correlation = await self._calculate_btc_hash_correlation(news_item, profit_story)
                
                # Determine vector bias and entropy class
                vector_bias = self._analyze_vector_bias(news_item, lexicon_words)
                entropy_class = self._calculate_entropy_class(news_item, profit_story)
                
                # Calculate processing priority
                priority = self._calculate_processing_priority(news_item, hash_correlation)
                
                # Create processed event
                news_event = NewsLexiconEvent(
                    news_item=news_item,
                    lexicon_words=lexicon_words,
                    profit_story=profit_story,
                    hash_correlation=hash_correlation,
                    vector_bias=vector_bias,
                    entropy_class=entropy_class,
                    processing_priority=priority
                )
                
                processed_events.append(news_event)
                
            except Exception as e:
                logger.error(f"Error processing news item through Lantern: {e}")
                continue
        
        # Add to backlog sorted by priority
        self.news_backlog.extend(processed_events)
        self.news_backlog.sort(key=lambda x: x.processing_priority, reverse=True)
        
        # Update metrics
        self.metrics["news_processed"] += len(processed_events)
        self.metrics["last_update"] = datetime.now()
        
        return processed_events
    
    async def _extract_lexicon_words(self, news_item: NewsItem) -> List[str]:
        """Extract relevant words from news that exist in Lantern's lexicon"""
        # Combine headline and content for analysis
        full_text = f"{news_item.headline} {news_item.content}".lower()
        
        # Get all words from Lantern's lexicon
        lexicon_words = list(self.lexicon_engine.lexicon.keys())
        
        # Find matches
        found_words = []
        for word in lexicon_words:
            if word.lower() in full_text:
                found_words.append(word)
        
        # Filter by relevance to crypto/trading context
        relevant_words = []
        for word in found_words:
            word_state = self.lexicon_engine.lexicon[word]
            # Include words with positive profit fitness or crypto relevance
            if (word_state.profit_fitness > 0.1 or 
                word in news_item.keywords_matched or
                any(crypto_word in word.lower() for crypto_word in 
                    ['bitcoin', 'btc', 'crypto', 'trade', 'profit', 'price'])):
                relevant_words.append(word)
        
        return relevant_words
    
    async def _generate_news_profit_story(self, news_item: NewsItem, lexicon_words: List[str]) -> List[str]:
        """Generate profit story from news using Lantern's story generation system"""
        try:
            # Use news timestamp as seed for deterministic story generation
            timestamp = news_item.timestamp
            
            # Create hash from news content for seeding
            news_hash = hashlib.sha256(
                f"{news_item.headline}_{timestamp.isoformat()}".encode()
            ).hexdigest()
            
            # Convert hash to seed index
            seed_index = int(news_hash[:6], 16) % len(self.lexicon_engine.lexicon)
            
            # Generate story biased toward lexicon words found in news
            story_length = min(len(lexicon_words) + 3, 12)  # Adaptive length
            
            # Get top profit words
            top_words = self.lexicon_engine.get_top_profit_words(100)
            
            # Merge news lexicon words with top profit words
            enhanced_word_set = list(set(lexicon_words + top_words))
            
            # Generate story
            story = self.lexicon_engine.generate_story(
                seed_index=seed_index,
                top_words=enhanced_word_set,
                length=story_length
            )
            
            return story
            
        except Exception as e:
            logger.error(f"Error generating profit story: {e}")
            # Fallback to lexicon words
            return lexicon_words[:8]
    
    async def _calculate_btc_hash_correlation(self, news_item: NewsItem, 
                                            profit_story: List[str]) -> NewsHashCorrelation:
        """Calculate correlation between news event and BTC hash structures"""
        
        # Generate news hash
        news_hash = hashlib.sha256(
            f"{news_item.source}:{news_item.headline}:{news_item.timestamp.isoformat()}".encode()
        ).hexdigest()
        
        # Get current BTC price/hash (simulated for now)
        current_time = datetime.now()
        btc_price = await self._get_current_btc_context()
        
        # Generate BTC hash
        btc_hash = self.lexicon_engine.generate_price_hash(btc_price, current_time)
        
        # Calculate hash similarity (Hamming distance)
        hash_similarity = self._calculate_hash_similarity(news_hash, btc_hash)
        
        # Calculate correlation score based on multiple factors
        correlation_score = self._calculate_correlation_score(
            news_item, profit_story, hash_similarity
        )
        
        # Estimate profit potential
        profit_potential = self.lexicon_engine.estimate_story_profitability(profit_story)
        
        correlation = NewsHashCorrelation(
            news_hash=news_hash,
            btc_hash=btc_hash,
            timestamp=current_time,
            correlation_score=correlation_score,
            sentiment_score=news_item.sentiment_score,
            lexicon_words=profit_story,
            hash_similarity=hash_similarity,
            profit_potential=profit_potential
        )
        
        # Cache correlation
        self.correlation_cache[news_hash] = correlation
        
        # Update metrics
        if correlation_score > self.config["correlation_threshold"]:
            self.metrics["correlations_found"] += 1
        
        return correlation
    
    def _calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """Calculate similarity between two hashes using Hamming distance"""
        if len(hash1) != len(hash2):
            return 0.0
        
        # Convert to binary and compare
        bin1 = bin(int(hash1[:16], 16))[2:].zfill(64)
        bin2 = bin(int(hash2[:16], 16))[2:].zfill(64)
        
        matches = sum(b1 == b2 for b1, b2 in zip(bin1, bin2))
        return matches / 64
    
    def _calculate_correlation_score(self, news_item: NewsItem, 
                                   profit_story: List[str], 
                                   hash_similarity: float) -> float:
        """Calculate comprehensive correlation score"""
        
        # Base factors
        sentiment_factor = abs(news_item.sentiment_score)  # Strong sentiment = higher correlation
        relevance_factor = news_item.relevance_score
        hash_factor = hash_similarity
        
        # Lexicon factor - how many high-profit words are in the story
        lexicon_factor = 0.0
        if profit_story:
            profit_scores = []
            for word in profit_story:
                if word in self.lexicon_engine.lexicon:
                    profit_scores.append(self.lexicon_engine.lexicon[word].profit_fitness)
            
            if profit_scores:
                lexicon_factor = np.mean(profit_scores)
        
        # Time factor - recent news has higher correlation
        time_delta = (datetime.now() - news_item.timestamp).total_seconds()
        time_factor = np.exp(-time_delta / self.config["hash_correlation_window"])
        
        # Weighted combination
        correlation = (
            0.3 * sentiment_factor +
            0.2 * relevance_factor +
            0.2 * hash_factor +
            0.2 * lexicon_factor +
            0.1 * time_factor
        )
        
        return min(correlation, 1.0)
    
    def _analyze_vector_bias(self, news_item: NewsItem, lexicon_words: List[str]) -> VectorBias:
        """Determine vector bias based on news sentiment and lexicon analysis"""
        
        # Analyze sentiment
        if news_item.sentiment_score > 0.5:
            base_bias = VectorBias.LONG
        elif news_item.sentiment_score < -0.5:
            base_bias = VectorBias.SHORT
        else:
            base_bias = VectorBias.HOLD
        
        # Analyze lexicon words for trading bias
        long_words = 0
        short_words = 0
        warning_words = 0
        
        for word in lexicon_words:
            if word in self.lexicon_engine.lexicon:
                word_state = self.lexicon_engine.lexicon[word]
                if word_state.vector_bias == VectorBias.LONG:
                    long_words += 1
                elif word_state.vector_bias == VectorBias.SHORT:
                    short_words += 1
                elif word_state.vector_bias == VectorBias.WARNING:
                    warning_words += 1
        
        # Override based on lexicon analysis
        if warning_words > 0:
            return VectorBias.WARNING
        elif long_words > short_words and news_item.sentiment_score > 0:
            return VectorBias.LONG
        elif short_words > long_words and news_item.sentiment_score < 0:
            return VectorBias.SHORT
        elif long_words == short_words:
            return VectorBias.ROTATE
        
        return base_bias
    
    def _calculate_entropy_class(self, news_item: NewsItem, profit_story: List[str]) -> EntropyClass:
        """Calculate entropy class based on news content and story complexity"""
        
        # Calculate narrative entropy from profit story
        entropy = self.lexicon_engine.calculate_narrative_entropy(profit_story)
        
        # Adjust based on news characteristics
        if news_item.relevance_score > 0.8:
            entropy += 0.2  # High relevance increases entropy
        
        if len(news_item.keywords_matched) > 5:
            entropy += 0.1  # Many keywords increase complexity
        
        # Classify
        if entropy > 0.7:
            return EntropyClass.HIGH
        elif entropy > 0.4:
            return EntropyClass.MEDIUM
        else:
            return EntropyClass.LOW
    
    def _calculate_processing_priority(self, news_item: NewsItem, 
                                     correlation: NewsHashCorrelation) -> float:
        """Calculate processing priority for backlog management"""
        
        # Base priority from correlation and sentiment
        priority = (
            correlation.correlation_score * 0.4 +
            abs(news_item.sentiment_score) * 0.3 +
            news_item.relevance_score * 0.2 +
            correlation.profit_potential * 0.1
        )
        
        # Boost priority for recent news
        time_delta = (datetime.now() - news_item.timestamp).total_seconds()
        recency_boost = np.exp(-time_delta / 1800)  # 30-minute decay
        priority *= (1 + recency_boost)
        
        return priority
    
    async def _manage_backlog_overflow(self):
        """Manage backlog when it exceeds maximum size"""
        
        # Sort by priority and keep top items
        self.news_backlog.sort(key=lambda x: x.processing_priority, reverse=True)
        
        # Remove lowest priority items
        overflow_count = len(self.news_backlog) - self.config["max_backlog_size"]
        if overflow_count > 0:
            removed_items = self.news_backlog[-overflow_count:]
            self.news_backlog = self.news_backlog[:-overflow_count]
            
            # Update metrics
            self.metrics["backlog_overflows"] += overflow_count
            
            # Trigger throttling if enabled
            if self.config["auto_throttle_enabled"]:
                await self._auto_throttle_processing()
            
            logger.warning(f"Removed {overflow_count} low-priority items from backlog")
    
    async def _auto_throttle_processing(self):
        """Automatically throttle processing to prevent overload"""
        
        # Check if BTC processor needs throttling
        processor_status = await self.btc_controller.get_system_status()
        
        if processor_status.get("memory_usage_gb", 0) > 8:
            # Disable memory-intensive features
            await self.btc_controller.disable_feature('mining_analysis')
            await self.btc_controller.disable_feature('storage')
            logger.info("Auto-throttled: Disabled memory-intensive BTC processor features")
        
        if processor_status.get("cpu_usage_percent", 0) > 70:
            # Reduce processing frequency
            self.config["processing_batch_size"] = max(5, self.config["processing_batch_size"] // 2)
            logger.info(f"Auto-throttled: Reduced batch size to {self.config['processing_batch_size']}")
    
    async def process_backlog_batch(self) -> List[Dict]:
        """Process a batch of items from the backlog"""
        
        if not self.news_backlog:
            return []
        
        batch_size = min(
            self.config["processing_batch_size"],
            len(self.news_backlog)
        )
        
        batch = self.news_backlog[:batch_size]
        self.news_backlog = self.news_backlog[batch_size:]
        
        processed_results = []
        
        for event in batch:
            try:
                # Update word fitness based on correlation strength
                if event.hash_correlation.correlation_score > self.config["correlation_threshold"]:
                    # Positive reinforcement for high correlation
                    profit_signal = event.hash_correlation.correlation_score * 100
                    self.lexicon_engine.update_fitness(event.lexicon_words, profit_signal)
                    self.metrics["profit_updates"] += 1
                
                # Store in memory systems
                await self._store_processed_event(event)
                
                # Create result
                result = {
                    "news_hash": event.news_item.hash_key,
                    "correlation_score": event.hash_correlation.correlation_score,
                    "vector_bias": event.vector_bias.value,
                    "entropy_class": event.entropy_class.value,
                    "profit_potential": event.hash_correlation.profit_potential,
                    "processing_priority": event.processing_priority,
                    "lexicon_words": event.lexicon_words,
                    "profit_story": event.profit_story
                }
                
                processed_results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing backlog item: {e}")
                continue
        
        return processed_results
    
    async def _store_processed_event(self, event: NewsLexiconEvent):
        """Store processed event in memory and hash systems"""
        
        # Store in memory agent
        memory_data = {
            "type": "lantern_news_event",
            "news_data": asdict(event.news_item),
            "lantern_data": {
                "lexicon_words": event.lexicon_words,
                "profit_story": event.profit_story,
                "vector_bias": event.vector_bias.value,
                "entropy_class": event.entropy_class.value,
                "correlation_score": event.hash_correlation.correlation_score,
                "profit_potential": event.hash_correlation.profit_potential
            },
            "timestamp": event.news_item.timestamp.isoformat(),
            "hash_key": event.news_item.hash_key
        }
        
        await self.memory_agent.store_event(memory_data)
        
        # Store in hash recollection system
        hash_data = {
            "news_hash": event.hash_correlation.news_hash,
            "btc_hash": event.hash_correlation.btc_hash,
            "correlation_score": event.hash_correlation.correlation_score,
            "lexicon_words": event.lexicon_words,
            "profit_story": event.profit_story,
            "vector_bias": event.vector_bias.value
        }
        
        self.hash_system.store_hash(event.news_item.hash_key, hash_data)
    
    async def _get_current_btc_context(self) -> float:
        """Get current BTC price context (placeholder - integrate with real data)"""
        # This would integrate with your actual BTC data source
        return 42000.0  # Placeholder
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        
        lexicon_stats = self.lexicon_engine.get_lexicon_stats()
        
        return {
            "processing_active": self.processing_active,
            "backlog_size": len(self.news_backlog),
            "correlation_cache_size": len(self.correlation_cache),
            "metrics": self.metrics,
            "config": self.config,
            "lexicon_stats": lexicon_stats,
            "last_update": self.metrics["last_update"].isoformat()
        }
    
    async def update_configuration(self, new_config: Dict):
        """Update system configuration"""
        
        for key, value in new_config.items():
            if key in self.config:
                self.config[key] = value
        
        # Apply configuration changes
        if "auto_throttle_enabled" in new_config and new_config["auto_throttle_enabled"]:
            await self._auto_throttle_processing()
        
        logger.info("Lantern News Intelligence Bridge configuration updated")
    
    async def start_continuous_processing(self, interval_seconds: int = 60):
        """Start continuous processing of news through Lantern system"""
        
        logger.info(f"Starting continuous Lantern news processing every {interval_seconds} seconds")
        
        while self.processing_active:
            try:
                # Get new news items
                news_items = await self.news_engine.aggregate_all_sources()
                
                if news_items:
                    # Process through Lantern
                    await self.process_news_through_lantern(news_items)
                
                # Process backlog batch
                batch_results = await self.process_backlog_batch()
                
                if batch_results:
                    logger.info(f"Processed {len(batch_results)} items through Lantern")
                
                # Update lexicon periodically
                if (datetime.now() - self.metrics["last_update"]).seconds > self.config["lexicon_update_interval"]:
                    await self._periodic_lexicon_update()
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in continuous Lantern processing: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_lexicon_update(self):
        """Perform periodic lexicon maintenance and optimization"""
        
        # Save lexicon state
        self.lexicon_engine._save_lexicon()
        
        # Clear old correlation cache
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.correlation_cache = {
            k: v for k, v in self.correlation_cache.items()
            if v.timestamp > cutoff_time
        }
        
        # Update metrics timestamp
        self.metrics["last_update"] = datetime.now()
        
        logger.info("Periodic lexicon update completed")


# Factory function for easy integration
def create_lantern_news_bridge(
    news_engine: Optional[NewsIntelligenceEngine] = None,
    btc_controller: Optional[BTCProcessorController] = None
) -> LanternNewsIntelligenceBridge:
    """Create and initialize Lantern News Intelligence Bridge"""
    
    bridge = LanternNewsIntelligenceBridge(
        news_engine=news_engine,
        btc_controller=btc_controller
    )
    
    return bridge


# Standalone runner for testing
async def main():
    """Test runner for Lantern News Intelligence Bridge"""
    
    bridge = create_lantern_news_bridge()
    await bridge.initialize()
    
    try:
        # Start processing
        await bridge.start_continuous_processing(interval_seconds=30)
        
    except KeyboardInterrupt:
        logger.info("Shutting down Lantern News Intelligence Bridge")
    finally:
        bridge.processing_active = False


if __name__ == "__main__":
    asyncio.run(main()) 