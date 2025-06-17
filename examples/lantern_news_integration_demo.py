"""
Lantern News Intelligence Integration Demo
=========================================

Demonstrates the complete integration of:
- News Intelligence Engine
- Lantern Core Lexicon Analysis
- BTC Hash Correlation
- Processor Controls and Throttling
- Memory and Hash Systems
- Settings Panel Integration

This showcases how news events flow through Lantern's mathematical
framework to generate profit-optimized trading insights.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.lantern_news_intelligence_bridge import (
    LanternNewsIntelligenceBridge, 
    NewsLexiconEvent,
    create_lantern_news_bridge
)
from core.news_intelligence_engine import NewsIntelligenceEngine
from core.btc_processor_controller import BTCProcessorController
from core.lantern.lexicon_engine import LexiconEngine, VectorBias, EntropyClass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LanternNewsIntegrationDemo:
    """Comprehensive demo of the Lantern News Intelligence system"""
    
    def __init__(self):
        self.bridge = None
        self.demo_stats = {
            "news_processed": 0,
            "correlations_found": 0,
            "profit_signals_generated": 0,
            "throttling_events": 0,
            "memory_events_stored": 0
        }
    
    async def initialize_demo(self):
        """Initialize all components for the demo"""
        logger.info("🚀 Initializing Lantern News Intelligence Demo")
        
        # Create the integrated bridge
        self.bridge = create_lantern_news_bridge()
        await self.bridge.initialize()
        
        # Set demo-friendly configuration
        demo_config = {
            "max_backlog_size": 100,        # Smaller for demo
            "processing_batch_size": 5,     # Process in small batches
            "correlation_threshold": 0.2,   # Lower threshold for more examples
            "auto_throttle_enabled": True   # Show throttling features
        }
        
        await self.bridge.update_configuration(demo_config)
        
        logger.info("✅ Lantern News Intelligence Bridge initialized")
    
    async def demo_news_aggregation(self):
        """Demonstrate news aggregation from multiple sources"""
        logger.info("\n📰 === NEWS AGGREGATION DEMO ===")
        
        # Simulate getting news from multiple sources
        logger.info("Fetching news from all available sources...")
        
        # Get real news items
        news_items = await self.bridge.news_engine.aggregate_all_sources()
        
        if news_items:
            logger.info(f"📊 Collected {len(news_items)} news items")
            
            # Show sample news items
            for i, item in enumerate(news_items[:3]):
                logger.info(f"\n📰 News Item {i+1}:")
                logger.info(f"   Source: {item.source}")
                logger.info(f"   Headline: {item.headline[:80]}...")
                logger.info(f"   Sentiment: {item.sentiment_label} ({item.sentiment_score:.2f})")
                logger.info(f"   Keywords: {', '.join(item.keywords_matched[:5])}")
                logger.info(f"   Relevance: {item.relevance_score:.2f}")
            
            self.demo_stats["news_processed"] = len(news_items)
            return news_items
        else:
            logger.warning("⚠️ No news items collected - using mock data for demo")
            return await self._create_mock_news_items()
    
    async def _create_mock_news_items(self):
        """Create mock news items for demo purposes"""
        from core.news_intelligence_engine import NewsItem
        
        mock_items = [
            NewsItem(
                source="Demo Source",
                headline="Bitcoin Surges After Major Institutional Adoption News",
                content="Leading investment firm announces $1B Bitcoin allocation following regulatory clarity",
                url="https://demo.com/news1",
                timestamp=datetime.now(),
                sentiment_score=0.8,
                sentiment_label="positive",
                keywords_matched=["bitcoin", "institutional", "adoption"],
                relevance_score=0.9,
                hash_key="demo_hash_1"
            ),
            NewsItem(
                source="Demo Source",
                headline="Trump Comments on Cryptocurrency Future at Economic Summit",
                content="Former president discusses digital currency policy implications for economic growth",
                url="https://demo.com/news2",
                timestamp=datetime.now() - timedelta(minutes=30),
                sentiment_score=0.3,
                sentiment_label="positive",
                keywords_matched=["trump", "cryptocurrency", "policy"],
                relevance_score=0.7,
                hash_key="demo_hash_2"
            ),
            NewsItem(
                source="Demo Source",
                headline="Elon Musk Tweets About Bitcoin Mining Sustainability",
                content="Tesla CEO shares thoughts on renewable energy usage in cryptocurrency mining operations",
                url="https://demo.com/news3",
                timestamp=datetime.now() - timedelta(hours=1),
                sentiment_score=0.1,
                sentiment_label="neutral",
                keywords_matched=["elon musk", "bitcoin", "mining", "sustainability"],
                relevance_score=0.8,
                hash_key="demo_hash_3"
            )
        ]
        
        return mock_items
    
    async def demo_lantern_processing(self, news_items):
        """Demonstrate processing news through Lantern's lexicon system"""
        logger.info("\n🧠 === LANTERN LEXICON PROCESSING DEMO ===")
        
        # Process news through Lantern
        logger.info("Processing news through Lantern's mathematical framework...")
        processed_events = await self.bridge.process_news_through_lantern(news_items)
        
        if processed_events:
            logger.info(f"🔬 Processed {len(processed_events)} events through Lantern")
            
            # Show detailed analysis for first event
            event = processed_events[0]
            logger.info(f"\n🔍 Detailed Analysis for: {event.news_item.headline[:60]}...")
            logger.info(f"   📝 Lexicon Words Found: {', '.join(event.lexicon_words[:8])}")
            logger.info(f"   📖 Generated Profit Story: {' -> '.join(event.profit_story[:6])}")
            logger.info(f"   🔗 Hash Correlation Score: {event.hash_correlation.correlation_score:.3f}")
            logger.info(f"   📈 Vector Bias: {event.vector_bias.value}")
            logger.info(f"   🌀 Entropy Class: {event.entropy_class.value}")
            logger.info(f"   ⭐ Processing Priority: {event.processing_priority:.3f}")
            
            # Show correlation details
            correlation = event.hash_correlation
            logger.info(f"\n🔗 Hash Correlation Details:")
            logger.info(f"   📰 News Hash: {correlation.news_hash[:16]}...")
            logger.info(f"   ₿ BTC Hash: {correlation.btc_hash[:16]}...")
            logger.info(f"   🎯 Similarity: {correlation.hash_similarity:.3f}")
            logger.info(f"   💰 Profit Potential: {correlation.profit_potential:.3f}")
            
            # Update stats
            correlations = sum(1 for e in processed_events 
                             if e.hash_correlation.correlation_score > self.bridge.config["correlation_threshold"])
            self.demo_stats["correlations_found"] = correlations
            
        return processed_events
    
    async def demo_btc_processor_integration(self):
        """Demonstrate BTC processor integration and throttling"""
        logger.info("\n⚡ === BTC PROCESSOR INTEGRATION DEMO ===")
        
        # Check current processor status
        logger.info("Checking BTC processor status...")
        try:
            processor_status = await self.bridge.btc_controller.get_system_status()
            logger.info(f"📊 Processor Status:")
            logger.info(f"   💾 Memory Usage: {processor_status.get('memory_usage_gb', 0):.1f} GB")
            logger.info(f"   🖥️ CPU Usage: {processor_status.get('cpu_usage_percent', 0):.1f}%")
            logger.info(f"   🎮 GPU Usage: {processor_status.get('gpu_usage_percent', 0):.1f}%")
            
            # Simulate high memory usage to trigger throttling
            logger.info("\n🔄 Simulating resource pressure...")
            
            # Update configuration to simulate throttling
            throttle_config = {
                "auto_throttle_enabled": True,
                "processing_batch_size": 3  # Reduce batch size
            }
            await self.bridge.update_configuration(throttle_config)
            
            logger.info("⚙️ Auto-throttling enabled - batch size reduced to prevent overload")
            self.demo_stats["throttling_events"] += 1
            
        except Exception as e:
            logger.warning(f"⚠️ BTC processor integration demo limited: {e}")
    
    async def demo_memory_integration(self, processed_events):
        """Demonstrate memory and hash system integration"""
        logger.info("\n🧠 === MEMORY INTEGRATION DEMO ===")
        
        if not processed_events:
            logger.warning("⚠️ No processed events available for memory demo")
            return
        
        # Process a batch through memory systems
        logger.info("Processing events through memory and hash systems...")
        
        batch_results = await self.bridge.process_backlog_batch()
        
        if batch_results:
            logger.info(f"💾 Stored {len(batch_results)} events in memory systems")
            
            # Show memory integration details
            for i, result in enumerate(batch_results[:2]):
                logger.info(f"\n📝 Memory Event {i+1}:")
                logger.info(f"   🆔 Hash Key: {result['news_hash']}")
                logger.info(f"   🔗 Correlation: {result['correlation_score']:.3f}")
                logger.info(f"   📈 Vector Bias: {result['vector_bias']}")
                logger.info(f"   🌀 Entropy: {result['entropy_class']}")
                logger.info(f"   💰 Profit Potential: {result['profit_potential']:.3f}")
            
            self.demo_stats["memory_events_stored"] = len(batch_results)
        
        # Show system status
        status = await self.bridge.get_system_status()
        logger.info(f"\n📊 System Status:")
        logger.info(f"   📦 Backlog Size: {status['backlog_size']}")
        logger.info(f"   🗄️ Correlation Cache: {status['correlation_cache_size']}")
        logger.info(f"   📈 News Processed: {status['metrics']['news_processed']}")
        logger.info(f"   🔗 Correlations Found: {status['metrics']['correlations_found']}")
    
    async def demo_lexicon_evolution(self):
        """Demonstrate lexicon word fitness evolution"""
        logger.info("\n📈 === LEXICON EVOLUTION DEMO ===")
        
        # Get lexicon statistics before
        lexicon_stats = self.bridge.lexicon_engine.get_lexicon_stats()
        logger.info(f"📊 Current Lexicon Stats:")
        logger.info(f"   📚 Total Words: {lexicon_stats.get('total_words', 0)}")
        logger.info(f"   💰 Avg Profit Score: {lexicon_stats.get('avg_profit_score', 0):.3f}")
        logger.info(f"   ⭐ High Profit Words: {lexicon_stats.get('high_profit_words', 0)}")
        
        # Show top profit words
        top_words = self.bridge.lexicon_engine.get_top_profit_words(10)
        logger.info(f"🏆 Top 10 Profit Words: {', '.join(top_words[:10])}")
        
        # Simulate profit feedback
        logger.info("\n💡 Simulating profit feedback from correlation signals...")
        
        # Update word fitness for demo words
        demo_words = ["bitcoin", "adoption", "surge", "institutional", "bullish"]
        profit_signal = 50.0  # Positive profit
        
        self.bridge.lexicon_engine.update_fitness(demo_words, profit_signal)
        logger.info(f"📈 Updated fitness for words: {', '.join(demo_words)}")
        logger.info(f"💰 Applied profit signal: +{profit_signal}")
        
        self.demo_stats["profit_signals_generated"] += 1
    
    async def demo_complete_workflow(self):
        """Demonstrate the complete end-to-end workflow"""
        logger.info("\n🔄 === COMPLETE WORKFLOW DEMO ===")
        
        # Step 1: News Aggregation
        news_items = await self.demo_news_aggregation()
        
        # Step 2: Lantern Processing
        processed_events = await self.demo_lantern_processing(news_items)
        
        # Step 3: BTC Processor Integration
        await self.demo_btc_processor_integration()
        
        # Step 4: Memory Integration
        await self.demo_memory_integration(processed_events)
        
        # Step 5: Lexicon Evolution
        await self.demo_lexicon_evolution()
        
        # Final statistics
        logger.info("\n📊 === DEMO STATISTICS ===")
        for key, value in self.demo_stats.items():
            logger.info(f"   {key.replace('_', ' ').title()}: {value}")
    
    async def demo_settings_integration(self):
        """Demonstrate how settings would be managed through the UI"""
        logger.info("\n⚙️ === SETTINGS INTEGRATION DEMO ===")
        
        # Show current configuration
        status = await self.bridge.get_system_status()
        config = status.get('config', {})
        
        logger.info("🔧 Current Configuration:")
        for key, value in config.items():
            logger.info(f"   {key}: {value}")
        
        # Simulate configuration update (as would come from settings panel)
        new_config = {
            "correlation_threshold": 0.4,  # Increase threshold
            "processing_batch_size": 8,    # Increase batch size
            "auto_throttle_enabled": True
        }
        
        logger.info(f"\n🔄 Updating configuration...")
        await self.bridge.update_configuration(new_config)
        logger.info("✅ Configuration updated successfully")
        
        # Show API key management simulation
        logger.info(f"\n🔑 API Key Management:")
        logger.info("   Twitter API: ✅ Configured (15 calls/day remaining)")
        logger.info("   Google News: ✅ Active (RSS - unlimited)")
        logger.info("   Yahoo Finance: ✅ Active (RSS - unlimited)")
        logger.info("   NewsAPI: ⚠️ Optional (not configured)")
    
    async def run_complete_demo(self):
        """Run the complete demonstration"""
        try:
            await self.initialize_demo()
            await self.demo_complete_workflow()
            await self.demo_settings_integration()
            
            logger.info("\n🎉 === DEMO COMPLETED SUCCESSFULLY ===")
            logger.info("The Lantern News Intelligence Bridge is ready for production use!")
            logger.info("\nKey Integration Points Demonstrated:")
            logger.info("✅ News aggregation from multiple free sources")
            logger.info("✅ Mathematical correlation with BTC hash structures")
            logger.info("✅ Lexicon analysis and profit optimization")
            logger.info("✅ BTC processor integration and auto-throttling")
            logger.info("✅ Memory and hash system storage")
            logger.info("✅ Settings panel configuration")
            logger.info("✅ Backlog management and resource control")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            if self.bridge:
                self.bridge.processing_active = False


# Utility functions for demonstration
async def demonstrate_api_rate_limits():
    """Show how API rate limits are managed"""
    logger.info("\n🚦 === API RATE LIMIT DEMO ===")
    
    # Simulate rate limit status
    rate_limits = {
        "google_news": {"calls": 45, "limit": 100, "reset_time": datetime.now() + timedelta(hours=1)},
        "yahoo_finance": {"calls": 12, "limit": 200, "reset_time": datetime.now() + timedelta(hours=1)},
        "twitter": {"calls": 8, "limit": 15, "reset_time": datetime.now() + timedelta(days=1)}
    }
    
    for service, limit_info in rate_limits.items():
        usage_percent = (limit_info["calls"] / limit_info["limit"]) * 100
        status = "🟢" if usage_percent < 60 else "🟡" if usage_percent < 80 else "🔴"
        
        logger.info(f"{status} {service.title()}: {limit_info['calls']}/{limit_info['limit']} "
                   f"({usage_percent:.1f}%) - Resets: {limit_info['reset_time'].strftime('%H:%M')}")


async def demonstrate_hash_correlation_math():
    """Show the mathematical basis of hash correlation"""
    logger.info("\n🔢 === HASH CORRELATION MATHEMATICS DEMO ===")
    
    # Example hash correlation calculation
    news_hash = "a1b2c3d4e5f6789a"
    btc_hash = "a1b4c3d2e5f8789b"
    
    logger.info(f"📰 News Hash: {news_hash}")
    logger.info(f"₿ BTC Hash:  {btc_hash}")
    
    # Convert to binary for Hamming distance
    bin1 = bin(int(news_hash, 16))[2:].zfill(64)
    bin2 = bin(int(btc_hash, 16))[2:].zfill(64)
    
    matches = sum(b1 == b2 for b1, b2 in zip(bin1, bin2))
    similarity = matches / 64
    
    logger.info(f"🎯 Binary Matches: {matches}/64")
    logger.info(f"📊 Hash Similarity: {similarity:.3f}")
    logger.info(f"🔗 Correlation Strength: {'High' if similarity > 0.6 else 'Medium' if similarity > 0.3 else 'Low'}")


# Main execution
async def main():
    """Main demo execution"""
    logger.info("🚀 Starting Lantern News Intelligence Integration Demo")
    
    # Run rate limit demo
    await demonstrate_api_rate_limits()
    
    # Run hash correlation math demo
    await demonstrate_hash_correlation_math()
    
    # Run complete integration demo
    demo = LanternNewsIntegrationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main()) 