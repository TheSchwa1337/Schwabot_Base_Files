"""
News Intelligence Engine for Schwabot
=====================================
Aggregates news from multiple sources, performs sentiment analysis,
and integrates with memory/hash system for contextual trading intelligence.

Supported APIs:
- Google News API
- Yahoo Finance News
- Twitter API (free tier)
- Custom RSS feeds

Key Features:
- Real-time sentiment analysis
- Keyword tracking (Trump, Musk, BTC, crypto)
- Memory integration with hash keys
- Rate limiting and error handling
- Dashboard integration
"""

import asyncio
import aiohttp
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from textblob import TextBlob
import tweepy
import feedparser
import re
from urllib.parse import quote

from .memory_agent import MemoryAgent
from .hash_recollection import HashRecollectionSystem
from .sustainment_integration_hooks import SustainmentHooks
from .config_utils import load_config


@dataclass
class NewsItem:
    """Structured news item with sentiment analysis"""
    source: str
    headline: str
    content: str
    url: str
    timestamp: datetime
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str    # positive, negative, neutral
    keywords_matched: List[str]
    relevance_score: float  # 0.0 to 1.0
    hash_key: str


@dataclass
class MarketContext:
    """Market context derived from news analysis"""
    overall_sentiment: float
    volatility_indicator: float
    key_events: List[str]
    social_momentum: float
    news_volume: int
    timestamp: datetime


class NewsIntelligenceEngine:
    """Main news aggregation and analysis engine"""
    
    def __init__(self, config_path: str = "config/news_config.yaml"):
        self.config = load_config(config_path)
        self.memory_agent = MemoryAgent()
        self.hash_system = HashRecollectionSystem()
        self.hooks = SustainmentHooks()
        
        # API clients
        self.twitter_client = None
        self.session = None
        
        # Tracking keywords
        self.crypto_keywords = [
            "bitcoin", "btc", "cryptocurrency", "crypto", "blockchain",
            "ethereum", "eth", "coinbase", "binance", "mining"
        ]
        
        self.influence_keywords = [
            "trump", "donald trump", "elon musk", "musk", "tesla",
            "fed", "federal reserve", "inflation", "interest rates"
        ]
        
        self.all_keywords = self.crypto_keywords + self.influence_keywords
        
        # Rate limiting
        self.api_limits = {
            "google_news": {"calls": 0, "reset_time": time.time() + 3600},
            "yahoo_news": {"calls": 0, "reset_time": time.time() + 3600},
            "twitter": {"calls": 0, "reset_time": time.time() + 86400},
        }
        
        # Memory storage
        self.news_memory = []
        self.sentiment_history = []
        
    async def initialize(self):
        """Initialize API clients and connections"""
        # Initialize HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize Twitter client if credentials available
        twitter_config = self.config.get("twitter", {})
        if twitter_config.get("bearer_token"):
            self.twitter_client = tweepy.Client(
                bearer_token=twitter_config["bearer_token"],
                wait_on_rate_limit=True
            )
        
        print("News Intelligence Engine initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, api_name: str) -> bool:
        """Check if API rate limit allows request"""
        limit_info = self.api_limits.get(api_name, {})
        current_time = time.time()
        
        # Reset counter if time window passed
        if current_time > limit_info.get("reset_time", 0):
            if api_name == "twitter":
                limit_info.update({"calls": 0, "reset_time": current_time + 86400})
            else:
                limit_info.update({"calls": 0, "reset_time": current_time + 3600})
        
        # Check limits
        limits = {
            "google_news": 100,  # Per hour
            "yahoo_news": 200,   # Per hour  
            "twitter": 15        # Per day (free tier)
        }
        
        return limit_info["calls"] < limits.get(api_name, 0)
    
    def _increment_rate_limit(self, api_name: str):
        """Increment rate limit counter"""
        if api_name in self.api_limits:
            self.api_limits[api_name]["calls"] += 1
    
    async def fetch_google_news(self, query: str) -> List[NewsItem]:
        """Fetch news from Google News API"""
        if not self._check_rate_limit("google_news"):
            print("Google News rate limit exceeded")
            return []
        
        try:
            # Using RSS feed approach (free alternative)
            url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    news_items = []
                    for entry in feed.entries[:10]:  # Limit to 10 items
                        news_item = await self._process_news_item(
                            source="Google News",
                            headline=entry.title,
                            content=entry.get("summary", ""),
                            url=entry.link,
                            timestamp=datetime.now()
                        )
                        if news_item:
                            news_items.append(news_item)
                    
                    self._increment_rate_limit("google_news")
                    return news_items
                    
        except Exception as e:
            print(f"Error fetching Google News: {e}")
            
        return []
    
    async def fetch_yahoo_news(self, symbol: str = "BTC-USD") -> List[NewsItem]:
        """Fetch news from Yahoo Finance"""
        if not self._check_rate_limit("yahoo_news"):
            print("Yahoo News rate limit exceeded")
            return []
        
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    news_items = []
                    for entry in feed.entries[:10]:
                        news_item = await self._process_news_item(
                            source="Yahoo Finance",
                            headline=entry.title,
                            content=entry.get("summary", ""),
                            url=entry.link,
                            timestamp=datetime.now()
                        )
                        if news_item:
                            news_items.append(news_item)
                    
                    self._increment_rate_limit("yahoo_news")
                    return news_items
                    
        except Exception as e:
            print(f"Error fetching Yahoo News: {e}")
            
        return []
    
    async def fetch_twitter_updates(self, usernames: List[str] = None) -> List[NewsItem]:
        """Fetch tweets from specified users or search terms"""
        if not self.twitter_client or not self._check_rate_limit("twitter"):
            print("Twitter rate limit exceeded or client not available")
            return []
        
        if usernames is None:
            usernames = ["realDonaldTrump", "elonmusk"]
        
        try:
            news_items = []
            
            # Search for crypto-related tweets
            for keyword in ["bitcoin", "BTC", "cryptocurrency"][:3]:  # Limit searches
                tweets = self.twitter_client.search_recent_tweets(
                    query=f"{keyword} -is:retweet",
                    max_results=10,
                    tweet_fields=["created_at", "author_id", "public_metrics"]
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        news_item = await self._process_news_item(
                            source="Twitter",
                            headline=f"Tweet about {keyword}",
                            content=tweet.text,
                            url=f"https://twitter.com/user/status/{tweet.id}",
                            timestamp=tweet.created_at
                        )
                        if news_item:
                            news_items.append(news_item)
            
            self._increment_rate_limit("twitter")
            return news_items
            
        except Exception as e:
            print(f"Error fetching Twitter updates: {e}")
            
        return []
    
    async def _process_news_item(self, source: str, headline: str, content: str, 
                                url: str, timestamp: datetime) -> Optional[NewsItem]:
        """Process raw news item into structured format with sentiment analysis"""
        try:
            # Combine headline and content for analysis
            full_text = f"{headline} {content}"
            
            # Sentiment analysis
            blob = TextBlob(full_text)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Check for keyword matches
            keywords_matched = []
            text_lower = full_text.lower()
            
            for keyword in self.all_keywords:
                if keyword.lower() in text_lower:
                    keywords_matched.append(keyword)
            
            # Skip if no relevant keywords found
            if not keywords_matched:
                return None
            
            # Calculate relevance score
            relevance_score = len(keywords_matched) / len(self.all_keywords)
            
            # Generate hash key for memory system
            hash_input = f"{source}:{headline}:{timestamp.isoformat()}"
            hash_key = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            
            news_item = NewsItem(
                source=source,
                headline=headline,
                content=content[:500],  # Truncate content
                url=url,
                timestamp=timestamp,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                keywords_matched=keywords_matched,
                relevance_score=relevance_score,
                hash_key=hash_key
            )
            
            # Store in memory system
            await self._store_in_memory(news_item)
            
            return news_item
            
        except Exception as e:
            print(f"Error processing news item: {e}")
            return None
    
    async def _store_in_memory(self, news_item: NewsItem):
        """Store news item in memory and hash system"""
        try:
            # Store in memory agent
            memory_data = {
                "type": "news_item",
                "data": asdict(news_item),
                "timestamp": news_item.timestamp.isoformat(),
                "hash_key": news_item.hash_key
            }
            
            await self.memory_agent.store_event(memory_data)
            
            # Store in hash recollection system
            hash_data = {
                "source": news_item.source,
                "sentiment": news_item.sentiment_score,
                "keywords": news_item.keywords_matched,
                "relevance": news_item.relevance_score
            }
            
            self.hash_system.store_hash(news_item.hash_key, hash_data)
            
            # Add to local memory
            self.news_memory.append(news_item)
            
            # Keep only last 1000 items in memory
            if len(self.news_memory) > 1000:
                self.news_memory = self.news_memory[-1000:]
                
        except Exception as e:
            print(f"Error storing news item in memory: {e}")
    
    async def aggregate_all_sources(self) -> List[NewsItem]:
        """Aggregate news from all available sources"""
        all_news = []
        
        # Fetch from all sources concurrently
        tasks = [
            self.fetch_google_news("bitcoin cryptocurrency"),
            self.fetch_google_news("trump bitcoin"),
            self.fetch_google_news("elon musk bitcoin"),
            self.fetch_yahoo_news("BTC-USD"),
            self.fetch_twitter_updates()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                print(f"Error in news aggregation: {result}")
        
        # Sort by timestamp and relevance
        all_news.sort(key=lambda x: (x.timestamp, x.relevance_score), reverse=True)
        
        return all_news
    
    def analyze_market_context(self, lookback_hours: int = 24) -> MarketContext:
        """Analyze recent news to derive market context"""
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_news = [
            item for item in self.news_memory 
            if item.timestamp >= cutoff_time
        ]
        
        if not recent_news:
            return MarketContext(
                overall_sentiment=0.0,
                volatility_indicator=0.0,
                key_events=[],
                social_momentum=0.0,
                news_volume=0,
                timestamp=datetime.now()
            )
        
        # Calculate metrics
        sentiments = [item.sentiment_score for item in recent_news]
        overall_sentiment = sum(sentiments) / len(sentiments)
        
        # Volatility based on sentiment variance
        sentiment_variance = sum((s - overall_sentiment) ** 2 for s in sentiments) / len(sentiments)
        volatility_indicator = min(sentiment_variance * 10, 1.0)  # Normalize to 0-1
        
        # Key events (high relevance negative news)
        key_events = [
            item.headline for item in recent_news 
            if item.relevance_score > 0.7 and abs(item.sentiment_score) > 0.5
        ][:5]  # Top 5
        
        # Social momentum (Twitter activity)
        twitter_items = [item for item in recent_news if item.source == "Twitter"]
        social_momentum = len(twitter_items) / max(len(recent_news), 1)
        
        return MarketContext(
            overall_sentiment=overall_sentiment,
            volatility_indicator=volatility_indicator,
            key_events=key_events,
            social_momentum=social_momentum,
            news_volume=len(recent_news),
            timestamp=datetime.now()
        )
    
    async def get_sentiment_for_dashboard(self) -> Dict:
        """Get formatted sentiment data for dashboard display"""
        context = self.analyze_market_context()
        
        # Recent news items for display
        recent_items = self.news_memory[-10:] if self.news_memory else []
        
        return {
            "overall_sentiment": context.overall_sentiment,
            "sentiment_label": "positive" if context.overall_sentiment > 0.1 
                             else "negative" if context.overall_sentiment < -0.1 
                             else "neutral",
            "volatility_indicator": context.volatility_indicator,
            "news_volume": context.news_volume,
            "social_momentum": context.social_momentum,
            "key_events": context.key_events,
            "recent_headlines": [
                {
                    "headline": item.headline,
                    "source": item.source,
                    "sentiment": item.sentiment_label,
                    "relevance": item.relevance_score,
                    "timestamp": item.timestamp.isoformat()
                }
                for item in recent_items
            ],
            "last_updated": datetime.now().isoformat()
        }
    
    async def start_continuous_monitoring(self, interval_minutes: int = 15):
        """Start continuous news monitoring"""
        print(f"Starting continuous news monitoring every {interval_minutes} minutes")
        
        while True:
            try:
                # Aggregate news from all sources
                news_items = await self.aggregate_all_sources()
                print(f"Collected {len(news_items)} news items")
                
                # Analyze market context and trigger hooks if needed
                context = self.analyze_market_context()
                
                # Trigger sustainment hooks based on sentiment changes
                if abs(context.overall_sentiment) > 0.5:
                    await self.hooks.trigger_sentiment_alert(context)
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                print(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


# Standalone runner for testing
async def main():
    """Test runner for news intelligence engine"""
    engine = NewsIntelligenceEngine()
    await engine.initialize()
    
    try:
        # Test aggregation
        news_items = await engine.aggregate_all_sources()
        print(f"Collected {len(news_items)} news items")
        
        for item in news_items[:5]:  # Show first 5
            print(f"\n{item.source}: {item.headline}")
            print(f"Sentiment: {item.sentiment_label} ({item.sentiment_score:.2f})")
            print(f"Keywords: {', '.join(item.keywords_matched)}")
            print(f"Relevance: {item.relevance_score:.2f}")
        
        # Test market context
        context = engine.analyze_market_context()
        print(f"\nMarket Context:")
        print(f"Overall Sentiment: {context.overall_sentiment:.2f}")
        print(f"Volatility Indicator: {context.volatility_indicator:.2f}")
        print(f"News Volume: {context.news_volume}")
        print(f"Key Events: {context.key_events}")
        
    finally:
        await engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 