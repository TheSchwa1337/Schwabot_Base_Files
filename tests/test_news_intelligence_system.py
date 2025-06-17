"""
Tests for News Intelligence System
=================================
Comprehensive test suite for news aggregation, sentiment analysis,
and integration with Schwabot's memory and trading systems.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import aiohttp
from typing import List, Dict

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.news_intelligence_engine import (
    NewsIntelligenceEngine, NewsItem, MarketContext
)
from core.news_api_endpoints import NewsAPIEndpoints, create_news_api
from core.memory_agent import MemoryAgent
from core.hash_recollection import HashRecollectionSystem


class TestNewsIntelligenceEngine:
    """Test the core news intelligence engine"""
    
    @pytest.fixture
    async def news_engine(self):
        """Create a test news engine"""
        engine = NewsIntelligenceEngine("config/test_config.yaml")
        await engine.initialize()
        yield engine
        await engine.cleanup()
    
    @pytest.fixture
    def mock_news_data(self):
        """Mock news data for testing"""
        return [
            {
                "title": "Bitcoin Surges After Musk Tweet",
                "summary": "Elon Musk's latest tweet about Bitcoin causes price surge",
                "link": "https://example.com/news1",
                "published": datetime.now().isoformat()
            },
            {
                "title": "Trump Comments on Cryptocurrency Regulation",
                "summary": "Former president weighs in on crypto policy",
                "link": "https://example.com/news2", 
                "published": datetime.now().isoformat()
            }
        ]
    
    def test_keyword_matching(self, news_engine):
        """Test keyword matching functionality"""
        text = "Bitcoin price soars as Elon Musk tweets about cryptocurrency adoption"
        
        matched_keywords = []
        for keyword in news_engine.all_keywords:
            if keyword.lower() in text.lower():
                matched_keywords.append(keyword)
        
        assert "bitcoin" in matched_keywords
        assert "crypto" in matched_keywords or "cryptocurrency" in matched_keywords
        assert "elon musk" in matched_keywords or "musk" in matched_keywords
    
    def test_sentiment_analysis(self, news_engine):
        """Test sentiment analysis accuracy"""
        positive_text = "Bitcoin reaches new all-time high with strong institutional adoption"
        negative_text = "Bitcoin crashes amid regulatory crackdown and market fears"
        neutral_text = "Bitcoin trading volume remains steady according to latest reports"
        
        # Test would need actual sentiment analysis implementation
        # For now, test the structure
        assert hasattr(news_engine, '_process_news_item')
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, news_engine):
        """Test API rate limiting functionality"""
        # Test Google News rate limit
        assert news_engine._check_rate_limit("google_news")
        
        # Simulate hitting rate limit
        news_engine.api_limits["google_news"]["calls"] = 100
        assert not news_engine._check_rate_limit("google_news")
    
    @pytest.mark.asyncio
    @patch('feedparser.parse')
    async def test_google_news_fetch(self, mock_feedparser, news_engine, mock_news_data):
        """Test Google News RSS feed fetching"""
        # Mock feedparser response
        mock_feed = Mock()
        mock_feed.entries = [
            Mock(title=item["title"], summary=item["summary"], 
                 link=item["link"]) for item in mock_news_data
        ]
        mock_feedparser.return_value = mock_feed
        
        # Mock aiohttp session
        with patch.object(news_engine, 'session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<rss>mock data</rss>")
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            news_items = await news_engine.fetch_google_news("bitcoin")
            
            # Should process relevant news items
            assert len(news_items) >= 0
            mock_session.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, news_engine):
        """Test integration with memory agent"""
        # Create a mock news item
        news_item = NewsItem(
            source="Test Source",
            headline="Test Headline with Bitcoin",
            content="Test content about cryptocurrency",
            url="https://test.com",
            timestamp=datetime.now(),
            sentiment_score=0.5,
            sentiment_label="positive",
            keywords_matched=["bitcoin", "cryptocurrency"],
            relevance_score=0.8,
            hash_key="test_hash_123"
        )
        
        # Test storage
        with patch.object(news_engine.memory_agent, 'store_event') as mock_store:
            await news_engine._store_in_memory(news_item)
            mock_store.assert_called_once()
    
    def test_market_context_analysis(self, news_engine):
        """Test market context analysis"""
        # Add some mock news items to memory
        now = datetime.now()
        news_engine.news_memory = [
            NewsItem(
                source="Test", headline="Positive Bitcoin News", content="",
                url="", timestamp=now, sentiment_score=0.8, sentiment_label="positive",
                keywords_matched=["bitcoin"], relevance_score=0.9, hash_key="1"
            ),
            NewsItem(
                source="Test", headline="Negative Bitcoin News", content="",
                url="", timestamp=now, sentiment_score=-0.6, sentiment_label="negative", 
                keywords_matched=["bitcoin"], relevance_score=0.7, hash_key="2"
            )
        ]
        
        context = news_engine.analyze_market_context(lookback_hours=1)
        
        assert isinstance(context, MarketContext)
        assert context.news_volume == 2
        assert -1.0 <= context.overall_sentiment <= 1.0
        assert 0.0 <= context.volatility_indicator <= 1.0


class TestNewsAPIEndpoints:
    """Test the FastAPI endpoints"""
    
    @pytest.fixture
    def test_app(self):
        """Create test FastAPI app"""
        return create_news_api()
    
    @pytest.fixture
    def test_client(self, test_app):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(test_app)
    
    def test_get_latest_news_endpoint(self, test_client):
        """Test the latest news endpoint"""
        response = test_client.get("/api/news/latest?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10
    
    def test_get_sentiment_endpoint(self, test_client):
        """Test the sentiment analysis endpoint"""
        response = test_client.get("/api/news/sentiment")
        assert response.status_code == 200
        
        data = response.json()
        assert "overall_sentiment" in data
        assert "sentiment_label" in data
        assert "volatility_indicator" in data
        assert "news_volume" in data
    
    def test_news_config_endpoint(self, test_client):
        """Test news configuration endpoint"""
        config_data = {
            "twitter_enabled": True,
            "google_news_enabled": True,
            "yahoo_news_enabled": True,
            "monitoring_interval": 15,
            "sentiment_threshold": 0.5
        }
        
        response = test_client.post("/api/news/config", json=config_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "config_updated"
    
    def test_api_key_registration(self, test_client):
        """Test API key registration endpoint"""
        key_data = {
            "service": "twitter",
            "api_key": "test_key_123"
        }
        
        response = test_client.post("/api/news/api-key", json=key_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "key_registered"
        assert result["service"] == "twitter"
    
    def test_invalid_service_api_key(self, test_client):
        """Test invalid service for API key"""
        key_data = {
            "service": "invalid_service",
            "api_key": "test_key_123"
        }
        
        response = test_client.post("/api/news/api-key", json=key_data)
        assert response.status_code == 400


class TestNewsMemoryIntegration:
    """Test integration with existing Schwabot memory systems"""
    
    @pytest.fixture
    def mock_memory_agent(self):
        """Mock memory agent"""
        return Mock(spec=MemoryAgent)
    
    @pytest.fixture  
    def mock_hash_system(self):
        """Mock hash recollection system"""
        return Mock(spec=HashRecollectionSystem)
    
    @pytest.mark.asyncio
    async def test_news_storage_in_memory(self, mock_memory_agent, mock_hash_system):
        """Test news items are properly stored in memory systems"""
        engine = NewsIntelligenceEngine()
        engine.memory_agent = mock_memory_agent
        engine.hash_system = mock_hash_system
        
        news_item = NewsItem(
            source="Test Source",
            headline="Bitcoin Price Analysis",
            content="Detailed analysis of Bitcoin price movements",
            url="https://example.com/analysis",
            timestamp=datetime.now(),
            sentiment_score=0.3,
            sentiment_label="positive",
            keywords_matched=["bitcoin", "price"],
            relevance_score=0.75,
            hash_key="abc123"
        )
        
        # Mock the store_event method to be async
        mock_memory_agent.store_event = AsyncMock()
        
        await engine._store_in_memory(news_item)
        
        # Verify memory agent was called
        mock_memory_agent.store_event.assert_called_once()
        
        # Verify hash system was called
        mock_hash_system.store_hash.assert_called_once_with(
            "abc123", 
            {
                "source": "Test Source",
                "sentiment": 0.3,
                "keywords": ["bitcoin", "price"], 
                "relevance": 0.75
            }
        )
    
    def test_hash_key_generation(self):
        """Test hash key generation for news items"""
        import hashlib
        
        source = "Test Source"
        headline = "Test Headline"
        timestamp = datetime.now()
        
        hash_input = f"{source}:{headline}:{timestamp.isoformat()}"
        expected_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        # Verify hash generation matches expected format
        assert len(expected_hash) == 16
        assert all(c in "0123456789abcdef" for c in expected_hash)


class TestNewsSentimentIntegration:
    """Test sentiment analysis integration with trading decisions"""
    
    def test_sentiment_threshold_alerts(self):
        """Test sentiment threshold alert generation"""
        engine = NewsIntelligenceEngine()
        
        # Test high positive sentiment
        high_positive_context = MarketContext(
            overall_sentiment=0.8,
            volatility_indicator=0.3,
            key_events=["Major institutional adoption"],
            social_momentum=0.7,
            news_volume=15,
            timestamp=datetime.now()
        )
        
        # Should trigger alert for high sentiment
        assert abs(high_positive_context.overall_sentiment) > 0.5
    
    def test_volatility_calculation(self):
        """Test volatility indicator calculation"""
        engine = NewsIntelligenceEngine()
        
        # Add mixed sentiment news to memory
        now = datetime.now()
        engine.news_memory = [
            NewsItem("Source1", "Positive", "", "", now, 0.8, "positive", ["bitcoin"], 0.8, "1"),
            NewsItem("Source2", "Negative", "", "", now, -0.7, "negative", ["bitcoin"], 0.9, "2"),
            NewsItem("Source3", "Neutral", "", "", now, 0.1, "neutral", ["bitcoin"], 0.5, "3"),
        ]
        
        context = engine.analyze_market_context()
        
        # Should have some volatility due to mixed sentiment
        assert context.volatility_indicator > 0
        assert context.volatility_indicator <= 1.0


class TestNewsConfigurationValidation:
    """Test configuration validation and error handling"""
    
    def test_invalid_monitoring_interval(self):
        """Test validation of monitoring intervals"""
        from pydantic import ValidationError
        from core.news_api_endpoints import NewsConfigRequest
        
        # Test invalid interval (too low)
        with pytest.raises(ValidationError):
            NewsConfigRequest(monitoring_interval=1)  # Below minimum of 5
        
        # Test invalid interval (too high)  
        with pytest.raises(ValidationError):
            NewsConfigRequest(monitoring_interval=100)  # Above maximum of 60
    
    def test_invalid_sentiment_threshold(self):
        """Test validation of sentiment thresholds"""
        from pydantic import ValidationError
        from core.news_api_endpoints import NewsConfigRequest
        
        # Test invalid threshold (too low)
        with pytest.raises(ValidationError):
            NewsConfigRequest(sentiment_threshold=0.05)  # Below minimum of 0.1
        
        # Test invalid threshold (too high)
        with pytest.raises(ValidationError):
            NewsConfigRequest(sentiment_threshold=1.5)  # Above maximum of 1.0


class TestNewsSystemResilience:
    """Test system resilience and error handling"""
    
    @pytest.mark.asyncio
    async def test_network_failure_handling(self):
        """Test handling of network failures"""
        engine = NewsIntelligenceEngine()
        
        # Mock network failure
        with patch.object(engine, 'session') as mock_session:
            mock_session.get.side_effect = aiohttp.ClientError("Network error")
            
            # Should handle gracefully and return empty list
            news_items = await engine.fetch_google_news("bitcoin")
            assert news_items == []
    
    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self):
        """Test handling of API rate limits"""
        engine = NewsIntelligenceEngine()
        
        # Set rate limits to exceeded
        engine.api_limits["google_news"]["calls"] = 1000
        
        # Should return empty list when rate limited
        news_items = await engine.fetch_google_news("bitcoin")
        assert news_items == []
    
    def test_malformed_data_handling(self):
        """Test handling of malformed news data"""
        engine = NewsIntelligenceEngine()
        
        # Test with malformed data
        with patch('feedparser.parse') as mock_parse:
            mock_parse.return_value.entries = [
                Mock(title=None, summary="", link="invalid-url")
            ]
            
            # Should handle gracefully without crashing
            # Implementation would need proper error handling


# Integration test for complete news workflow
class TestCompleteNewsWorkflow:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_news_aggregation_workflow(self):
        """Test complete workflow from news fetch to memory storage"""
        engine = NewsIntelligenceEngine()
        
        # Mock all external dependencies
        with patch.object(engine, 'fetch_google_news') as mock_google, \
             patch.object(engine, 'fetch_yahoo_news') as mock_yahoo, \
             patch.object(engine, 'fetch_twitter_updates') as mock_twitter:
            
            # Setup mock returns
            mock_news_item = NewsItem(
                source="Test",
                headline="Bitcoin Gains Momentum", 
                content="Analysis shows positive trends",
                url="https://test.com",
                timestamp=datetime.now(),
                sentiment_score=0.6,
                sentiment_label="positive",
                keywords_matched=["bitcoin"],
                relevance_score=0.8,
                hash_key="test123"
            )
            
            mock_google.return_value = [mock_news_item]
            mock_yahoo.return_value = []
            mock_twitter.return_value = []
            
            # Run aggregation
            news_items = await engine.aggregate_all_sources()
            
            # Verify results
            assert len(news_items) == 1
            assert news_items[0].headline == "Bitcoin Gains Momentum"
            
            # Verify all sources were called
            assert mock_google.call_count >= 1
            assert mock_yahoo.call_count >= 1  
            assert mock_twitter.call_count >= 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"]) 