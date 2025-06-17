"""
News API Endpoints for Schwabot Dashboard
=========================================
FastAPI endpoints for news intelligence integration with the dashboard.
Provides real-time news data, sentiment analysis, and market context.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from .news_intelligence_engine import NewsIntelligenceEngine, NewsItem, MarketContext
from .config_utils import load_config


# Pydantic models for API responses
class NewsItemResponse(BaseModel):
    source: str
    headline: str
    content: str
    url: str
    timestamp: str
    sentiment_score: float
    sentiment_label: str
    keywords_matched: List[str]
    relevance_score: float
    hash_key: str


class MarketContextResponse(BaseModel):
    overall_sentiment: float
    sentiment_label: str
    volatility_indicator: float
    key_events: List[str]
    social_momentum: float
    news_volume: int
    timestamp: str


class SentimentHistoryResponse(BaseModel):
    timestamp: str
    sentiment: float
    volume: int


class NewsConfigRequest(BaseModel):
    twitter_enabled: bool = True
    google_news_enabled: bool = True
    yahoo_news_enabled: bool = True
    monitoring_interval: int = Field(default=15, ge=5, le=60)
    sentiment_threshold: float = Field(default=0.5, ge=0.1, le=1.0)


class APIKeyRequest(BaseModel):
    service: str  # "twitter", "newsapi", etc.
    api_key: str
    additional_params: Optional[Dict] = None


class NewsAPIEndpoints:
    """News API endpoints manager"""
    
    def __init__(self, app: FastAPI, news_engine: NewsIntelligenceEngine):
        self.app = app
        self.news_engine = news_engine
        self.websocket_connections = []
        self.monitoring_task = None
        
        # Setup CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure this for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/api/news/latest", response_model=List[NewsItemResponse])
        async def get_latest_news(limit: int = 20, source: Optional[str] = None):
            """Get latest news items"""
            try:
                news_items = self.news_engine.news_memory[-limit:] if self.news_engine.news_memory else []
                
                if source:
                    news_items = [item for item in news_items if item.source.lower() == source.lower()]
                
                # Convert to response format
                response_items = []
                for item in reversed(news_items):  # Most recent first
                    response_items.append(NewsItemResponse(
                        source=item.source,
                        headline=item.headline,
                        content=item.content,
                        url=item.url,
                        timestamp=item.timestamp.isoformat(),
                        sentiment_score=item.sentiment_score,
                        sentiment_label=item.sentiment_label,
                        keywords_matched=item.keywords_matched,
                        relevance_score=item.relevance_score,
                        hash_key=item.hash_key
                    ))
                
                return response_items
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")
        
        @self.app.get("/api/news/sentiment", response_model=MarketContextResponse)
        async def get_market_sentiment():
            """Get current market sentiment analysis"""
            try:
                context = self.news_engine.analyze_market_context()
                
                # Determine sentiment label
                if context.overall_sentiment > 0.1:
                    sentiment_label = "positive"
                elif context.overall_sentiment < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                return MarketContextResponse(
                    overall_sentiment=context.overall_sentiment,
                    sentiment_label=sentiment_label,
                    volatility_indicator=context.volatility_indicator,
                    key_events=context.key_events,
                    social_momentum=context.social_momentum,
                    news_volume=context.news_volume,
                    timestamp=context.timestamp.isoformat()
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")
        
        @self.app.get("/api/news/sentiment-history", response_model=List[SentimentHistoryResponse])
        async def get_sentiment_history(hours: int = 24):
            """Get sentiment history for charting"""
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                # Group news by hour and calculate average sentiment
                hourly_sentiment = {}
                
                for item in self.news_engine.news_memory:
                    if item.timestamp >= cutoff_time:
                        hour_key = item.timestamp.replace(minute=0, second=0, microsecond=0)
                        if hour_key not in hourly_sentiment:
                            hourly_sentiment[hour_key] = {"sentiments": [], "count": 0}
                        
                        hourly_sentiment[hour_key]["sentiments"].append(item.sentiment_score)
                        hourly_sentiment[hour_key]["count"] += 1
                
                # Calculate averages and format response
                history = []
                for hour, data in sorted(hourly_sentiment.items()):
                    avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
                    history.append(SentimentHistoryResponse(
                        timestamp=hour.isoformat(),
                        sentiment=avg_sentiment,
                        volume=data["count"]
                    ))
                
                return history
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching sentiment history: {str(e)}")
        
        @self.app.post("/api/news/refresh")
        async def refresh_news(background_tasks: BackgroundTasks):
            """Manually trigger news refresh"""
            try:
                background_tasks.add_task(self._fetch_and_broadcast_news)
                return {"status": "refresh_started", "message": "News refresh initiated"}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error refreshing news: {str(e)}")
        
        @self.app.post("/api/news/config")
        async def update_news_config(config: NewsConfigRequest):
            """Update news monitoring configuration"""
            try:
                # Update configuration in news engine
                # This would typically update the config file or database
                
                return {
                    "status": "config_updated",
                    "message": "News configuration updated successfully",
                    "config": config.dict()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")
        
        @self.app.post("/api/news/api-key")
        async def register_api_key(request: APIKeyRequest):
            """Register API key for news services"""
            try:
                # In production, encrypt and store securely
                # For now, just validate the request
                
                if request.service not in ["twitter", "newsapi", "polygon"]:
                    raise HTTPException(status_code=400, detail="Unsupported service")
                
                # Store encrypted key (implementation needed)
                # key_hash = hashlib.sha256(request.api_key.encode()).hexdigest()
                
                return {
                    "status": "key_registered",
                    "service": request.service,
                    "message": f"API key for {request.service} registered successfully"
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error registering API key: {str(e)}")
        
        @self.app.get("/api/news/sources")
        async def get_news_sources():
            """Get available news sources and their status"""
            try:
                sources = {
                    "google_news": {
                        "enabled": True,
                        "status": "active",
                        "rate_limit_remaining": 100 - self.news_engine.api_limits["google_news"]["calls"],
                        "last_fetch": datetime.now().isoformat()
                    },
                    "yahoo_finance": {
                        "enabled": True,
                        "status": "active", 
                        "rate_limit_remaining": 200 - self.news_engine.api_limits["yahoo_news"]["calls"],
                        "last_fetch": datetime.now().isoformat()
                    },
                    "twitter": {
                        "enabled": bool(self.news_engine.twitter_client),
                        "status": "active" if self.news_engine.twitter_client else "disabled",
                        "rate_limit_remaining": 15 - self.news_engine.api_limits["twitter"]["calls"],
                        "last_fetch": datetime.now().isoformat()
                    }
                }
                
                return sources
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error fetching sources: {str(e)}")
        
        @self.app.websocket("/ws/news")
        async def websocket_news_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time news updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive and send heartbeat
                    await asyncio.sleep(30)
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
            except Exception as e:
                print(f"WebSocket error: {e}")
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize news engine on startup"""
            await self.news_engine.initialize()
            
            # Start background monitoring task
            self.monitoring_task = asyncio.create_task(self._background_news_monitor())
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            await self.news_engine.cleanup()
    
    async def _fetch_and_broadcast_news(self):
        """Fetch news and broadcast to WebSocket connections"""
        try:
            # Fetch latest news
            news_items = await self.news_engine.aggregate_all_sources()
            
            if news_items:
                # Broadcast to WebSocket connections
                message = {
                    "type": "news_update",
                    "data": [
                        {
                            "source": item.source,
                            "headline": item.headline,
                            "sentiment": item.sentiment_label,
                            "relevance": item.relevance_score,
                            "timestamp": item.timestamp.isoformat()
                        }
                        for item in news_items[-5:]  # Last 5 items
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
                await self._broadcast_to_websockets(message)
            
            # Also broadcast sentiment update
            sentiment_data = await self.news_engine.get_sentiment_for_dashboard()
            sentiment_message = {
                "type": "sentiment_update",
                "data": sentiment_data,
                "timestamp": datetime.now().isoformat()
            }
            
            await self._broadcast_to_websockets(sentiment_message)
            
        except Exception as e:
            print(f"Error in news fetch and broadcast: {e}")
    
    async def _broadcast_to_websockets(self, message: Dict):
        """Broadcast message to all WebSocket connections"""
        if not self.websocket_connections:
            return
        
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.websocket_connections.remove(ws)
    
    async def _background_news_monitor(self):
        """Background task for continuous news monitoring"""
        try:
            await self.news_engine.start_continuous_monitoring(interval_minutes=15)
        except asyncio.CancelledError:
            print("News monitoring task cancelled")
        except Exception as e:
            print(f"Error in background news monitor: {e}")


# Factory function to create the news API
def create_news_api() -> FastAPI:
    """Create FastAPI app with news endpoints"""
    app = FastAPI(
        title="Schwabot News Intelligence API",
        description="Real-time news aggregation and sentiment analysis for cryptocurrency trading",
        version="1.0.0"
    )
    
    # Initialize news engine
    news_engine = NewsIntelligenceEngine()
    
    # Setup endpoints
    news_api = NewsAPIEndpoints(app, news_engine)
    
    return app


# For standalone testing
if __name__ == "__main__":
    import uvicorn
    app = create_news_api()
    uvicorn.run(app, host="0.0.0.0", port=8000) 