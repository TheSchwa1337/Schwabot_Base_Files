# 📰 Schwabot News Intelligence Integration

Comprehensive news aggregation and sentiment analysis system for cryptocurrency trading intelligence.

## 🎯 **Overview**

The News Intelligence Engine provides real-time news aggregation from multiple sources, performs sentiment analysis, and integrates with Schwabot's memory and hash systems for contextual trading decisions.

### **Key Features**
- **Multi-source aggregation**: Google News, Yahoo Finance, Twitter
- **Real-time sentiment analysis**: TextBlob-powered with keyword targeting
- **Memory integration**: Full integration with hash recollection system
- **Rate limiting**: Respects free tier API limits
- **Dashboard integration**: Live WebSocket updates to the UI
- **Keyword tracking**: Trump, Elon Musk, crypto-focused monitoring

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install -r requirements_news_integration.txt
```

### **2. Configure API Keys**
Create a `.env` file in the project root:
```bash
# Twitter API (get from developer.twitter.com)
TWITTER_BEARER_TOKEN=your_bearer_token_here
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret

# Optional: News API keys (for enhanced features)
NEWSAPI_KEY=your_newsapi_key
POLYGON_API_KEY=your_polygon_key
```

### **3. Update Configuration**
Edit `config/news_config.yaml` to customize:
- **Keywords to track**
- **Monitoring intervals**
- **Sentiment thresholds**
- **API rate limits**

### **4. Start the News Engine**
```bash
# As standalone service
python core/news_intelligence_engine.py

# Or integrated with main dashboard
python core/news_api_endpoints.py
```

### **5. Access the Dashboard**
Navigate to `http://localhost:3000` to see the enhanced dashboard with live news feeds.

## 📊 **Architecture Overview**

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   News Sources      │───▶│  News Intelligence   │───▶│   Memory & Hash     │
│                     │    │      Engine          │    │     Systems         │
│ • Google News RSS   │    │                      │    │                     │
│ • Yahoo Finance     │    │ • Sentiment Analysis │    │ • Memory Agent      │
│ • Twitter API       │    │ • Keyword Matching   │    │ • Hash Recollection │
│ • Custom RSS        │    │ • Rate Limiting      │    │ • Sustainment Hooks │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                       │
                                       ▼
                           ┌──────────────────────┐
                           │   FastAPI Endpoints  │
                           │                      │
                           │ • REST API           │
                           │ • WebSocket Streams  │
                           │ • Configuration      │
                           └──────────────────────┘
                                       │
                                       ▼
                           ┌──────────────────────┐
                           │  React Dashboard     │
                           │                      │
                           │ • Live News Feed     │
                           │ • Sentiment Charts   │
                           │ • Key Events         │
                           │ • Settings Panel     │
                           └──────────────────────┘
```

## 🔧 **API Endpoints**

### **REST API**
```
GET  /api/news/latest              # Get latest news items
GET  /api/news/sentiment           # Current market sentiment
GET  /api/news/sentiment-history   # Historical sentiment data
GET  /api/news/sources             # Available sources status
POST /api/news/refresh             # Manual news refresh
POST /api/news/config              # Update configuration
POST /api/news/api-key             # Register API keys
```

### **WebSocket**
```
WS   /ws/news                      # Real-time news updates
```

### **Example API Usage**
```python
import requests

# Get latest news
response = requests.get("http://localhost:8000/api/news/latest?limit=10")
news_items = response.json()

# Get current sentiment
response = requests.get("http://localhost:8000/api/news/sentiment")
sentiment = response.json()
print(f"Market sentiment: {sentiment['sentiment_label']} ({sentiment['overall_sentiment']:.2f})")
```

## 📈 **Dashboard Features**

### **1. Real-time News Feed**
- Live headlines from all sources
- Sentiment indicators (positive/negative/neutral)
- Keyword highlighting
- Relevance scoring

### **2. Market Sentiment Analysis**
- Overall sentiment gauge
- Historical sentiment trends
- Volatility indicators
- Social momentum tracking

### **3. Key Market Events**
- High-impact news detection
- Regulatory announcements
- Institutional adoption news
- Technical analysis alerts

### **4. Settings & Configuration**
- Source toggles (Twitter/Google/Yahoo)
- Monitoring intervals
- Sentiment thresholds
- API key management

## 🧠 **Memory Integration**

### **Hash Key Generation**
Each news item generates a unique hash key for memory storage:
```python
hash_input = f"{source}:{headline}:{timestamp.isoformat()}"
hash_key = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
```

### **Memory Storage Format**
```python
memory_data = {
    "type": "news_item",
    "data": news_item_dict,
    "timestamp": timestamp,
    "hash_key": hash_key
}

hash_data = {
    "source": source,
    "sentiment": sentiment_score,
    "keywords": matched_keywords,
    "relevance": relevance_score
}
```

### **Sustainment Hooks Integration**
The system triggers sustainment hooks for:
- **High sentiment changes** (> 0.5 threshold)
- **Volume spikes** (sudden news volume increase)
- **Key events** (regulatory/adoption news)

## 🎛️ **Configuration Options**

### **News Sources**
```yaml
# config/news_config.yaml
google_news:
  enabled: true
  rss_endpoint: "https://news.google.com/rss/search"

yahoo_finance:
  enabled: true
  rss_endpoint: "https://feeds.finance.yahoo.com/rss/2.0/headline"

twitter:
  bearer_token: "${TWITTER_BEARER_TOKEN}"
```

### **Keywords**
```yaml
keywords:
  crypto: ["bitcoin", "btc", "cryptocurrency", "crypto", "blockchain"]
  influencers: ["trump", "donald trump", "elon musk", "musk"]
  economic: ["fed", "federal reserve", "inflation", "rates"]
```

### **Rate Limits**
```yaml
rate_limits:
  google_news:
    requests_per_hour: 100
  twitter:
    requests_per_day: 15  # Free tier
```

## 🧪 **Testing**

### **Run Tests**
```bash
# Run all news integration tests
pytest tests/test_news_intelligence_system.py -v

# Run specific test categories
pytest tests/test_news_intelligence_system.py::TestNewsIntelligenceEngine -v
pytest tests/test_news_intelligence_system.py::TestNewsAPIEndpoints -v
pytest tests/test_news_intelligence_system.py::TestNewsMemoryIntegration -v
```

### **Test Coverage**
- **News aggregation** from all sources
- **Sentiment analysis** accuracy
- **Rate limiting** enforcement
- **Memory integration** with hash systems
- **API endpoint** functionality
- **WebSocket** real-time updates
- **Error handling** and resilience

## 🔍 **Monitoring & Debugging**

### **Logs**
```bash
# View news engine logs
tail -f logs/news_intelligence.log

# Check API rate limits
grep "rate limit" logs/news_intelligence.log
```

### **Health Checks**
```python
# Check system status
response = requests.get("http://localhost:8000/api/news/sources")
sources = response.json()

for source, status in sources.items():
    print(f"{source}: {status['status']} - {status['rate_limit_remaining']} calls remaining")
```

## 🚨 **Rate Limiting & Best Practices**

### **API Limits**
- **Google News RSS**: 100 requests/hour (free)
- **Yahoo Finance RSS**: 200 requests/hour (free)
- **Twitter API**: 15 requests/day (free tier)

### **Best Practices**
1. **Monitor rate limits** regularly
2. **Cache results** for repeated queries
3. **Use RSS feeds** when possible (free)
4. **Implement backoff** for failed requests
5. **Store credentials** securely

### **Error Handling**
```python
try:
    news_items = await engine.fetch_google_news("bitcoin")
except aiohttp.ClientError as e:
    logger.warning(f"Network error: {e}")
    # Fallback to cached data
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Continue with other sources
```

## 🔐 **Security Considerations**

### **API Key Storage**
- Store keys in **environment variables**
- **Never commit** keys to version control
- Use **encrypted storage** for production
- **Rotate keys** regularly

### **Rate Limiting**
- Implement **circuit breakers**
- **Respect API limits** strictly
- **Monitor usage** continuously
- **Implement backoff** strategies

## 🎯 **Trading Integration**

### **Sentiment Signals**
```python
# Example: Using sentiment for trading decisions
context = engine.analyze_market_context()

if context.overall_sentiment > 0.7:
    # Strong positive sentiment - consider long positions
    signal = "BULLISH"
elif context.overall_sentiment < -0.7:
    # Strong negative sentiment - consider short positions  
    signal = "BEARISH"
else:
    # Neutral sentiment - hold or reduce positions
    signal = "NEUTRAL"

# Integrate with existing trading logic
await trading_engine.process_sentiment_signal(signal, context)
```

### **Volatility Alerts**
```python
if context.volatility_indicator > 0.8:
    # High volatility detected
    await risk_manager.increase_monitoring()
    await position_manager.reduce_size()
```

## 📱 **Dashboard Usage**

### **Live News Feed**
- **Real-time updates** via WebSocket
- **Sentiment badges** on each headline
- **Keyword highlighting** for tracked terms
- **Click headlines** to open full articles

### **Sentiment Charts**
- **24-hour sentiment trends**
- **Volume indicators**
- **Volatility measures**
- **Social momentum tracking**

### **Settings Panel**
- **Toggle news sources** on/off
- **Adjust monitoring intervals**
- **Set sentiment thresholds**
- **Register API keys**

## 🛠️ **Troubleshooting**

### **Common Issues**

**1. No news items appearing**
```bash
# Check API credentials
echo $TWITTER_BEARER_TOKEN
# Check rate limits
grep "rate limit" logs/news_intelligence.log
```

**2. WebSocket connection issues**
```javascript
// Check browser console for WebSocket errors
// Verify backend is running on correct port
```

**3. Sentiment analysis not working**
```bash
# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

**4. Memory integration failures**
```python
# Check memory agent configuration
# Verify hash recollection system
```

## 🔄 **Continuous Monitoring**

### **System Health**
- **API response times**
- **Rate limit usage**
- **Memory consumption**
- **WebSocket connections**

### **Alerts**
- **API limit approaching**
- **High sentiment volatility**
- **System errors**
- **Connection failures**

## 📚 **Further Reading**

- [Twitter API Documentation](https://developer.twitter.com/en/docs)
- [Google News RSS Feeds](https://news.google.com/rss)
- [Yahoo Finance API](https://finance.yahoo.com)
- [TextBlob Sentiment Analysis](https://textblob.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🎉 **Success Metrics**

Once properly configured, you should see:
- ✅ **Real-time news** flowing into the dashboard
- ✅ **Sentiment analysis** updating every 15 minutes
- ✅ **Memory integration** storing news events
- ✅ **Trading signals** based on sentiment thresholds
- ✅ **Performance** within rate limits

**The news intelligence system is now providing contextual market awareness to enhance Schwabot's trading decisions!** 