import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Activity, DollarSign, Clock, Zap, Globe, Twitter, RefreshCw, Settings, Bell } from 'lucide-react';

// Types for news data
interface NewsItem {
  source: string;
  headline: string;
  content: string;
  url: string;
  timestamp: string;
  sentiment_score: number;
  sentiment_label: string;
  keywords_matched: string[];
  relevance_score: number;
  hash_key: string;
}

interface MarketContext {
  overall_sentiment: number;
  sentiment_label: string;
  volatility_indicator: number;
  key_events: string[];
  social_momentum: number;
  news_volume: number;
  timestamp: string;
}

interface SentimentHistory {
  timestamp: string;
  sentiment: number;
  volume: number;
}

const NewsIntegrationDashboard: React.FC = () => {
  // State management
  const [currentTime, setCurrentTime] = useState(new Date());
  const [systemStatus, setSystemStatus] = useState('active');
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [marketContext, setMarketContext] = useState<MarketContext | null>(null);
  const [sentimentHistory, setSentimentHistory] = useState<SentimentHistory[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [newsConfig, setNewsConfig] = useState({
    twitterEnabled: true,
    googleNewsEnabled: true,
    yahooNewsEnabled: true,
    monitoringInterval: 15,
    sentimentThreshold: 0.5
  });

  // WebSocket connection
  const [ws, setWs] = useState<WebSocket | null>(null);

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Initialize WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const websocket = new WebSocket('ws://localhost:8000/ws/news');
      
      websocket.onopen = () => {
        console.log('WebSocket connected');
        setWsConnected(true);
      };
      
      websocket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
      };
      
      websocket.onclose = () => {
        console.log('WebSocket disconnected');
        setWsConnected(false);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setWsConnected(false);
      };
      
      setWs(websocket);
    };

    connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  // Handle WebSocket messages
  const handleWebSocketMessage = (message: any) => {
    switch (message.type) {
      case 'news_update':
        // Update news items with new data
        setNewsItems(prev => [...message.data, ...prev].slice(0, 50)); // Keep last 50
        break;
      case 'sentiment_update':
        // Update market context
        setMarketContext(message.data);
        break;
      case 'heartbeat':
        // Connection is alive
        break;
    }
  };

  // Fetch initial data
  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    setIsLoading(true);
    try {
      // Fetch latest news
      const newsResponse = await fetch('/api/news/latest?limit=20');
      const newsData = await newsResponse.json();
      setNewsItems(newsData);

      // Fetch market sentiment
      const sentimentResponse = await fetch('/api/news/sentiment');
      const sentimentData = await sentimentResponse.json();
      setMarketContext(sentimentData);

      // Fetch sentiment history
      const historyResponse = await fetch('/api/news/sentiment-history?hours=24');
      const historyData = await historyResponse.json();
      setSentimentHistory(historyData);

    } catch (error) {
      console.error('Error fetching initial data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Manual refresh
  const handleRefresh = async () => {
    setIsLoading(true);
    try {
      await fetch('/api/news/refresh', { method: 'POST' });
      // Data will be updated via WebSocket
    } catch (error) {
      console.error('Error refreshing news:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Update configuration
  const handleConfigUpdate = async () => {
    try {
      await fetch('/api/news/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newsConfig)
      });
      setShowSettings(false);
    } catch (error) {
      console.error('Error updating config:', error);
    }
  };

  // Component helpers
  const StatusIndicator: React.FC<{ status: string; label: string }> = ({ status, label }) => {
    const getColor = () => {
      switch(status) {
        case 'active': return 'text-green-500';
        case 'warning': return 'text-yellow-500';
        case 'error': return 'text-red-500';
        default: return 'text-gray-500';
      }
    };

    const getIcon = () => {
      switch(status) {
        case 'active': return <CheckCircle className="w-4 h-4" />;
        case 'warning': return <AlertTriangle className="w-4 h-4" />;
        case 'error': return <AlertTriangle className="w-4 h-4" />;
        default: return <Activity className="w-4 h-4" />;
      }
    };

    return (
      <div className={`flex items-center space-x-2 ${getColor()}`}>
        {getIcon()}
        <span className="text-sm font-medium">{label}</span>
      </div>
    );
  };

  const SentimentBadge: React.FC<{ sentiment: string; score: number }> = ({ sentiment, score }) => {
    const getColor = () => {
      switch(sentiment) {
        case 'positive': return 'bg-green-500';
        case 'negative': return 'bg-red-500';
        default: return 'bg-gray-500';
      }
    };

    return (
      <span className={`px-2 py-1 text-xs rounded-full text-white ${getColor()}`}>
        {sentiment} ({score > 0 ? '+' : ''}{score.toFixed(2)})
      </span>
    );
  };

  // Entropy data (keeping existing mock for now)
  const entropyData = [
    { time: '09:00', value: 0.45, threshold: 0.7 },
    { time: '09:05', value: 0.62, threshold: 0.7 },
    { time: '09:10', value: 0.78, threshold: 0.7 },
    { time: '09:15', value: 0.54, threshold: 0.7 },
    { time: '09:20', value: 0.39, threshold: 0.7 },
    { time: '09:25', value: 0.67, threshold: 0.7 }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-blue-400">Schwabot Trading System</h1>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-lg font-mono">{currentTime.toLocaleTimeString()}</div>
              <StatusIndicator status="active" label="System Operational" />
            </div>
            <div className="flex space-x-2">
              <button
                onClick={handleRefresh}
                disabled={isLoading}
                className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 bg-gray-600 hover:bg-gray-700 rounded-lg"
              >
                <Settings className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="mb-6 bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">News Configuration</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={newsConfig.twitterEnabled}
                  onChange={(e) => setNewsConfig({...newsConfig, twitterEnabled: e.target.checked})}
                />
                <span>Twitter Monitoring</span>
              </label>
            </div>
            <div>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={newsConfig.googleNewsEnabled}
                  onChange={(e) => setNewsConfig({...newsConfig, googleNewsEnabled: e.target.checked})}
                />
                <span>Google News</span>
              </label>
            </div>
            <div>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={newsConfig.yahooNewsEnabled}
                  onChange={(e) => setNewsConfig({...newsConfig, yahooNewsEnabled: e.target.checked})}
                />
                <span>Yahoo Finance</span>
              </label>
            </div>
            <div>
              <label className="block text-sm">
                <span>Monitoring Interval (minutes)</span>
                <input
                  type="number"
                  min="5"
                  max="60"
                  value={newsConfig.monitoringInterval}
                  onChange={(e) => setNewsConfig({...newsConfig, monitoringInterval: parseInt(e.target.value)})}
                  className="w-full mt-1 p-2 bg-gray-700 rounded"
                />
              </label>
            </div>
          </div>
          <div className="mt-4 flex space-x-2">
            <button
              onClick={handleConfigUpdate}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
            >
              Save Configuration
            </button>
            <button
              onClick={() => setShowSettings(false)}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Key Metrics Row */}
      <div className="grid grid-cols-5 gap-6 mb-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Total P&L</h3>
            <DollarSign className="w-5 h-5 text-green-400" />
          </div>
          <div className="text-2xl font-bold text-green-400">+$127,543</div>
          <div className="text-sm text-gray-400">+15.7% this month</div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Market Sentiment</h3>
            <Globe className="w-5 h-5 text-blue-400" />
          </div>
          <div className="text-2xl font-bold">
            {marketContext ? (
              <SentimentBadge 
                sentiment={marketContext.sentiment_label} 
                score={marketContext.overall_sentiment} 
              />
            ) : (
              <span className="text-gray-400">Loading...</span>
            )}
          </div>
          <div className="text-sm text-gray-400">
            {marketContext ? `${marketContext.news_volume} news items` : 'Analyzing...'}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">News Volume</h3>
            <Bell className="w-5 h-5 text-yellow-400" />
          </div>
          <div className="text-2xl font-bold">
            {marketContext ? marketContext.news_volume : 0}
          </div>
          <div className="text-sm text-gray-400">Last 24 hours</div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Social Momentum</h3>
            <Twitter className="w-5 h-5 text-blue-400" />
          </div>
          <div className="text-2xl font-bold">
            {marketContext ? `${(marketContext.social_momentum * 100).toFixed(0)}%` : '0%'}
          </div>
          <div className="text-sm text-gray-400">Twitter activity</div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Connection</h3>
            <Activity className="w-5 h-5 text-green-400" />
          </div>
          <div className="text-2xl font-bold">
            <StatusIndicator 
              status={wsConnected ? 'active' : 'error'} 
              label={wsConnected ? 'Live' : 'Offline'} 
            />
          </div>
          <div className="text-sm text-gray-400">News feed status</div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-3 gap-6">
        
        {/* Market Entropy Monitor (existing) */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Market Entropy</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={entropyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Area 
                type="monotone" 
                dataKey="value" 
                stroke="#3B82F6" 
                fill="#3B82F6" 
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
          <div className="mt-2 text-sm text-gray-400">
            Current: 0.67 | Threshold: 0.70
          </div>
        </div>

        {/* Sentiment History Chart */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Sentiment Trends</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={sentimentHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="timestamp" 
                stroke="#9CA3AF"
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis stroke="#9CA3AF" domain={[-1, 1]} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
                labelFormatter={(value) => new Date(value).toLocaleString()}
              />
              <Line 
                type="monotone" 
                dataKey="sentiment" 
                stroke="#10B981" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Key Events */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Key Market Events</h3>
          <div className="space-y-3 max-h-48 overflow-y-auto">
            {marketContext?.key_events?.map((event, index) => (
              <div key={index} className="p-3 bg-gray-700 rounded">
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400 mt-1 flex-shrink-0" />
                  <span className="text-sm">{event}</span>
                </div>
              </div>
            )) || (
              <div className="text-gray-400 text-sm">No significant events detected</div>
            )}
          </div>
        </div>

        {/* Live News Feed */}
        <div className="col-span-2 bg-gray-800 rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Live News Feed</h3>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm text-gray-400">
                {wsConnected ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {newsItems.map((item, index) => (
              <div key={item.hash_key || index} className="p-4 bg-gray-700 rounded-lg">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-400">{item.source}</span>
                    <SentimentBadge sentiment={item.sentiment_label} score={item.sentiment_score} />
                  </div>
                  <span className="text-xs text-gray-400">
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <h4 className="font-medium mb-2 line-clamp-2">{item.headline}</h4>
                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-1">
                    {item.keywords_matched.slice(0, 3).map((keyword, idx) => (
                      <span key={idx} className="px-2 py-1 text-xs bg-blue-600 rounded">
                        {keyword}
                      </span>
                    ))}
                  </div>
                  <div className="text-xs text-gray-400">
                    Relevance: {(item.relevance_score * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            ))}
            {newsItems.length === 0 && (
              <div className="text-center text-gray-400 py-8">
                {isLoading ? 'Loading news...' : 'No news items available'}
              </div>
            )}
          </div>
        </div>

        {/* System Health (existing) */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">System Health</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">CPU Usage</span>
                <span className="text-sm">67%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '67%' }}></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Memory</span>
                <span className="text-sm">45%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '45%' }}></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Network I/O</span>
                <span className="text-sm">23%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '23%' }}></div>
              </div>
            </div>

            <div className="pt-4 space-y-2">
              <StatusIndicator status="active" label="News Engine" />
              <StatusIndicator status="active" label="Sentiment Analysis" />
              <StatusIndicator status={wsConnected ? "active" : "warning"} label="Data Feed" />
              <StatusIndicator status="active" label="Memory Integration" />
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 text-center text-gray-400 text-sm">
        Schwabot v2.1.0 + News Intelligence | Last Updated: {currentTime.toLocaleDateString()} | 
        Uptime: 47d 8h 23m | Processing: 847 ticks/sec | News Sources: {newsConfig.twitterEnabled ? 'Twitter' : ''} {newsConfig.googleNewsEnabled ? 'Google' : ''} {newsConfig.yahooNewsEnabled ? 'Yahoo' : ''}
      </div>
    </div>
  );
};

export default NewsIntegrationDashboard; 