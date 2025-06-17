import React, { useState, useEffect } from 'react';
import { Settings, Shield, Zap, Brain, AlertTriangle, CheckCircle, RefreshCw, Key, Globe, Twitter, DollarSign, Hash, Target } from 'lucide-react';

interface APICredentials {
  twitter_bearer_token: string;
  twitter_api_key: string;
  twitter_api_secret: string;
  newsapi_key: string;
  polygon_api_key: string;
}

interface BTCProcessorSettings {
  mining_analysis: boolean;
  block_timing: boolean;
  nonce_sequences: boolean;
  difficulty_tracking: boolean;
  hash_generation: boolean;
  load_balancing: boolean;
  memory_management: boolean;
  storage: boolean;
  monitoring: boolean;
  max_memory_usage_gb: number;
  max_cpu_usage_percent: number;
  max_gpu_usage_percent: number;
}

interface LanternNewsSettings {
  max_backlog_size: number;
  correlation_threshold: number;
  processing_batch_size: number;
  lexicon_update_interval: number;
  hash_correlation_window: number;
  profit_weight_factor: number;
  entropy_threshold: number;
  auto_throttle_enabled: boolean;
}

interface NewsSourceSettings {
  google_news_enabled: boolean;
  yahoo_finance_enabled: boolean;
  twitter_enabled: boolean;
  monitoring_interval: number;
  sentiment_threshold: number;
}

interface RateLimitStatus {
  google_news: { calls: number; limit: number; reset_time: string };
  yahoo_finance: { calls: number; limit: number; reset_time: string };
  twitter: { calls: number; limit: number; reset_time: string };
}

const LanternNewsSettingsPanel: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState('api-keys');
  const [isLoading, setIsLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  
  // Settings state
  const [apiCredentials, setApiCredentials] = useState<APICredentials>({
    twitter_bearer_token: '',
    twitter_api_key: '',
    twitter_api_secret: '',
    newsapi_key: '',
    polygon_api_key: ''
  });
  
  const [btcProcessorSettings, setBTCProcessorSettings] = useState<BTCProcessorSettings>({
    mining_analysis: true,
    block_timing: true,
    nonce_sequences: false,
    difficulty_tracking: true,
    hash_generation: true,
    load_balancing: true,
    memory_management: true,
    storage: false,
    monitoring: true,
    max_memory_usage_gb: 8,
    max_cpu_usage_percent: 70,
    max_gpu_usage_percent: 80
  });
  
  const [lanternSettings, setLanternSettings] = useState<LanternNewsSettings>({
    max_backlog_size: 500,
    correlation_threshold: 0.3,
    processing_batch_size: 10,
    lexicon_update_interval: 300,
    hash_correlation_window: 3600,
    profit_weight_factor: 0.7,
    entropy_threshold: 0.6,
    auto_throttle_enabled: true
  });
  
  const [newsSourceSettings, setNewsSourceSettings] = useState<NewsSourceSettings>({
    google_news_enabled: true,
    yahoo_finance_enabled: true,
    twitter_enabled: false,
    monitoring_interval: 15,
    sentiment_threshold: 0.5
  });
  
  const [rateLimitStatus, setRateLimitStatus] = useState<RateLimitStatus | null>(null);
  const [systemStatus, setSystemStatus] = useState<any>(null);

  // Load initial settings
  useEffect(() => {
    loadSettings();
    loadRateLimitStatus();
    loadSystemStatus();
  }, []);

  const loadSettings = async () => {
    try {
      // Load API credentials (masked for security)
      const apiResponse = await fetch('/api/settings/credentials');
      if (apiResponse.ok) {
        const apiData = await apiResponse.json();
        setApiCredentials(prev => ({
          ...prev,
          ...apiData.masked_credentials
        }));
      }

      // Load BTC processor settings
      const btcResponse = await fetch('/api/btc-processor/config');
      if (btcResponse.ok) {
        const btcData = await btcResponse.json();
        setBTCProcessorSettings(btcData);
      }

      // Load Lantern settings
      const lanternResponse = await fetch('/api/lantern-news/config');
      if (lanternResponse.ok) {
        const lanternData = await lanternResponse.json();
        setLanternSettings(lanternData);
      }

      // Load news source settings
      const newsResponse = await fetch('/api/news/config');
      if (newsResponse.ok) {
        const newsData = await newsResponse.json();
        setNewsSourceSettings(newsData);
      }

    } catch (error) {
      console.error('Error loading settings:', error);
    }
  };

  const loadRateLimitStatus = async () => {
    try {
      const response = await fetch('/api/news/rate-limits');
      if (response.ok) {
        const data = await response.json();
        setRateLimitStatus(data);
      }
    } catch (error) {
      console.error('Error loading rate limit status:', error);
    }
  };

  const loadSystemStatus = async () => {
    try {
      const response = await fetch('/api/lantern-news/status');
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data);
      }
    } catch (error) {
      console.error('Error loading system status:', error);
    }
  };

  const saveAPICredentials = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/settings/credentials', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiCredentials)
      });

      if (response.ok) {
        setSaveStatus('API credentials saved successfully');
        // Test connections
        await testAPIConnections();
      } else {
        setSaveStatus('Error saving API credentials');
      }
    } catch (error) {
      setSaveStatus('Error saving API credentials');
    } finally {
      setIsLoading(false);
    }
  };

  const testAPIConnections = async () => {
    try {
      const response = await fetch('/api/news/test-connections', {
        method: 'POST'
      });
      
      if (response.ok) {
        const results = await response.json();
        console.log('API connection test results:', results);
      }
    } catch (error) {
      console.error('Error testing API connections:', error);
    }
  };

  const saveBTCProcessorSettings = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/btc-processor/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(btcProcessorSettings)
      });

      if (response.ok) {
        setSaveStatus('BTC Processor settings saved successfully');
      } else {
        setSaveStatus('Error saving BTC Processor settings');
      }
    } catch (error) {
      setSaveStatus('Error saving BTC Processor settings');
    } finally {
      setIsLoading(false);
    }
  };

  const saveLanternSettings = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/lantern-news/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(lanternSettings)
      });

      if (response.ok) {
        setSaveStatus('Lantern News settings saved successfully');
      } else {
        setSaveStatus('Error saving Lantern News settings');
      }
    } catch (error) {
      setSaveStatus('Error saving Lantern News settings');
    } finally {
      setIsLoading(false);
    }
  };

  const saveNewsSourceSettings = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/news/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newsSourceSettings)
      });

      if (response.ok) {
        setSaveStatus('News source settings saved successfully');
      } else {
        setSaveStatus('Error saving news source settings');
      }
    } catch (error) {
      setSaveStatus('Error saving news source settings');
    } finally {
      setIsLoading(false);
    }
  };

  const emergencyCleanup = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/btc-processor/emergency-cleanup', {
        method: 'POST'
      });

      if (response.ok) {
        setSaveStatus('Emergency cleanup completed');
        loadSystemStatus(); // Refresh status
      } else {
        setSaveStatus('Error during emergency cleanup');
      }
    } catch (error) {
      setSaveStatus('Error during emergency cleanup');
    } finally {
      setIsLoading(false);
    }
  };

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
        default: return <RefreshCw className="w-4 h-4" />;
      }
    };

    return (
      <div className={`flex items-center space-x-2 ${getColor()}`}>
        {getIcon()}
        <span className="text-sm font-medium">{label}</span>
      </div>
    );
  };

  const RateLimitDisplay: React.FC<{ service: string; data: any }> = ({ service, data }) => {
    const usagePercent = (data.calls / data.limit) * 100;
    const getBarColor = () => {
      if (usagePercent > 80) return 'bg-red-500';
      if (usagePercent > 60) return 'bg-yellow-500';
      return 'bg-green-500';
    };

    return (
      <div className="p-3 bg-gray-700 rounded">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium capitalize">{service}</span>
          <span className="text-xs text-gray-400">{data.calls}/{data.limit}</span>
        </div>
        <div className="w-full bg-gray-600 rounded-full h-2">
          <div 
            className={`h-2 rounded-full ${getBarColor()}`}
            style={{ width: `${Math.min(usagePercent, 100)}%` }}
          ></div>
        </div>
        <div className="text-xs text-gray-400 mt-1">
          Resets: {new Date(data.reset_time).toLocaleTimeString()}
        </div>
      </div>
    );
  };

  const tabs = [
    { id: 'api-keys', label: 'API Keys', icon: Key },
    { id: 'news-sources', label: 'News Sources', icon: Globe },
    { id: 'btc-processor', label: 'BTC Processor', icon: Zap },
    { id: 'lantern-core', label: 'Lantern Core', icon: Brain },
    { id: 'system-status', label: 'System Status', icon: Shield }
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-6 max-w-6xl mx-auto">
      <div className="flex items-center space-x-2 mb-6">
        <Settings className="w-6 h-6 text-blue-400" />
        <h2 className="text-2xl font-bold text-white">Lantern News Intelligence Settings</h2>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-6 bg-gray-700 p-1 rounded-lg">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:text-white hover:bg-gray-600'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Save Status */}
      {saveStatus && (
        <div className={`mb-4 p-3 rounded ${
          saveStatus.includes('Error') ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
        }`}>
          {saveStatus}
        </div>
      )}

      {/* Tab Content */}
      <div className="space-y-6">
        
        {/* API Keys Tab */}
        {activeTab === 'api-keys' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-white">API Credentials</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Twitter API */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Twitter className="w-5 h-5 text-blue-400" />
                  <h4 className="font-medium text-white">Twitter API</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Bearer Token</label>
                    <input
                      type="password"
                      value={apiCredentials.twitter_bearer_token}
                      onChange={(e) => setApiCredentials({...apiCredentials, twitter_bearer_token: e.target.value})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                      placeholder="Your Twitter Bearer Token"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">API Key</label>
                    <input
                      type="password"
                      value={apiCredentials.twitter_api_key}
                      onChange={(e) => setApiCredentials({...apiCredentials, twitter_api_key: e.target.value})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                      placeholder="Your Twitter API Key"
                    />
                  </div>
                </div>
              </div>

              {/* News API */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Globe className="w-5 h-5 text-green-400" />
                  <h4 className="font-medium text-white">News APIs</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">NewsAPI Key (Optional)</label>
                    <input
                      type="password"
                      value={apiCredentials.newsapi_key}
                      onChange={(e) => setApiCredentials({...apiCredentials, newsapi_key: e.target.value})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                      placeholder="NewsAPI Key for Enhanced Features"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Polygon API Key (Optional)</label>
                    <input
                      type="password"
                      value={apiCredentials.polygon_api_key}
                      onChange={(e) => setApiCredentials({...apiCredentials, polygon_api_key: e.target.value})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                      placeholder="Polygon API Key for Market Data"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={saveAPICredentials}
                disabled={isLoading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded disabled:opacity-50 flex items-center space-x-2"
              >
                <Key className="w-4 h-4" />
                <span>{isLoading ? 'Saving...' : 'Save & Test Credentials'}</span>
              </button>
            </div>
          </div>
        )}

        {/* News Sources Tab */}
        {activeTab === 'news-sources' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-white">News Source Configuration</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Source Toggles */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Active Sources</h4>
                <div className="space-y-3">
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={newsSourceSettings.google_news_enabled}
                      onChange={(e) => setNewsSourceSettings({...newsSourceSettings, google_news_enabled: e.target.checked})}
                      className="w-4 h-4 text-blue-600"
                    />
                    <span className="text-white">Google News RSS (Free)</span>
                  </label>
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={newsSourceSettings.yahoo_finance_enabled}
                      onChange={(e) => setNewsSourceSettings({...newsSourceSettings, yahoo_finance_enabled: e.target.checked})}
                      className="w-4 h-4 text-blue-600"
                    />
                    <span className="text-white">Yahoo Finance RSS (Free)</span>
                  </label>
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={newsSourceSettings.twitter_enabled}
                      onChange={(e) => setNewsSourceSettings({...newsSourceSettings, twitter_enabled: e.target.checked})}
                      className="w-4 h-4 text-blue-600"
                    />
                    <span className="text-white">Twitter API (15 calls/day)</span>
                  </label>
                </div>
              </div>

              {/* Monitoring Settings */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Monitoring Settings</h4>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Monitoring Interval (minutes)
                    </label>
                    <input
                      type="number"
                      min="5"
                      max="60"
                      value={newsSourceSettings.monitoring_interval}
                      onChange={(e) => setNewsSourceSettings({...newsSourceSettings, monitoring_interval: parseInt(e.target.value)})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Sentiment Alert Threshold
                    </label>
                    <input
                      type="number"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={newsSourceSettings.sentiment_threshold}
                      onChange={(e) => setNewsSourceSettings({...newsSourceSettings, sentiment_threshold: parseFloat(e.target.value)})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Rate Limits Display */}
            {rateLimitStatus && (
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Current Rate Limits</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <RateLimitDisplay service="google_news" data={rateLimitStatus.google_news} />
                  <RateLimitDisplay service="yahoo_finance" data={rateLimitStatus.yahoo_finance} />
                  <RateLimitDisplay service="twitter" data={rateLimitStatus.twitter} />
                </div>
              </div>
            )}

            <button
              onClick={saveNewsSourceSettings}
              disabled={isLoading}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded disabled:opacity-50 flex items-center space-x-2"
            >
              <Globe className="w-4 h-4" />
              <span>{isLoading ? 'Saving...' : 'Save News Source Settings'}</span>
            </button>
          </div>
        )}

        {/* BTC Processor Tab */}
        {activeTab === 'btc-processor' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-white">BTC Processor Controls</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Feature Toggles */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Processing Features</h4>
                <div className="space-y-3">
                  {Object.entries(btcProcessorSettings)
                    .filter(([key]) => typeof btcProcessorSettings[key as keyof BTCProcessorSettings] === 'boolean')
                    .map(([key, value]) => (
                    <label key={key} className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        checked={value as boolean}
                        onChange={(e) => setBTCProcessorSettings({
                          ...btcProcessorSettings,
                          [key]: e.target.checked
                        })}
                        className="w-4 h-4 text-blue-600"
                      />
                      <span className="text-white capitalize">
                        {key.replace(/_/g, ' ')}
                      </span>
                      {key === 'mining_analysis' && (
                        <span className="text-xs text-yellow-400">(Memory Intensive)</span>
                      )}
                    </label>
                  ))}
                </div>
              </div>

              {/* Resource Limits */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Resource Limits</h4>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Max Memory Usage (GB)
                    </label>
                    <input
                      type="number"
                      min="4"
                      max="32"
                      value={btcProcessorSettings.max_memory_usage_gb}
                      onChange={(e) => setBTCProcessorSettings({
                        ...btcProcessorSettings,
                        max_memory_usage_gb: parseInt(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Max CPU Usage (%)
                    </label>
                    <input
                      type="number"
                      min="50"
                      max="95"
                      value={btcProcessorSettings.max_cpu_usage_percent}
                      onChange={(e) => setBTCProcessorSettings({
                        ...btcProcessorSettings,
                        max_cpu_usage_percent: parseInt(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Max GPU Usage (%)
                    </label>
                    <input
                      type="number"
                      min="50"
                      max="95"
                      value={btcProcessorSettings.max_gpu_usage_percent}
                      onChange={(e) => setBTCProcessorSettings({
                        ...btcProcessorSettings,
                        max_gpu_usage_percent: parseInt(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={saveBTCProcessorSettings}
                disabled={isLoading}
                className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded disabled:opacity-50 flex items-center space-x-2"
              >
                <Zap className="w-4 h-4" />
                <span>{isLoading ? 'Saving...' : 'Save BTC Processor Settings'}</span>
              </button>
              <button
                onClick={emergencyCleanup}
                disabled={isLoading}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded disabled:opacity-50 flex items-center space-x-2"
              >
                <AlertTriangle className="w-4 h-4" />
                <span>{isLoading ? 'Cleaning...' : 'Emergency Cleanup'}</span>
              </button>
            </div>
          </div>
        )}

        {/* Lantern Core Tab */}
        {activeTab === 'lantern-core' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-white">Lantern Core Mathematical Integration</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Hash Correlation Settings */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Hash className="w-5 h-5 text-purple-400" />
                  <h4 className="font-medium text-white">Hash Correlation</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Correlation Threshold
                    </label>
                    <input
                      type="number"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={lanternSettings.correlation_threshold}
                      onChange={(e) => setLanternSettings({
                        ...lanternSettings,
                        correlation_threshold: parseFloat(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Hash Correlation Window (seconds)
                    </label>
                    <input
                      type="number"
                      min="1800"
                      max="7200"
                      step="300"
                      value={lanternSettings.hash_correlation_window}
                      onChange={(e) => setLanternSettings({
                        ...lanternSettings,
                        hash_correlation_window: parseInt(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>

              {/* Profit Allocation Settings */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Target className="w-5 h-5 text-green-400" />
                  <h4 className="font-medium text-white">Profit Optimization</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Profit Weight Factor
                    </label>
                    <input
                      type="number"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={lanternSettings.profit_weight_factor}
                      onChange={(e) => setLanternSettings({
                        ...lanternSettings,
                        profit_weight_factor: parseFloat(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Entropy Threshold
                    </label>
                    <input
                      type="number"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={lanternSettings.entropy_threshold}
                      onChange={(e) => setLanternSettings({
                        ...lanternSettings,
                        entropy_threshold: parseFloat(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>

              {/* Processing Management */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Brain className="w-5 h-5 text-blue-400" />
                  <h4 className="font-medium text-white">Processing Management</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Max Backlog Size
                    </label>
                    <input
                      type="number"
                      min="100"
                      max="2000"
                      step="50"
                      value={lanternSettings.max_backlog_size}
                      onChange={(e) => setLanternSettings({
                        ...lanternSettings,
                        max_backlog_size: parseInt(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Processing Batch Size
                    </label>
                    <input
                      type="number"
                      min="5"
                      max="50"
                      value={lanternSettings.processing_batch_size}
                      onChange={(e) => setLanternSettings({
                        ...lanternSettings,
                        processing_batch_size: parseInt(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <label className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={lanternSettings.auto_throttle_enabled}
                      onChange={(e) => setLanternSettings({
                        ...lanternSettings,
                        auto_throttle_enabled: e.target.checked
                      })}
                      className="w-4 h-4 text-blue-600"
                    />
                    <span className="text-white">Auto-Throttle Processing</span>
                  </label>
                </div>
              </div>

              {/* Lexicon Update Settings */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <RefreshCw className="w-5 h-5 text-orange-400" />
                  <h4 className="font-medium text-white">Lexicon Updates</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Update Interval (seconds)
                    </label>
                    <input
                      type="number"
                      min="60"
                      max="3600"
                      step="60"
                      value={lanternSettings.lexicon_update_interval}
                      onChange={(e) => setLanternSettings({
                        ...lanternSettings,
                        lexicon_update_interval: parseInt(e.target.value)
                      })}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>

            <button
              onClick={saveLanternSettings}
              disabled={isLoading}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded disabled:opacity-50 flex items-center space-x-2"
            >
              <Brain className="w-4 h-4" />
              <span>{isLoading ? 'Saving...' : 'Save Lantern Core Settings'}</span>
            </button>
          </div>
        )}

        {/* System Status Tab */}
        {activeTab === 'system-status' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-white">System Status & Health</h3>
            
            {systemStatus && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Processing Status */}
                <div className="bg-gray-700 p-4 rounded-lg">
                  <h4 className="font-medium text-white mb-4">Processing Status</h4>
                  <div className="space-y-3">
                    <StatusIndicator 
                      status={systemStatus.processing_active ? 'active' : 'error'} 
                      label={`Processing: ${systemStatus.processing_active ? 'Active' : 'Inactive'}`} 
                    />
                    <div className="text-sm text-gray-300">
                      <div>Backlog Size: {systemStatus.backlog_size}</div>
                      <div>Correlation Cache: {systemStatus.correlation_cache_size}</div>
                      <div>Last Update: {new Date(systemStatus.last_update).toLocaleString()}</div>
                    </div>
                  </div>
                </div>

                {/* Metrics */}
                <div className="bg-gray-700 p-4 rounded-lg">
                  <h4 className="font-medium text-white mb-4">Processing Metrics</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <div>News Processed: {systemStatus.metrics?.news_processed || 0}</div>
                    <div>Correlations Found: {systemStatus.metrics?.correlations_found || 0}</div>
                    <div>Profit Updates: {systemStatus.metrics?.profit_updates || 0}</div>
                    <div>Backlog Overflows: {systemStatus.metrics?.backlog_overflows || 0}</div>
                  </div>
                </div>

                {/* Lexicon Stats */}
                {systemStatus.lexicon_stats && (
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h4 className="font-medium text-white mb-4">Lexicon Statistics</h4>
                    <div className="space-y-2 text-sm text-gray-300">
                      <div>Total Words: {systemStatus.lexicon_stats.total_words || 0}</div>
                      <div>Avg Profit Score: {systemStatus.lexicon_stats.avg_profit_score?.toFixed(3) || 0}</div>
                      <div>High Profit Words: {systemStatus.lexicon_stats.high_profit_words || 0}</div>
                    </div>
                  </div>
                )}

                {/* Configuration Summary */}
                <div className="bg-gray-700 p-4 rounded-lg">
                  <h4 className="font-medium text-white mb-4">Current Configuration</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <div>Batch Size: {systemStatus.config?.processing_batch_size}</div>
                    <div>Correlation Threshold: {systemStatus.config?.correlation_threshold}</div>
                    <div>Auto-Throttle: {systemStatus.config?.auto_throttle_enabled ? 'Enabled' : 'Disabled'}</div>
                  </div>
                </div>
              </div>
            )}

            <div className="flex space-x-3">
              <button
                onClick={loadSystemStatus}
                disabled={isLoading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded disabled:opacity-50 flex items-center space-x-2"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Refresh Status</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LanternNewsSettingsPanel; 