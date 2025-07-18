import React, { useState, useEffect } from 'react';
import { 
  Hash, DollarSign, Zap, Brain, AlertTriangle, CheckCircle, RefreshCw, 
  Settings, Target, Clock, TrendingUp, Activity, BarChart3, Eye,
  Play, Pause, Square, TestTube, Shield
} from 'lucide-react';

interface NewsProfitConfig {
  correlation_threshold: number;
  hash_window_minutes: number;
  profit_crystallization_threshold: number;
  max_keywords_per_event: number;
  entropy_classes: number;
  base_entry_delay_minutes: number;
  base_exit_window_minutes: number;
  min_confidence_threshold: number;
  default_position_size: number;
  max_position_size: number;
  default_stop_loss: number;
  default_take_profit: number;
  // Advanced allocation settings
  cpu_allocation_percentage: number;
  gpu_allocation_percentage: number;
  processing_mode: string;
  thermal_scaling_enabled: boolean;
  dynamic_allocation_enabled: boolean;
  thermal_cpu_threshold: number;
  thermal_gpu_threshold: number;
  thermal_emergency_threshold: number;
}

interface SystemStatus {
  processed_events: number;
  profitable_correlations: number;
  successful_trades: number;
  active_signatures: number;
  active_correlations: number;
  active_profit_timings: number;
  runtime_info: {
    bridge_initialized: boolean;
    profit_navigator_active: boolean;
    btc_controller_active: boolean;
    fractal_controller_active: boolean;
  };
}

interface ProfitTiming {
  signature_hash: string;
  entry_time: string;
  exit_time: string;
  confidence: number;
  profit_expectation: number;
  risk_factor: number;
  is_active: boolean;
  time_remaining: number;
}

interface CorrelationData {
  [signature: string]: number;
}

interface Analytics {
  processing_metrics: {
    total_events_processed: number;
    profitable_correlations: number;
    successful_trades: number;
    success_rate: number;
    correlation_rate: number;
  };
  correlation_analysis: {
    active_correlations: number;
    average_correlation: number;
    correlation_distribution: {
      ranges: { [key: string]: number };
      total: number;
    };
  };
  timing_analysis: {
    active_timings: number;
    average_confidence: number;
    profit_expectation_stats: {
      average: number;
      max: number;
      min: number;
    };
  };
}

interface ThermalState {
  thermal_available: boolean;
  cpu_temp: number;
  gpu_temp: number;
  thermal_zone: string;
  cpu_load: number;
  gpu_load: number;
  drift_coefficient?: number;
  thermal_scaling_active?: boolean;
}

interface AllocationInfo {
  current_allocation: {
    cpu_percentage: number;
    gpu_percentage: number;
    processing_mode: string;
  };
  dynamic_allocation: [number, number];
  allocation_history: any[];
  processing_modes: string[];
  allocation_switches: number;
  thermal_management: {
    thermal_scaling_enabled: boolean;
    dynamic_allocation_enabled: boolean;
  };
}

const NewsProfitSettingsPanel: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState('configuration');
  const [isLoading, setIsLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Enhanced state for advanced settings
  const [thermalState, setThermalState] = useState<ThermalState | null>(null);
  const [allocationInfo, setAllocationInfo] = useState<AllocationInfo | null>(null);
  const [processingLoad, setProcessingLoad] = useState<any>(null);
  const [advancedMode, setAdvancedMode] = useState(false);

  // Configuration state
  const [config, setConfig] = useState<NewsProfitConfig>({
    correlation_threshold: 0.25,
    hash_window_minutes: 60,
    profit_crystallization_threshold: 0.15,
    max_keywords_per_event: 10,
    entropy_classes: 4,
    base_entry_delay_minutes: 10,
    base_exit_window_minutes: 30,
    min_confidence_threshold: 0.3,
    default_position_size: 0.1,
    max_position_size: 0.3,
    default_stop_loss: 0.02,
    default_take_profit: 0.06,
    // Advanced settings
    cpu_allocation_percentage: 70.0,
    gpu_allocation_percentage: 30.0,
    processing_mode: "hybrid",
    thermal_scaling_enabled: true,
    dynamic_allocation_enabled: true,
    thermal_cpu_threshold: 75.0,
    thermal_gpu_threshold: 70.0,
    thermal_emergency_threshold: 85.0
  });

  // System state
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [profitTimings, setProfitTimings] = useState<ProfitTiming[]>([]);
  const [correlations, setCorrelations] = useState<CorrelationData>({});
  const [analytics, setAnalytics] = useState<Analytics | null>(null);

  // Load initial data
  useEffect(() => {
    loadConfiguration();
    loadSystemStatus();
    loadProfitTimings();
    loadCorrelations();
    loadAnalytics();
    loadThermalStatus();
    loadAllocationInfo();
    loadProcessingLoad();

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      if (activeTab === 'monitoring') {
        loadSystemStatus();
        loadProfitTimings();
        loadCorrelations();
        loadThermalStatus();
        loadProcessingLoad();
      } else if (activeTab === 'advanced') {
        loadThermalStatus();
        loadAllocationInfo();
        loadProcessingLoad();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [activeTab]);

  const loadConfiguration = async () => {
    try {
      const response = await fetch('/api/news-profit/config');
      if (response.ok) {
        const data = await response.json();
        setConfig(data.config);
      }
    } catch (error) {
      console.error('Error loading configuration:', error);
    }
  };

  const saveConfiguration = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/news-profit/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        setSaveStatus('Configuration saved successfully');
      } else {
        setSaveStatus('Error saving configuration');
      }
    } catch (error) {
      setSaveStatus('Error saving configuration');
    } finally {
      setIsLoading(false);
    }
  };

  const loadSystemStatus = async () => {
    try {
      const response = await fetch('/api/news-profit/status');
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data.status);
      }
    } catch (error) {
      console.error('Error loading system status:', error);
    }
  };

  const loadProfitTimings = async () => {
    try {
      const response = await fetch('/api/news-profit/timings?active_only=true');
      if (response.ok) {
        const data = await response.json();
        setProfitTimings(data.timings);
      }
    } catch (error) {
      console.error('Error loading profit timings:', error);
    }
  };

  const loadCorrelations = async () => {
    try {
      const response = await fetch('/api/news-profit/correlations?min_correlation=0.2&limit=20');
      if (response.ok) {
        const data = await response.json();
        setCorrelations(data.correlations);
      }
    } catch (error) {
      console.error('Error loading correlations:', error);
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await fetch('/api/news-profit/analytics');
      if (response.ok) {
        const data = await response.json();
        setAnalytics(data.analytics);
      }
    } catch (error) {
      console.error('Error loading analytics:', error);
    }
  };

  const loadThermalStatus = async () => {
    try {
      const response = await fetch('/api/news-profit/thermal');
      if (response.ok) {
        const data = await response.json();
        setThermalState(data.thermal_status.current_state);
      }
    } catch (error) {
      console.error('Error loading thermal status:', error);
    }
  };

  const loadAllocationInfo = async () => {
    try {
      const response = await fetch('/api/news-profit/allocation');
      if (response.ok) {
        const data = await response.json();
        setAllocationInfo(data.allocation_info);
        
        // Update config with current allocation
        setConfig(prev => ({
          ...prev,
          cpu_allocation_percentage: data.allocation_info.current_allocation.cpu_percentage,
          gpu_allocation_percentage: data.allocation_info.current_allocation.gpu_percentage,
          processing_mode: data.allocation_info.current_allocation.processing_mode,
          thermal_scaling_enabled: data.allocation_info.thermal_management.thermal_scaling_enabled,
          dynamic_allocation_enabled: data.allocation_info.thermal_management.dynamic_allocation_enabled
        }));
      }
    } catch (error) {
      console.error('Error loading allocation info:', error);
    }
  };

  const loadProcessingLoad = async () => {
    try {
      const response = await fetch('/api/news-profit/processing-load');
      if (response.ok) {
        const data = await response.json();
        setProcessingLoad(data.processing_load);
      }
    } catch (error) {
      console.error('Error loading processing load:', error);
    }
  };

  const updateAllocationSettings = async (settings: any) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/news-profit/allocation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });

      if (response.ok) {
        setSaveStatus('Allocation settings updated successfully');
        await loadAllocationInfo();
      } else {
        setSaveStatus('Error updating allocation settings');
      }
    } catch (error) {
      setSaveStatus('Error updating allocation settings');
    } finally {
      setIsLoading(false);
    }
  };

  const testPipeline = async () => {
    setIsProcessing(true);
    try {
      const response = await fetch('/api/news-profit/test', {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        setSaveStatus(`Pipeline test completed: ${data.test_results.fact_events_extracted} events processed`);
        
        // Refresh data after test
        await loadSystemStatus();
        await loadProfitTimings();
        await loadCorrelations();
      } else {
        setSaveStatus('Pipeline test failed');
      }
    } catch (error) {
      setSaveStatus('Pipeline test error');
    } finally {
      setIsProcessing(false);
    }
  };

  const executeTrades = async (dryRun = true) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/news-profit/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          min_confidence: config.min_confidence_threshold,
          max_risk: 0.5,
          dry_run: dryRun
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSaveStatus(`${dryRun ? 'Simulated' : 'Executed'} ${data.execution_results.length} trades`);
      } else {
        setSaveStatus('Trade execution failed');
      }
    } catch (error) {
      setSaveStatus('Trade execution error');
    } finally {
      setIsLoading(false);
    }
  };

  const StatusIndicator: React.FC<{ active: boolean; label: string }> = ({ active, label }) => (
    <div className={`flex items-center space-x-2 ${active ? 'text-green-500' : 'text-red-500'}`}>
      {active ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
      <span className="text-sm font-medium">{label}</span>
    </div>
  );

  const MetricCard: React.FC<{ 
    title: string; 
    value: string | number; 
    icon: React.ReactNode; 
    color?: string 
  }> = ({ title, value, icon, color = 'text-blue-400' }) => (
    <div className="bg-gray-700 p-4 rounded-lg">
      <div className={`flex items-center space-x-2 ${color} mb-2`}>
        {icon}
        <span className="text-sm font-medium">{title}</span>
      </div>
      <div className="text-2xl font-bold text-white">{value}</div>
    </div>
  );

  const CorrelationBar: React.FC<{ signature: string; correlation: number }> = ({ signature, correlation }) => (
    <div className="bg-gray-700 p-3 rounded mb-2">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs text-gray-400 font-mono">{signature.substring(0, 16)}...</span>
        <span className="text-sm font-medium text-white">{(correlation * 100).toFixed(1)}%</span>
      </div>
      <div className="w-full bg-gray-600 rounded-full h-2">
        <div 
          className={`h-2 rounded-full ${correlation > 0.6 ? 'bg-green-500' : correlation > 0.3 ? 'bg-yellow-500' : 'bg-red-500'}`}
          style={{ width: `${Math.min(correlation * 100, 100)}%` }}
        ></div>
      </div>
    </div>
  );

  const ProfitTimingCard: React.FC<{ timing: ProfitTiming }> = ({ timing }) => (
    <div className="bg-gray-700 p-4 rounded-lg">
      <div className="flex justify-between items-start mb-2">
        <span className="text-xs text-gray-400 font-mono">{timing.signature_hash.substring(0, 12)}...</span>
        <span className={`px-2 py-1 rounded text-xs font-medium ${
          timing.is_active ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'
        }`}>
          {timing.is_active ? 'ACTIVE' : 'INACTIVE'}
        </span>
      </div>
      
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-gray-400">Confidence:</span>
          <span className="text-white ml-1">{(timing.confidence * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-400">Profit:</span>
          <span className="text-green-400 ml-1">{(timing.profit_expectation * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-400">Risk:</span>
          <span className="text-red-400 ml-1">{(timing.risk_factor * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-gray-400">Time Left:</span>
          <span className="text-white ml-1">{Math.max(0, Math.floor(timing.time_remaining / 60))}m</span>
        </div>
      </div>
    </div>
  );

  const ThermalIndicator: React.FC<{ state: ThermalState }> = ({ state }) => {
    if (!state.thermal_available) {
      return (
        <div className="flex items-center space-x-2 text-gray-400">
          <AlertTriangle className="w-4 h-4" />
          <span className="text-sm">Thermal monitoring unavailable</span>
        </div>
      );
    }

    const getThermalColor = (zone: string) => {
      switch (zone) {
        case 'cool': return 'text-blue-400';
        case 'normal': return 'text-green-400';
        case 'warm': return 'text-yellow-400';
        case 'hot': return 'text-orange-400';
        case 'critical': return 'text-red-400';
        default: return 'text-gray-400';
      }
    };

    return (
      <div className="space-y-2">
        <div className={`flex items-center space-x-2 ${getThermalColor(state.thermal_zone)}`}>
          <Activity className="w-4 h-4" />
          <span className="text-sm font-medium">Thermal Zone: {state.thermal_zone.toUpperCase()}</span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-gray-400">CPU:</span>
            <span className="text-white ml-1">{state.cpu_temp.toFixed(1)}°C</span>
          </div>
          <div>
            <span className="text-gray-400">GPU:</span>
            <span className="text-white ml-1">{state.gpu_temp.toFixed(1)}°C</span>
          </div>
          <div>
            <span className="text-gray-400">CPU Load:</span>
            <span className="text-white ml-1">{state.cpu_load.toFixed(1)}%</span>
          </div>
          <div>
            <span className="text-gray-400">GPU Load:</span>
            <span className="text-white ml-1">{state.gpu_load.toFixed(1)}%</span>
          </div>
        </div>
      </div>
    );
  };

  const ProcessingAllocationSlider: React.FC<{
    label: string;
    value: number;
    onChange: (value: number) => void;
    color?: string;
    disabled?: boolean;
  }> = ({ label, value, onChange, color = 'blue', disabled = false }) => (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-sm text-gray-300">{label}</label>
        <span className="text-sm font-medium text-white">{value.toFixed(1)}%</span>
      </div>
      <input
        type="range"
        min="5"
        max="95"
        step="5"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className={`w-full slider-${color} ${disabled ? 'opacity-50' : ''}`}
      />
    </div>
  );

  const tabs = [
    { id: 'configuration', label: 'Configuration', icon: Settings },
    { id: 'monitoring', label: 'Live Monitoring', icon: Activity },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'execution', label: 'Trade Execution', icon: Zap },
    { id: 'testing', label: 'Testing', icon: TestTube },
    { id: 'advanced', label: 'Advanced Settings', icon: Brain }
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-6 max-w-7xl mx-auto">
      <div className="flex items-center space-x-2 mb-6">
        <Hash className="w-6 h-6 text-purple-400" />
        <h2 className="text-2xl font-bold text-white">News-Profit Mathematical Bridge</h2>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-6 bg-gray-700 p-1 rounded-lg">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-purple-600 text-white'
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
          saveStatus.includes('Error') || saveStatus.includes('failed') 
            ? 'bg-red-500/20 text-red-400' 
            : 'bg-green-500/20 text-green-400'
        }`}>
          {saveStatus}
        </div>
      )}

      {/* Tab Content */}
      <div className="space-y-6">
        
        {/* Configuration Tab */}
        {activeTab === 'configuration' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-white">Mathematical Pipeline Configuration</h3>
            
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
                      Correlation Threshold ({(config.correlation_threshold * 100).toFixed(0)}%)
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.8"
                      step="0.05"
                      value={config.correlation_threshold}
                      onChange={(e) => setConfig({...config, correlation_threshold: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Hash Window (minutes)</label>
                    <input
                      type="number"
                      min="15"
                      max="240"
                      step="15"
                      value={config.hash_window_minutes}
                      onChange={(e) => setConfig({...config, hash_window_minutes: parseInt(e.target.value)})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Max Keywords per Event</label>
                    <input
                      type="number"
                      min="5"
                      max="20"
                      value={config.max_keywords_per_event}
                      onChange={(e) => setConfig({...config, max_keywords_per_event: parseInt(e.target.value)})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              </div>

              {/* Profit Timing Settings */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Clock className="w-5 h-5 text-green-400" />
                  <h4 className="font-medium text-white">Profit Timing</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Profit Threshold ({(config.profit_crystallization_threshold * 100).toFixed(1)}%)
                    </label>
                    <input
                      type="range"
                      min="0.05"
                      max="0.5"
                      step="0.01"
                      value={config.profit_crystallization_threshold}
                      onChange={(e) => setConfig({...config, profit_crystallization_threshold: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Entry Delay (minutes)</label>
                    <input
                      type="number"
                      min="1"
                      max="60"
                      value={config.base_entry_delay_minutes}
                      onChange={(e) => setConfig({...config, base_entry_delay_minutes: parseInt(e.target.value)})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Exit Window (minutes)</label>
                    <input
                      type="number"
                      min="10"
                      max="180"
                      value={config.base_exit_window_minutes}
                      onChange={(e) => setConfig({...config, base_exit_window_minutes: parseInt(e.target.value)})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              </div>

              {/* Risk Management */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Shield className="w-5 h-5 text-yellow-400" />
                  <h4 className="font-medium text-white">Risk Management</h4>
                </div>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Min Confidence ({(config.min_confidence_threshold * 100).toFixed(0)}%)
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={config.min_confidence_threshold}
                      onChange={(e) => setConfig({...config, min_confidence_threshold: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Default Position Size ({(config.default_position_size * 100).toFixed(0)}%)
                    </label>
                    <input
                      type="range"
                      min="0.01"
                      max="0.5"
                      step="0.01"
                      value={config.default_position_size}
                      onChange={(e) => setConfig({...config, default_position_size: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Stop Loss ({(config.default_stop_loss * 100).toFixed(1)}%)
                    </label>
                    <input
                      type="range"
                      min="0.005"
                      max="0.1"
                      step="0.005"
                      value={config.default_stop_loss}
                      onChange={(e) => setConfig({...config, default_stop_loss: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">
                      Take Profit ({(config.default_take_profit * 100).toFixed(1)}%)
                    </label>
                    <input
                      type="range"
                      min="0.01"
                      max="0.2"
                      step="0.01"
                      value={config.default_take_profit}
                      onChange={(e) => setConfig({...config, default_take_profit: parseFloat(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* System Integration */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Brain className="w-5 h-5 text-blue-400" />
                  <h4 className="font-medium text-white">System Integration</h4>
                </div>
                {systemStatus && (
                  <div className="space-y-2">
                    <StatusIndicator 
                      active={systemStatus.runtime_info.bridge_initialized}
                      label="Mathematical Bridge"
                    />
                    <StatusIndicator 
                      active={systemStatus.runtime_info.profit_navigator_active}
                      label="Profit Navigator"
                    />
                    <StatusIndicator 
                      active={systemStatus.runtime_info.btc_controller_active}
                      label="BTC Controller"
                    />
                    <StatusIndicator 
                      active={systemStatus.runtime_info.fractal_controller_active}
                      label="Fractal Controller"
                    />
                  </div>
                )}
              </div>
            </div>

            <button
              onClick={saveConfiguration}
              disabled={isLoading}
              className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium disabled:opacity-50 flex items-center space-x-2"
            >
              <Settings className="w-4 h-4" />
              <span>{isLoading ? 'Saving...' : 'Save Configuration'}</span>
            </button>
          </div>
        )}

        {/* Live Monitoring Tab */}
        {activeTab === 'monitoring' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Live System Monitoring</h3>
              <button
                onClick={() => {
                  loadSystemStatus();
                  loadProfitTimings();
                  loadCorrelations();
                }}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center space-x-2"
              >
                <RefreshCw className="w-4 h-4" />
                <span>Refresh</span>
              </button>
            </div>

            {/* System Metrics */}
            {systemStatus && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <MetricCard
                  title="Events Processed"
                  value={systemStatus.processed_events}
                  icon={<Activity className="w-5 h-5" />}
                  color="text-blue-400"
                />
                <MetricCard
                  title="Profitable Correlations"
                  value={systemStatus.profitable_correlations}
                  icon={<Hash className="w-5 h-5" />}
                  color="text-green-400"
                />
                <MetricCard
                  title="Successful Trades"
                  value={systemStatus.successful_trades}
                  icon={<DollarSign className="w-5 h-5" />}
                  color="text-purple-400"
                />
                <MetricCard
                  title="Active Timings"
                  value={systemStatus.active_profit_timings}
                  icon={<Clock className="w-5 h-5" />}
                  color="text-orange-400"
                />
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Active Correlations */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4 flex items-center space-x-2">
                  <Hash className="w-4 h-4 text-purple-400" />
                  <span>Top Hash Correlations</span>
                </h4>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {Object.entries(correlations).length > 0 ? (
                    Object.entries(correlations).map(([signature, correlation]) => (
                      <CorrelationBar key={signature} signature={signature} correlation={correlation} />
                    ))
                  ) : (
                    <div className="text-gray-400 text-center py-4">No active correlations</div>
                  )}
                </div>
              </div>

              {/* Active Profit Timings */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4 flex items-center space-x-2">
                  <Target className="w-4 h-4 text-green-400" />
                  <span>Active Profit Timings</span>
                </h4>
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {profitTimings.length > 0 ? (
                    profitTimings.map((timing) => (
                      <ProfitTimingCard key={timing.signature_hash} timing={timing} />
                    ))
                  ) : (
                    <div className="text-gray-400 text-center py-4">No active profit timings</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && analytics && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Performance Analytics</h3>
              <button
                onClick={loadAnalytics}
                className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center space-x-2"
              >
                <BarChart3 className="w-4 h-4" />
                <span>Refresh Analytics</span>
              </button>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <MetricCard
                title="Success Rate"
                value={`${(analytics.processing_metrics.success_rate * 100).toFixed(1)}%`}
                icon={<TrendingUp className="w-5 h-5" />}
                color="text-green-400"
              />
              <MetricCard
                title="Correlation Rate"
                value={`${(analytics.processing_metrics.correlation_rate * 100).toFixed(1)}%`}
                icon={<Hash className="w-5 h-5" />}
                color="text-purple-400"
              />
              <MetricCard
                title="Avg Confidence"
                value={`${(analytics.timing_analysis.average_confidence * 100).toFixed(1)}%`}
                icon={<Target className="w-5 h-5" />}
                color="text-blue-400"
              />
            </div>

            {/* Detailed Analytics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Correlation Distribution</h4>
                <div className="space-y-2">
                  {Object.entries(analytics.correlation_analysis.correlation_distribution.ranges).map(([range, count]) => (
                    <div key={range} className="flex justify-between items-center">
                      <span className="text-gray-300">{range}</span>
                      <span className="text-white font-medium">{count}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Profit Expectations</h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Average:</span>
                    <span className="text-green-400 font-medium">
                      {(analytics.timing_analysis.profit_expectation_stats.average * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Maximum:</span>
                    <span className="text-green-400 font-medium">
                      {(analytics.timing_analysis.profit_expectation_stats.max * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300">Minimum:</span>
                    <span className="text-green-400 font-medium">
                      {(analytics.timing_analysis.profit_expectation_stats.min * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Execution Tab */}
        {activeTab === 'execution' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-white">Trade Execution Control</h3>
            
            <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-lg p-4">
              <div className="flex items-center space-x-2 text-yellow-400 mb-2">
                <AlertTriangle className="w-5 h-5" />
                <span className="font-medium">Safety Notice</span>
              </div>
              <p className="text-yellow-300 text-sm">
                All executions are in DRY RUN mode by default. Real trading requires manual confirmation 
                and should only be enabled after thorough testing.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Execution Parameters</h4>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Minimum Confidence</label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={config.min_confidence_threshold}
                      className="w-full"
                      readOnly
                    />
                    <span className="text-xs text-gray-400">
                      {(config.min_confidence_threshold * 100).toFixed(0)}% confidence required
                    </span>
                  </div>
                  <div>
                    <label className="block text-sm text-gray-300 mb-1">Position Size</label>
                    <input
                      type="range"
                      min="0.01"
                      max="0.3"
                      step="0.01"
                      value={config.default_position_size}
                      className="w-full"
                      readOnly
                    />
                    <span className="text-xs text-gray-400">
                      {(config.default_position_size * 100).toFixed(0)}% of portfolio
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">Current Opportunities</h4>
                <div className="text-sm space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Valid Timings:</span>
                    <span className="text-white">{profitTimings.filter(t => t.is_active).length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">High Confidence:</span>
                    <span className="text-white">
                      {profitTimings.filter(t => t.is_active && t.confidence > 0.7).length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Expected Profit:</span>
                    <span className="text-green-400">
                      {profitTimings.length > 0 
                        ? (profitTimings.reduce((sum, t) => sum + t.profit_expectation, 0) / profitTimings.length * 100).toFixed(2)
                        : 0}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex space-x-4">
              <button
                onClick={() => executeTrades(true)}
                disabled={isLoading}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium disabled:opacity-50 flex items-center space-x-2"
              >
                <Eye className="w-4 h-4" />
                <span>{isLoading ? 'Simulating...' : 'Simulate Trades (Dry Run)'}</span>
              </button>
              
              <button
                onClick={() => executeTrades(false)}
                disabled={isLoading || profitTimings.filter(t => t.is_active).length === 0}
                className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium disabled:opacity-50 flex items-center space-x-2"
              >
                <Zap className="w-4 h-4" />
                <span>{isLoading ? 'Executing...' : 'Execute Real Trades'}</span>
              </button>
            </div>
          </div>
        )}

        {/* Testing Tab */}
        {activeTab === 'testing' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-white">Pipeline Testing</h3>
            
            <div className="bg-gray-700 p-4 rounded-lg">
              <h4 className="font-medium text-white mb-4">Test Mathematical Pipeline</h4>
              <p className="text-gray-300 text-sm mb-4">
                Run the complete news-to-profit pipeline with mock data to verify all components are working correctly.
                This tests: News Processing → Hash Correlation → Profit Timing → Execution Signals
              </p>
              
              <button
                onClick={testPipeline}
                disabled={isProcessing}
                className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium disabled:opacity-50 flex items-center space-x-2"
              >
                <TestTube className="w-4 h-4" />
                <span>{isProcessing ? 'Testing Pipeline...' : 'Run Pipeline Test'}</span>
              </button>
            </div>

            {systemStatus && (
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="font-medium text-white mb-4">System Health Check</h4>
                <div className="grid grid-cols-2 gap-4">
                  <StatusIndicator 
                    active={systemStatus.runtime_info.bridge_initialized}
                    label="Mathematical Bridge"
                  />
                  <StatusIndicator 
                    active={systemStatus.runtime_info.profit_navigator_active}
                    label="Profit Navigator"
                  />
                  <StatusIndicator 
                    active={systemStatus.runtime_info.btc_controller_active}
                    label="BTC Controller"
                  />
                  <StatusIndicator 
                    active={systemStatus.runtime_info.fractal_controller_active}
                    label="Fractal Controller"
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Advanced Settings Tab */}
        {activeTab === 'advanced' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Advanced Processing Controls</h3>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="advancedMode"
                  checked={advancedMode}
                  onChange={(e) => setAdvancedMode(e.target.checked)}
                  className="rounded"
                />
                <label htmlFor="advancedMode" className="text-sm text-gray-300">Expert Mode</label>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Processing Allocation Control */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Brain className="w-5 h-5 text-blue-400" />
                  <h4 className="font-medium text-white">Processing Allocation</h4>
                </div>
                
                <div className="space-y-4">
                  {/* CPU Allocation Slider */}
                  <ProcessingAllocationSlider
                    label="CPU Allocation"
                    value={config.cpu_allocation_percentage}
                    onChange={(value) => {
                      setConfig(prev => ({
                        ...prev,
                        cpu_allocation_percentage: value,
                        gpu_allocation_percentage: 100 - value
                      }));
                    }}
                    color="blue"
                  />
                  
                  {/* GPU Allocation Slider */}
                  <ProcessingAllocationSlider
                    label="GPU Allocation"
                    value={config.gpu_allocation_percentage}
                    onChange={(value) => {
                      setConfig(prev => ({
                        ...prev,
                        gpu_allocation_percentage: value,
                        cpu_allocation_percentage: 100 - value
                      }));
                    }}
                    color="green"
                  />

                  {/* Processing Mode Selection */}
                  <div className="space-y-2">
                    <label className="block text-sm text-gray-300">Processing Mode</label>
                    <select
                      value={config.processing_mode}
                      onChange={(e) => setConfig({...config, processing_mode: e.target.value})}
                      className="w-full p-2 bg-gray-600 text-white rounded focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="hybrid">Hybrid (Balanced)</option>
                      <option value="cpu_only">CPU Only</option>
                      <option value="gpu_preferred">GPU Preferred</option>
                      <option value="thermal_aware">Thermal Aware</option>
                    </select>
                  </div>

                  {/* Dynamic Allocation Controls */}
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="thermalScaling"
                        checked={config.thermal_scaling_enabled}
                        onChange={(e) => setConfig({...config, thermal_scaling_enabled: e.target.checked})}
                      />
                      <label htmlFor="thermalScaling" className="text-sm text-gray-300">
                        Thermal Scaling
                      </label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="dynamicAllocation"
                        checked={config.dynamic_allocation_enabled}
                        onChange={(e) => setConfig({...config, dynamic_allocation_enabled: e.target.checked})}
                      />
                      <label htmlFor="dynamicAllocation" className="text-sm text-gray-300">
                        Dynamic Allocation
                      </label>
                    </div>
                  </div>

                  {/* Apply Allocation Settings Button */}
                  <button
                    onClick={() => updateAllocationSettings({
                      cpu_allocation_percentage: config.cpu_allocation_percentage,
                      processing_mode: config.processing_mode,
                      thermal_scaling_enabled: config.thermal_scaling_enabled,
                      dynamic_allocation_enabled: config.dynamic_allocation_enabled
                    })}
                    disabled={isLoading}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium disabled:opacity-50"
                  >
                    {isLoading ? 'Applying...' : 'Apply Allocation Settings'}
                  </button>
                </div>
              </div>

              {/* Thermal Management */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Activity className="w-5 h-5 text-orange-400" />
                  <h4 className="font-medium text-white">Thermal Management</h4>
                </div>
                
                {thermalState && <ThermalIndicator state={thermalState} />}
                
                {advancedMode && (
                  <div className="space-y-3 mt-4">
                    <div>
                      <label className="block text-sm text-gray-300 mb-1">
                        CPU Thermal Threshold ({config.thermal_cpu_threshold.toFixed(0)}°C)
                      </label>
                      <input
                        type="range"
                        min="60"
                        max="90"
                        step="1"
                        value={config.thermal_cpu_threshold}
                        onChange={(e) => setConfig({...config, thermal_cpu_threshold: parseFloat(e.target.value)})}
                        className="w-full"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-300 mb-1">
                        GPU Thermal Threshold ({config.thermal_gpu_threshold.toFixed(0)}°C)
                      </label>
                      <input
                        type="range"
                        min="60"
                        max="85"
                        step="1"
                        value={config.thermal_gpu_threshold}
                        onChange={(e) => setConfig({...config, thermal_gpu_threshold: parseFloat(e.target.value)})}
                        className="w-full"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-300 mb-1">
                        Emergency Threshold ({config.thermal_emergency_threshold.toFixed(0)}°C)
                      </label>
                      <input
                        type="range"
                        min="80"
                        max="95"
                        step="1"
                        value={config.thermal_emergency_threshold}
                        onChange={(e) => setConfig({...config, thermal_emergency_threshold: parseFloat(e.target.value)})}
                        className="w-full"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Real-time Processing Load */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <BarChart3 className="w-5 h-5 text-purple-400" />
                  <h4 className="font-medium text-white">Processing Load</h4>
                </div>
                
                {processingLoad && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-400">CPU Ops:</span>
                        <span className="text-white ml-1">
                          {processingLoad.load_statistics.avg_cpu_operations.toFixed(1)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">GPU Ops:</span>
                        <span className="text-white ml-1">
                          {processingLoad.load_statistics.avg_gpu_operations.toFixed(1)}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">CPU Ratio:</span>
                        <span className="text-white ml-1">
                          {(processingLoad.load_statistics.cpu_utilization_ratio * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">GPU Ratio:</span>
                        <span className="text-white ml-1">
                          {(processingLoad.load_statistics.gpu_utilization_ratio * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    
                    {/* Thermal Throttle Indicator */}
                    {processingLoad.load_statistics.thermal_throttle_frequency > 0 && (
                      <div className="bg-yellow-500/20 border border-yellow-500/50 rounded p-2">
                        <div className="flex items-center space-x-2 text-yellow-400">
                          <AlertTriangle className="w-4 h-4" />
                          <span className="text-xs">
                            Thermal throttling: {(processingLoad.load_statistics.thermal_throttle_frequency * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Current Allocation Display */}
                    <div className="space-y-2">
                      <div className="text-xs text-gray-400">Current Allocation</div>
                      <div className="flex space-x-2">
                        <div className="flex-1 bg-blue-500/20 rounded p-2 text-center">
                          <div className="text-xs text-blue-400">CPU</div>
                          <div className="text-sm font-medium text-white">
                            {processingLoad.current_allocation.cpu_percentage.toFixed(0)}%
                          </div>
                        </div>
                        <div className="flex-1 bg-green-500/20 rounded p-2 text-center">
                          <div className="text-xs text-green-400">GPU</div>
                          <div className="text-sm font-medium text-white">
                            {processingLoad.current_allocation.gpu_percentage.toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* SERC Integration Status */}
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="flex items-center space-x-2 mb-4">
                  <Hash className="w-5 h-5 text-cyan-400" />
                  <h4 className="font-medium text-white">SERC Integration</h4>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Mathematical Core:</span>
                    <span className="text-green-400">Active</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Thermal Bridge:</span>
                    <span className={thermalState?.thermal_available ? 'text-green-400' : 'text-red-400'}>
                      {thermalState?.thermal_available ? 'Connected' : 'Unavailable'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">GPU Manager:</span>
                    <span className="text-green-400">Integrated</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Data Pipeline:</span>
                    <span className="text-green-400">Flowing</span>
                  </div>
                  
                  {allocationInfo && (
                    <div className="mt-3 pt-3 border-t border-gray-600">
                      <div className="text-xs text-gray-400 mb-1">Performance Counters</div>
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Allocation Switches:</span>
                        <span className="text-white">{allocationInfo.allocation_switches}</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Thermal Events:</span>
                        <span className="text-white">
                          {processingLoad?.performance_counters?.thermal_throttle_events || 0}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Advanced Settings Save */}
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-400">
                Advanced settings integrate with thermal management and GPU cool-down systems for optimal profit extraction
              </div>
              <button
                onClick={saveConfiguration}
                disabled={isLoading}
                className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg font-medium disabled:opacity-50 flex items-center space-x-2"
              >
                <Settings className="w-4 h-4" />
                <span>{isLoading ? 'Saving...' : 'Save Advanced Settings'}</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NewsProfitSettingsPanel; 