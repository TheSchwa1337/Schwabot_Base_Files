import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Activity, DollarSign, Clock, Zap } from 'lucide-react';

const PracticalSchwabot = () => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [systemStatus, setSystemStatus] = useState('active');

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Mock real-time data
  const entropyData = [
    { time: '09:00', value: 0.45, threshold: 0.7 },
    { time: '09:05', value: 0.62, threshold: 0.7 },
    { time: '09:10', value: 0.78, threshold: 0.7 },
    { time: '09:15', value: 0.54, threshold: 0.7 },
    { time: '09:20', value: 0.39, threshold: 0.7 },
    { time: '09:25', value: 0.67, threshold: 0.7 }
  ];

  const patternData = [
    { pattern: 'Trend Continuation', confidence: 0.85, active: true },
    { pattern: 'Mean Reversion', confidence: 0.72, active: false },
    { pattern: 'Breakout Signal', confidence: 0.91, active: true },
    { pattern: 'Support/Resistance', confidence: 0.68, active: false },
    { pattern: 'Volume Surge', confidence: 0.79, active: true }
  ];

  const riskMetrics = [
    { metric: 'Exposure', value: 0.65, max: 1 },
    { metric: 'Volatility', value: 0.42, max: 1 },
    { metric: 'Correlation', value: 0.33, max: 1 },
    { metric: 'Liquidity', value: 0.88, max: 1 },
    { metric: 'Drawdown', value: 0.15, max: 1 }
  ];

  const performanceData = [
    { time: '1D', pnl: 2.3, benchmark: 1.1 },
    { time: '1W', pnl: 8.7, benchmark: 3.2 },
    { time: '1M', pnl: 15.4, benchmark: 7.8 },
    { time: '3M', pnl: 32.1, benchmark: 18.9 },
    { time: '1Y', pnl: 127.5, benchmark: 45.3 }
  ];

  const StatusIndicator = ({ status, label }) => {
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

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-blue-400">Schwabot Trading System</h1>
          <div className="text-right">
            <div className="text-lg font-mono">{currentTime.toLocaleTimeString()}</div>
            <StatusIndicator status="active" label="System Operational" />
          </div>
        </div>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-4 gap-6 mb-8">
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
            <h3 className="text-sm font-medium text-gray-400">Active Patterns</h3>
            <Zap className="w-5 h-5 text-blue-400" />
          </div>
          <div className="text-2xl font-bold">3/5</div>
          <div className="text-sm text-gray-400">High confidence signals</div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">System Load</h3>
            <Activity className="w-5 h-5 text-yellow-400" />
          </div>
          <div className="text-2xl font-bold">67%</div>
          <div className="text-sm text-gray-400">Processing 847 ticks/sec</div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Max Drawdown</h3>
            <TrendingDown className="w-5 h-5 text-red-400" />
          </div>
          <div className="text-2xl font-bold">-8.2%</div>
          <div className="text-sm text-gray-400">Within limits</div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-3 gap-6">
        
        {/* Market Entropy Monitor */}
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
              <Line 
                type="monotone" 
                dataKey="threshold" 
                stroke="#EF4444" 
                strokeDasharray="5 5"
              />
            </AreaChart>
          </ResponsiveContainer>
          <div className="mt-2 text-sm text-gray-400">
            Current: 0.67 | Threshold: 0.70
          </div>
        </div>

        {/* Pattern Recognition */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Active Patterns</h3>
          <div className="space-y-3">
            {patternData.map((pattern, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${pattern.active ? 'bg-green-500' : 'bg-gray-500'}`}></div>
                  <span className="text-sm">{pattern.pattern}</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-mono">{(pattern.confidence * 100).toFixed(0)}%</div>
                  <div className="w-16 bg-gray-700 rounded-full h-1">
                    <div 
                      className="bg-blue-500 h-1 rounded-full" 
                      style={{ width: `${pattern.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Risk Assessment */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Risk Radar</h3>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={riskMetrics}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
              <PolarRadiusAxis 
                domain={[0, 1]} 
                tick={{ fill: '#9CA3AF', fontSize: 10 }}
                tickCount={6}
              />
              <Radar
                name="Risk Level"
                dataKey="value"
                stroke="#EF4444"
                fill="#EF4444"
                fillOpacity={0.3}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Performance Chart */}
        <div className="col-span-2 bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Performance vs Benchmark</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="pnl" fill="#10B981" name="Schwabot P&L %" />
              <Bar dataKey="benchmark" fill="#6B7280" name="Benchmark %" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* System Health */}
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
              <StatusIndicator status="active" label="Pattern Engine" />
              <StatusIndicator status="active" label="Risk Manager" />
              <StatusIndicator status="warning" label="Data Feed" />
              <StatusIndicator status="active" label="Execution Engine" />
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 text-center text-gray-400 text-sm">
        Schwabot v2.1.0 | Last Updated: {currentTime.toLocaleDateString()} | 
        Uptime: 47d 8h 23m | Processing: 847 ticks/sec
      </div>
    </div>
  );
};

export default PracticalSchwabot; 