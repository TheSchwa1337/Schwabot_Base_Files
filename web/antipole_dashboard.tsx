import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Activity, DollarSign, Clock, Zap, Target, Shield, Cpu, Thermometer, Eye } from 'lucide-react';

interface AntiPoleState {
  delta_psi_bar: number;
  icap_probability: number;
  hash_entropy: number;
  is_ready: boolean;
  profit_tier: string | null;
  phase_lock: boolean;
}

interface ThermalState {
  thermal_load: number;
  state: string;
  cooldown_active: boolean;
  safe_to_trade: boolean;
  cpu_temp: number;
  cpu_usage: number;
}

interface PortfolioState {
  total_value: number;
  cash_balance: number;
  btc_position: number;
  btc_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
  win_rate: number;
  max_drawdown: number;
  active_opportunities: number;
}

interface ProfitOpportunity {
  id: string;
  opportunity_type: string;
  confidence: number;
  profit_tier: string;
  expected_return: number;
  entry_price?: number;
  exit_price?: number;
  position_size?: number;
}

interface GlyphData {
  id: string;
  glyph_type: string;
  coordinates: number[];
  color: number[];
  size: number;
  intensity: number;
  metadata: any;
}

interface TesseractFrame {
  frame_id: string;
  timestamp: string;
  glyphs: GlyphData[];
  camera_position: number[];
  profit_tier: string | null;
  thermal_state: string;
  system_health: any;
}

const AntiPoleDashboard: React.FC = () => {
  const [connected, setConnected] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  
  // Anti-Pole System State
  const [antipoleState, setAntipoleState] = useState<AntiPoleState>({
    delta_psi_bar: 0,
    icap_probability: 0,
    hash_entropy: 0.5,
    is_ready: false,
    profit_tier: null,
    phase_lock: false
  });
  
  const [thermalState, setThermalState] = useState<ThermalState>({
    thermal_load: 0.3,
    state: 'COOL',
    cooldown_active: false,
    safe_to_trade: true,
    cpu_temp: 45.0,
    cpu_usage: 25.0
  });
  
  const [portfolioState, setPortfolioState] = useState<PortfolioState>({
    total_value: 100000,
    cash_balance: 80000,
    btc_position: 0.5,
    btc_value: 20000,
    unrealized_pnl: 0,
    realized_pnl: 0,
    win_rate: 0.65,
    max_drawdown: 0.08,
    active_opportunities: 0
  });
  
  const [opportunities, setOpportunities] = useState<ProfitOpportunity[]>([]);
  const [tesseractFrame, setTesseractFrame] = useState<TesseractFrame | null>(null);
  
  // Historical data for charts
  const [entropyHistory, setEntropyHistory] = useState<Array<{time: string, entropy: number, icap: number, drift: number}>>([]);
  const [thermalHistory, setThermalHistory] = useState<Array<{time: string, load: number, temp: number}>>([]);
  const [pnlHistory, setPnlHistory] = useState<Array<{time: string, value: number, drawdown: number}>>([]);

  // WebSocket connection
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
    
    ws.onopen = () => {
      setConnected(true);
      setWsConnection(ws);
      console.log('🔗 Connected to Anti-Pole Tesseract server');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'tesseract_frame') {
          handleTesseractFrame(data);
        } else if (data.type === 'tick_report') {
          handleTickReport(data);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      setConnected(false);
      setWsConnection(null);
      console.log('❌ Disconnected from Anti-Pole server');
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    return () => {
      ws.close();
    };
  }, []);

  // Clock update
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const handleTesseractFrame = (frameData: any) => {
    setTesseractFrame({
      frame_id: frameData.frame_id,
      timestamp: frameData.timestamp,
      glyphs: frameData.glyphs,
      camera_position: frameData.camera_position,
      profit_tier: frameData.profit_tier,
      thermal_state: frameData.thermal_state,
      system_health: frameData.system_health
    });
  };

  const handleTickReport = (report: any) => {
    const timestamp = new Date(report.timestamp);
    const timeStr = timestamp.toLocaleTimeString();
    
    // Update Anti-Pole state
    if (report.antipole_state) {
      setAntipoleState(report.antipole_state);
      
      // Add to entropy history
      setEntropyHistory(prev => [...prev.slice(-49), {
        time: timeStr,
        entropy: report.antipole_state.hash_entropy,
        icap: report.antipole_state.icap_probability,
        drift: report.antipole_state.delta_psi_bar
      }]);
    }
    
    // Update thermal state
    if (report.thermal_state) {
      setThermalState(report.thermal_state);
      
      // Add to thermal history
      setThermalHistory(prev => [...prev.slice(-49), {
        time: timeStr,
        load: report.thermal_state.thermal_load,
        temp: report.thermal_state.cpu_temp || 45
      }]);
    }
    
    // Update portfolio state
    if (report.portfolio_state) {
      setPortfolioState(report.portfolio_state);
      
      // Add to P&L history
      setPnlHistory(prev => [...prev.slice(-49), {
        time: timeStr,
        value: report.portfolio_state.total_value,
        drawdown: report.portfolio_state.max_drawdown * -100
      }]);
    }
    
    // Update opportunities
    if (report.opportunities) {
      setOpportunities(report.opportunities);
    }
  };

  const getThermalColor = (state: string) => {
    const colors = {
      'COLD': '#3B82F6',
      'COOL': '#10B981',
      'WARM': '#F59E0B',
      'HOT': '#EF4444',
      'CRITICAL': '#DC2626',
      'EMERGENCY': '#7F1D1D'
    };
    return colors[state as keyof typeof colors] || '#6B7280';
  };

  const getProfitTierColor = (tier: string | null) => {
    const colors = {
      'PLATINUM': '#E5E7EB',
      'GOLD': '#F59E0B',
      'SILVER': '#9CA3AF',
      'BRONZE': '#92400E'
    };
    return colors[tier as keyof typeof colors] || '#6B7280';
  };

  const StatusIndicator: React.FC<{status: boolean, label: string}> = ({status, label}) => (
    <div className={`flex items-center space-x-2 ${status ? 'text-green-500' : 'text-red-500'}`}>
      {status ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
      <span className="text-sm font-medium">{label}</span>
    </div>
  );

  const TesseractVisualization: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    
    useEffect(() => {
      if (!tesseractFrame || !canvasRef.current) return;
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#111827';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw glyphs (simplified 2D projection of 4D coordinates)
      tesseractFrame.glyphs.forEach(glyph => {
        const [x, y, z, w] = glyph.coordinates;
        
        // Project 4D to 2D (simple projection ignoring z,w for visualization)
        const screenX = (x + 10) * (canvas.width / 20);  // Map [-10,10] to canvas width
        const screenY = (y + 10) * (canvas.height / 20); // Map [-10,10] to canvas height
        
        // Draw glyph
        ctx.beginPath();
        ctx.arc(screenX, screenY, glyph.size * 5, 0, 2 * Math.PI);
        
        const [r, g, b, a] = glyph.color;
        ctx.fillStyle = `rgba(${Math.floor(r*255)}, ${Math.floor(g*255)}, ${Math.floor(b*255)}, ${a})`;
        ctx.fill();
        
        // Add glow effect for high intensity
        if (glyph.intensity > 0.7) {
          ctx.shadowColor = ctx.fillStyle;
          ctx.shadowBlur = glyph.intensity * 10;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      });
      
      // Draw profit tier indicator
      if (tesseractFrame.profit_tier) {
        ctx.fillStyle = getProfitTierColor(tesseractFrame.profit_tier);
        ctx.font = '16px Arial';
        ctx.fillText(`${tesseractFrame.profit_tier} TIER`, 10, 30);
      }
      
    }, [tesseractFrame]);
    
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4 text-white">4D Tesseract Visualization</h3>
        <canvas 
          ref={canvasRef}
          width={400}
          height={300}
          className="border border-gray-600 rounded"
        />
        {tesseractFrame && (
          <div className="mt-2 text-sm text-gray-400">
            Frame: {tesseractFrame.frame_id.slice(0, 8)}... | 
            Glyphs: {tesseractFrame.glyphs.length} | 
            Camera: [{tesseractFrame.camera_position.map(c => c.toFixed(1)).join(', ')}]
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-blue-400">🔺 Anti-Pole Navigator 🔻</h1>
            <p className="text-gray-400">Quantum Profit Navigation & Tesseract Visualization</p>
          </div>
          <div className="text-right">
            <div className="text-lg font-mono">{currentTime.toLocaleTimeString()}</div>
            <StatusIndicator status={connected} label="Tesseract Link" />
            <StatusIndicator status={thermalState.safe_to_trade} label="Thermal Safe" />
            <StatusIndicator status={antipoleState.is_ready} label="Anti-Pole Ready" />
          </div>
        </div>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-5 gap-6 mb-8">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">ICAP Probability</h3>
            <Target className="w-5 h-5 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-blue-400">
            {(antipoleState.icap_probability * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-400">
            {antipoleState.is_ready ? '🔥 READY' : '⏳ Waiting'}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Profit Tier</h3>
            <Zap className="w-5 h-5" style={{color: getProfitTierColor(antipoleState.profit_tier)}} />
          </div>
          <div className="text-2xl font-bold" style={{color: getProfitTierColor(antipoleState.profit_tier)}}>
            {antipoleState.profit_tier || 'NONE'}
          </div>
          <div className="text-sm text-gray-400">
            {antipoleState.phase_lock ? '🔒 Phase Lock' : '🔄 Floating'}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Portfolio Value</h3>
            <DollarSign className="w-5 h-5 text-green-400" />
          </div>
          <div className="text-2xl font-bold text-green-400">
            ${portfolioState.total_value.toLocaleString()}
          </div>
          <div className="text-sm text-gray-400">
            P&L: ${portfolioState.realized_pnl.toFixed(0)}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Thermal State</h3>
            <Thermometer className="w-5 h-5" style={{color: getThermalColor(thermalState.state)}} />
          </div>
          <div className="text-2xl font-bold" style={{color: getThermalColor(thermalState.state)}}>
            {thermalState.state}
          </div>
          <div className="text-sm text-gray-400">
            Load: {(thermalState.thermal_load * 100).toFixed(0)}%
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-400">Opportunities</h3>
            <Eye className="w-5 h-5 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-purple-400">
            {opportunities.length}
          </div>
          <div className="text-sm text-gray-400">
            Active signals
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-3 gap-6 mb-8">
        
        {/* Anti-Pole Entropy Monitor */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Anti-Pole Dynamics</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={entropyHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Line type="monotone" dataKey="entropy" stroke="#3B82F6" name="Hash Entropy" />
              <Line type="monotone" dataKey="icap" stroke="#10B981" name="ICAP" />
              <Line type="monotone" dataKey="drift" stroke="#F59E0B" name="Drift" />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-2 text-sm text-gray-400">
            Current Drift: {antipoleState.delta_psi_bar.toFixed(6)} | 
            Entropy: {antipoleState.hash_entropy.toFixed(3)}
          </div>
        </div>

        {/* Profit Opportunities */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Active Opportunities</h3>
          <div className="space-y-3 max-h-48 overflow-y-auto">
            {opportunities.length === 0 ? (
              <div className="text-gray-500 text-center py-8">
                No opportunities detected
              </div>
            ) : (
              opportunities.map((opp, index) => (
                <div key={opp.id} className="bg-gray-700 rounded p-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-medium text-sm">
                        {opp.opportunity_type} - {opp.profit_tier}
                      </div>
                      <div className="text-xs text-gray-400">
                        Confidence: {(opp.confidence * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-400">
                        Expected: {(opp.expected_return * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="text-right">
                      {opp.entry_price && (
                        <div className="text-sm">${opp.entry_price.toLocaleString()}</div>
                      )}
                      {opp.position_size && (
                        <div className="text-xs text-gray-400">
                          Size: {opp.position_size.toFixed(4)}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Thermal Monitor */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Thermal Monitor</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={thermalHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Area 
                type="monotone" 
                dataKey="load" 
                stroke="#EF4444" 
                fill="#EF4444" 
                fillOpacity={0.3}
                name="Thermal Load"
              />
              <Area 
                type="monotone" 
                dataKey="temp" 
                stroke="#F59E0B" 
                fill="#F59E0B" 
                fillOpacity={0.2}
                name="CPU Temp"
              />
            </AreaChart>
          </ResponsiveContainer>
          <div className="mt-2 space-y-1">
            <div className="flex justify-between text-sm">
              <span>CPU Temperature:</span>
              <span style={{color: getThermalColor(thermalState.state)}}>
                {thermalState.cpu_temp.toFixed(1)}°C
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span>Cooldown Active:</span>
              <span className={thermalState.cooldown_active ? 'text-red-400' : 'text-green-400'}>
                {thermalState.cooldown_active ? 'YES' : 'NO'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-2 gap-6">
        
        {/* Tesseract Visualization */}
        <TesseractVisualization />

        {/* Portfolio Performance */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Portfolio Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={pnlHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#10B981" 
                name="Portfolio Value"
              />
              <Line 
                type="monotone" 
                dataKey="drawdown" 
                stroke="#EF4444" 
                name="Drawdown %"
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-gray-400">Win Rate</div>
              <div className="text-lg font-semibold text-green-400">
                {(portfolioState.win_rate * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-gray-400">Max Drawdown</div>
              <div className="text-lg font-semibold text-red-400">
                {(portfolioState.max_drawdown * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-gray-400">BTC Position</div>
              <div className="text-lg font-semibold text-blue-400">
                {portfolioState.btc_position.toFixed(4)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 text-center text-gray-400 text-sm">
        Anti-Pole Navigator v4.0 | Connected: {connected ? '🟢' : '🔴'} | 
        Tesseract Glyphs: {tesseractFrame?.glyphs.length || 0} | 
        Last Update: {currentTime.toLocaleString()}
      </div>
    </div>
  );
};

export default AntiPoleDashboard; 