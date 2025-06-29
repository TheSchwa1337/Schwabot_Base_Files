import {
  Activity, TrendingUp, TrendingDown, Hash, Zap, Target, Brain, Layers,
  AlertTriangle, CheckCircle, Clock, Database
} from 'lucide-react';

export default function CompleteTradingDashboard() {
  // System State
  const [activeMode, setActiveMode] = useState('paradox');
  const [isLive, setIsLive] = useState(false);
  const [detonationActive, setDetonationActive] = useState(false);

  // Trading State
  const [marketData, setMarketData] = useState({ price: 50000, volume: 1000, rsi: 50, entropy: 0.5, drift: 0, vwap: 49950, atr: 500, kellyFraction: 0.25 });
  const [ringValues, setRingValues] = useState({ R1: 0, R2: 0, R3: 0, R4: 0, R5: 0, R6: 0, R7: 0, R8: 0, R9: 0, R10: 0 });
  const [hashStream, setHashStream] = useState([]);
  const [timingHashes, setTimingHashes] = useState([]);
  const [glyphSignals, setGlyphSignals] = useState([]);
  const [stopPatterns, setStopPatterns] = useState([]);
  const [priceHistory, setPriceHistory] = useState([]);
  const [tpfState, setTpfState] = useState('INITIALIZING');
  const [paradoxVisible, setParadoxVisible] = useState(false);
  const [stabilized, setStabilized] = useState(false);
  const [phase, setPhase] = useState(0);

  // WebSocket State
  const [websocket, setWebsocket] = useState(null);

  // WebSocket Connection
  useEffect(() => {
    if (!isLive) {
      if (websocket) {
        console.log("Closing WebSocket connection.");
        websocket.close();
        setWebsocket(null);
      }
      return;
    }

    if (websocket) {
      console.log("WebSocket already open.");
      return;
    }

    const ws = new WebSocket('ws://localhost:8765');

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setWebsocket(ws);
    };

    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      //console.log("Received update:", update);

      switch (update.update_type) {
        case 'initial_state':
          setMarketData(update.data.market_data || marketData);
          setRingValues(update.data.ring_values || ringValues);
          setHashStream(update.data.hash_stream || hashStream);
          setTimingHashes(update.data.timing_hashes || timingHashes);
          setGlyphSignals(update.data.glyph_signals || glyphSignals);
          setStopPatterns(update.data.stop_patterns || stopPatterns);
          setPriceHistory(update.data.price_history || priceHistory);
          setTpfState(update.data.tpf_state || tpfState);
          setParadoxVisible(update.data.paradox_visible || paradoxVisible);
          setStabilized(update.data.stabilized || stabilized);
          setPhase(update.data.phase || phase);
          break;
        case 'market_data':
          setMarketData(update.data);
          // Add to price history for charting
          setPriceHistory(prev => [...prev.slice(-99), { timestamp: Date.now(), price: update.data.price, vwap: update.data.vwap }]);
          break;
        case 'hash_stream':
          setHashStream(prev => [...prev.slice(-49), update.data]);
          break;
        case 'timing_hashes':
          setTimingHashes(prev => [...prev.slice(-19), update.data]);
          break;
        case 'ring_values':
          setRingValues(update.data);
          break;
        case 'stop_patterns':
          setStopPatterns(update.data);
          break;
        case 'glyph_signals':
          setGlyphSignals(prev => [...prev.slice(-19), update.data]);
          break;
        case 'tpf_state_update':
          setTpfState(update.data.tpf_state);
          setParadoxVisible(update.data.paradox_visible);
          setStabilized(update.data.stabilized);
          setPhase(update.data.phase);
          break;
        case 'performance_metrics':
          // setPerformanceMetrics(update.data); // Uncomment if performanceMetrics state is defined
          break;
        default:
          console.warn("Unknown update type:", update.update_type);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setWebsocket(null); // Clear the WebSocket instance on close
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };

    // Cleanup function to close WebSocket when component unmounts
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [isLive]); // Dependency array: reconnect when isLive changes

  // Remove the old Live Data Simulation
  /*
  useEffect(() => {
    if (!isLive) return;

    const interval = setInterval(() => {
      const timestamp = Date.now();
      
      // Update market data with realistic movements
      setMarketData(prev => {
        const priceChange = (Math.random() - 0.5) * 100;
        const newPrice = prev.price + priceChange;
        const volumeChange = (Math.random() - 0.5) * 100;
        const newVolume = Math.max(100, prev.volume + volumeChange);
        
        return {
          ...prev,
          price: newPrice,
          volume: newVolume,
          rsi: Math.max(0, Math.min(100, prev.rsi + (Math.random() - 0.5) * 5)),
          entropy: Math.max(0, Math.min(1, prev.entropy + (Math.random() - 0.5) * 0.1)),
          drift: Math.sin(Date.now() * 0.001) * 2,
          vwap: newPrice * 0.999 + prev.vwap * 0.001,
          atr: Math.abs(priceChange) * 0.1 + prev.atr * 0.9,
          kellyFraction: Math.max(0, Math.min(1, prev.kellyFraction + (Math.random() - 0.5) * 0.05))
        };
      });

      // Generate hash stream data
      const hashValue = Math.random().toString(36).substring(2, 10);
      const entropyTag = Math.floor(Math.random() * 144);
      
      setHashStream(prev => [...prev.slice(-49), {
        timestamp,
        hash: hashValue,
        entropy: entropyTag,
        confidence: Math.random(),
        pattern: Array.from({length: 8}, () => Math.floor(Math.random() * 16))
      }]);

      // Update timing hashes
      setTimingHashes(prev => [...prev.slice(-19), {
        timestamp,
        hash: hashValue,
        state: tpfState
      }]);

      // Update ring values (RITTLE-GEMM simulation)
      setRingValues(prev => ({
        R1: prev.R1 * 0.95 + Math.random() * 0.1,
        R2: prev.R2 * 0.9 + (Math.random() - 0.5) * 0.2,
        R3: prev.R3 * 0.95 + Math.random() * 0.05,
        R4: Math.random(),
        R5: Math.random() * 4 - 2,
        R6: 0.5 + Math.sin(timestamp * 0.001) * 0.3,
        R7: prev.R7 + (Math.random() - 0.5) * 0.01,
        R8: prev.R8 * 0.9 + (Math.random() > 0.7 ? Math.random() * 0.1 : 0),
        R9: Math.abs(Math.random() - prev.R4),
        R10: Math.random() > 0.9 ? 1 : 0
      }));

      // Update TPF state
      setPhase(prev => (prev + 1) % 100);
      
    }, 200);

    return () => clearInterval(interval);
  }, [isLive, tpfState]);
  */

  // TPF State Management - this now needs to be driven by backend
  // For now, I'll keep the logic that updates `phase`, `paradoxVisible`, and `stabilized` based on a simulated `phase`
  // but ideally, these states should also come from the backend.
  useEffect(() => {
    // This part should eventually be driven by backend data
    if (tpfState === 'PARADOX_DETECTED' && !paradoxVisible) {
      setParadoxVisible(true);
      setStabilized(false);
    } else if (tpfState === 'TPF_STABILIZED' && !stabilized) {
      setStabilized(true);
      setParadoxVisible(false);
    } else if (tpfState === 'INITIALIZING' && (paradoxVisible || stabilized)) {
      setParadoxVisible(false);
      setStabilized(false);
    }
  }, [tpfState, paradoxVisible, stabilized]);

  // Generate trading signals based on current state - this will now send a command to the backend
  const generateTradingSignal = useCallback(() => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      const signalCommand = {
        command_type: "generate_signal",
        data: {
          price: marketData.price,
          rsi: marketData.rsi,
          tpfState: tpfState,
          stabilized: stabilized,
          paradoxVisible: paradoxVisible,
          // other relevant market data to help backend generate signal
        }
      };
      websocket.send(JSON.stringify(signalCommand));
      console.log("Sent generate_signal command to backend.");
    } else {
      console.warn("WebSocket not connected. Cannot generate signal.");
    }
  }, [websocket, marketData, tpfState, stabilized, paradoxVisible]);

  // Detonation Protocol - this will now send a command to the backend
  const triggerDetonation = useCallback(() => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      setDetonationActive(true);
      const detonationCommand = {
        command_type: "trigger_detonation",
        data: {
          // You might send current market context if needed by backend for detonation
        }
      };
      websocket.send(JSON.stringify(detonationCommand));
      console.log("Sent trigger_detonation command to backend.");
      setTimeout(() => setDetonationActive(false), 3000); // Visual feedback on frontend
    } else {
      console.warn("WebSocket not connected. Cannot trigger detonation.");
    }
  }, [websocket]);

  // Calculate derived metrics - these will now be calculated based on received data
  const currentMetrics = {
    hashEntropy: hashStream.reduce((sum, h) => sum + (h.entropy || 0), 0) / Math.max(hashStream.length, 1),
    signalStrength: glyphSignals.filter(s => s.confidence > 0.7).length / Math.max(glyphSignals.length, 1),
    ringStability: Object.values(ringValues).reduce((sum, val) => sum + Math.abs(val), 0) / 10,
    tpfCoherence: stabilized ? 1.0 : paradoxVisible ? 0.3 : 0.6
  };

  return (
    <div className="w-full h-screen bg-gradient-to-br from-black via-gray-900 to-blue-900 text-white overflow-hidden">
      {/* Header Controls */}
      <div className="flex justify-between items-center p-4 border-b border-gray-700">
        <div className="flex items-center space-x-4">
          <h1 className="text-xl font-bold">Schwabot Trading Dashboard</h1>
          <div className="flex space-x-2">
            {['paradox', 'hash', 'rings', 'signals'].map(mode => (
              <button
                key={mode}
                onClick={() => setActiveMode(mode)}
                className={`px-3 py-1 rounded text-sm ${
                  activeMode === mode ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm">
            <span className="text-gray-400">TPF State:</span>
            <span className={`ml-2 font-mono ${
              tpfState === 'TPF_STABILIZED' ? 'text-green-400' :
              tpfState === 'PARADOX_DETECTED' ? 'text-red-400' : 'text-yellow-400'
            }`}>
              {tpfState}
            </span>
          </div>
          
          <button
            onClick={() => setIsLive(!isLive)}
            className={`px-4 py-2 rounded font-bold ${
              isLive ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'
            }`}
          >
            {isLive ? 'LIVE' : 'START'}
          </button>
          
          <button
            onClick={triggerDetonation}
            className={`px-4 py-2 rounded font-bold ${
              detonationActive 
                ? 'bg-red-600 animate-pulse' 
                : 'bg-orange-600 hover:bg-orange-700'
            }`}
          >
            {detonationActive ? 'DETONATING...' : '1337 PROTOCOL'}
          </button>
        </div>
      </div>

      {/* Main Dashboard Content */}
      <div className="flex h-full">
        {/* Left Panel - Core Visualizations */}
        <div className="flex-1 p-4 space-y-4">
          {activeMode === 'paradox' && (
            <div className="space-y-4">
              {/* TPF Paradox Visualization */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3 flex items-center">
                  <Brain className="w-5 h-5 mr-2" />
                  Recursive Paradox Engine
                </h3>
                <div className="relative h-64 flex items-center justify-center">
                  <svg viewBox="0 0 100 100" width="200" height="200" className={detonationActive ? 'animate-spin' : ''}>
                    {/* Triangle */}
                    <polygon 
                      points="50,10 10,90 90,90" 
                      fill="none" 
                      stroke={stabilized ? "#00ff00" : paradoxVisible ? "#ff3300" : "#00aa00"} 
                      strokeWidth="1.5"
                    />
                    {/* Fourth point paradox */}
                    <circle 
                      cx="50" 
                      cy="90" 
                      r="2" 
                      fill="#ff3300" 
                      opacity={paradoxVisible && !stabilized ? "1" : "0"}
                    />
                    {/* Inner circle */}
                    <circle 
                      cx="50" 
                      cy="60" 
                      r={20 + (marketData.rsi / 100) * 10} 
                      fill="none" 
                      stroke="#00aaff" 
                      strokeWidth="0.5" 
                      opacity="0.6"
                    />
                  </svg>
                </div>
                <div className="text-center text-sm text-gray-400">
                  Phase: {phase} | Coherence: {currentMetrics.tpfCoherence.toFixed(2)}
                </div>
              </div>

              {/* Market Data Chart */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3">Price & VWAP</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={priceHistory.slice(-50)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="timestamp" hide />
                      <YAxis domain={['dataMin - 100', 'dataMax + 100']} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="price" stroke="#8884d8" dot={false} />
                      <Line type="monotone" dataKey="vwap" stroke="#82ca9d" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeMode === 'hash' && (
            <div className="space-y-4">
              {/* Hash Stream Visualization */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3 flex items-center">
                  <Hash className="w-5 h-5 mr-2" />
                  Live Hash Stream
                </h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {hashStream.slice(-10).map((hash, i) => (
                    <div key={i} className="flex items-center space-x-4 text-sm bg-gray-700 p-2 rounded">
                      <div className="font-mono text-blue-400">{hash.hash ? hash.hash.substring(0,8) : 'N/A'}</div>
                      <div className="text-gray-400">E:{hash.entropy || 'N/A'}</div>
                      <div className={`text-xs px-2 py-1 rounded ${
                        (hash.confidence || 0) > 0.7 ? 'bg-green-900 text-green-300' : 'bg-yellow-900 text-yellow-300'
                      }`}>
                        {((hash.confidence || 0) * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-gray-500">
                        {hash.pattern ? hash.pattern.slice(0, 4).join('-') : 'N/A'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Hash Entropy Distribution */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3">Hash Entropy Distribution</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={hashStream.slice(-20).map((h, i) => ({ index: i, entropy: h.entropy }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="index" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="entropy" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeMode === 'rings' && (
            <div className="space-y-4">
              {/* RITTLE-GEMM Ring Values */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3 flex items-center">
                  <Layers className="w-5 h-5 mr-2" />
                  RITTLE-GEMM Ring Values
                </h3>
                <div className="grid grid-cols-5 gap-4">
                  {Object.entries(ringValues).map(([ring, value]) => (
                    <div key={ring} className="text-center">
                      <div className="text-xs text-gray-400">{ring}</div>
                      <div className={`text-sm font-mono ${
                        Math.abs(value) > 0.5 ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {value.toFixed(3)}
                      </div>
                      <div className="w-full bg-gray-700 rounded h-2 mt-1">
                        <div 
                          className="bg-blue-500 h-2 rounded transition-all duration-200"
                          style={{ width: `${Math.min(Math.abs(value) * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Ring Stability Chart */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3">Ring Stability Over Time</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart data={Object.entries(ringValues).map(([ring, value], i) => ({ ring: i, value, name: ring }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="ring" />
                      <YAxis />
                      <Tooltip formatter={(value, name, props) => [value.toFixed(3), props.payload.name]} />
                      <Scatter dataKey="value" fill="#8884d8" />
                      <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeMode === 'signals' && (
            <div className="space-y-4">
              {/* Trading Signals */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3 flex items-center">
                  <Target className="w-5 h-5 mr-2" />
                  Glyph Trading Signals
                </h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {glyphSignals.slice(-8).map((signal, i) => (
                    <div key={i} className={`p-3 rounded flex justify-between items-center ${
                      signal.type === 'BUY' ? 'bg-green-900' :
                      signal.type === 'SELL' ? 'bg-red-900' : 'bg-yellow-900'
                    }`}>
                      <div className="flex items-center space-x-3">
                        {signal.type === 'BUY' ? <TrendingUp className="w-4 h-4" /> :
                         signal.type === 'SELL' ? <TrendingDown className="w-4 h-4" /> :
                         <Activity className="w-4 h-4" />}
                        <span className="font-bold">{signal.type}</span>
                        <span className="text-sm">${signal.price.toFixed(2)}</span>
                      </div>
                      <div className="text-right text-sm">
                        <div>{((signal.confidence || 0) * 100).toFixed(0)}%</div>
                        <div className="text-xs text-gray-400">{signal.tpfState || 'N/A'}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Signal Confidence Chart */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-bold mb-3">Signal Confidence Over Time</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={glyphSignals.slice(-20).map((s, i) => ({ index: i, confidence: (s.confidence || 0) * 100 }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="index" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Line type="monotone" dataKey="confidence" stroke="#8884d8" dot={false} />
                      <ReferenceLine y={70} stroke="#ff6b6b" strokeDasharray="2 2" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Status & Metrics */}
        <div className="w-80 p-4 border-l border-gray-700 space-y-4">
          {/* Market Status */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-3 flex items-center">
              <Activity className="w-5 h-5 mr-2" />
              Market Status
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Price:</span>
                <span className="font-mono">${marketData.price.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Volume:</span>
                <span className="font-mono">{marketData.volume.toFixed(0)}</span>
              </div>
              <div className="flex justify-between">
                <span>RSI:</span>
                <span className={`font-mono ${
                  marketData.rsi > 70 ? 'text-red-400' : 
                  marketData.rsi < 30 ? 'text-green-400' : 'text-gray-300'
                }`}>
                  {marketData.rsi.toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span>ATR:</span>
                <span className="font-mono">{marketData.atr.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Kelly:</span>
                <span className="font-mono">{marketData.kellyFraction.toFixed(3)}</span>
              </div>
            </div>
          </div>

          {/* System Metrics */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-3 flex items-center">
              <Database className="w-5 h-5 mr-2" />
              System Metrics
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Hash Entropy:</span>
                <span className="font-mono">{currentMetrics.hashEntropy.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Signal Strength:</span>
                <span className="font-mono">{(currentMetrics.signalStrength * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Ring Stability:</span>
                <span className="font-mono">{currentMetrics.ringStability.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span>TPF Coherence:</span>
                <span className="font-mono">{currentMetrics.tpfCoherence.toFixed(3)}</span>
              </div>
            </div>
          </div>

          {/* Recent Activity Log */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-3 flex items-center">
              <Clock className="w-5 h-5 mr-2" />
              Activity Log
            </h3>
            <div className="space-y-1 text-xs max-h-48 overflow-y-auto">
              {timingHashes.slice(-10).map((entry, i) => (
                <div key={i} className="flex items-center space-x-2 text-gray-400">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                  <span className="font-mono">{entry.hash ? entry.hash.substring(0, 6) : 'N/A'}...</span>
                  <span className={`text-xs px-1 rounded ${
                    entry.state === 'TPF_STABILIZED' ? 'bg-green-900 text-green-300' :
                    entry.state === 'PARADOX_DETECTED' ? 'bg-red-900 text-red-300' :
                    'bg-gray-700 text-gray-300'
                  }`}>
                    {entry.state || 'N/A'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <button 
                onClick={generateTradingSignal}
                className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
              >
                Generate Signal
              </button>
              <button 
                onClick={() => { if (websocket && websocket.readyState === WebSocket.OPEN) websocket.send(JSON.stringify({ command_type: "clear_hash_stream" })); }}
                className="w-full px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded text-sm"
              >
                Clear Hash Stream
              </button>
              <button 
                onClick={() => { if (websocket && websocket.readyState === WebSocket.OPEN) websocket.send(JSON.stringify({ command_type: "clear_glyph_signals" })); }}
                className="w-full px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded text-sm"
              >
                Clear Signals
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 