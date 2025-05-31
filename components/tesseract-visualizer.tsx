import React, { useState, useEffect } from 'react';
import { Cube, Activity, Zap, Grid, Cpu, Gpu } from 'lucide-react';

interface TesseractMetrics {
  magnitude: number;
  centroid_distance: number;
  axis_correlation: number;
  stability: number;
  harmonic_ratio: number;
  primary_dominance: number;
  dimensional_spread: number;
}

interface SystemMetrics {
  gpu_utilization: number;
  gpu_temperature: number;
  cpu_utilization: number;
  cpu_temperature: number;
  memory_usage: number;
  zpe_zone: string;
  drift_band: number;
}

interface PatternState {
  dimensions: number[];
  metrics: TesseractMetrics;
  system: SystemMetrics;
  coherence: number;
  homeostasis: number;
}

const TesseractVisualizer: React.FC = () => {
  const [patternState, setPatternState] = useState<PatternState>({
    dimensions: Array(8).fill(0),
    metrics: {
      magnitude: 0,
      centroid_distance: 0,
      axis_correlation: 0,
      stability: 0,
      harmonic_ratio: 0,
      primary_dominance: 0,
      dimensional_spread: 0
    },
    system: {
      gpu_utilization: 0,
      gpu_temperature: 0,
      cpu_utilization: 0,
      cpu_temperature: 0,
      memory_usage: 0,
      zpe_zone: 'SAFE',
      drift_band: 0
    },
    coherence: 1.0,
    homeostasis: 1.0
  });

  useEffect(() => {
    const updatePattern = () => {
      // Simulate pattern updates
      const newDimensions = patternState.dimensions.map(d => 
        Math.max(0, Math.min(15, d + (Math.random() - 0.5) * 2))
      );
      
      // Calculate new metrics
      const magnitude = Math.sqrt(newDimensions.reduce((sum, d) => sum + d * d, 0));
      const centroid_distance = Math.sqrt(newDimensions.reduce((sum, d) => sum + Math.pow(d - 7.5, 2), 0));
      const mean = newDimensions.reduce((sum, d) => sum + d, 0) / newDimensions.length;
      const variance = newDimensions.reduce((sum, d) => sum + Math.pow(d - mean, 2), 0) / newDimensions.length;
      const stability = 1.0 / (1.0 + variance);
      
      // Calculate drift band
      const drift_band = Math.abs(centroid_distance - 7.5) / 7.5;
      
      // Simulate system metrics
      const gpu_util = Math.min(100, Math.max(0, patternState.system.gpu_utilization + (Math.random() - 0.5) * 10));
      const cpu_util = Math.min(100, Math.max(0, patternState.system.cpu_utilization + (Math.random() - 0.5) * 5));
      
      // Determine ZPE zone based on drift band
      let zpe_zone = 'SAFE';
      if (drift_band > 0.4) {
        zpe_zone = 'UNSAFE';
      } else if (drift_band > 0.2) {
        zpe_zone = 'WARM';
      }
      
      setPatternState(prev => ({
        dimensions: newDimensions,
        metrics: {
          magnitude,
          centroid_distance,
          axis_correlation: Math.random() * 2 - 1,
          stability,
          harmonic_ratio: Math.random() * 2,
          primary_dominance: Math.random(),
          dimensional_spread: Math.max(...newDimensions) - Math.min(...newDimensions)
        },
        system: {
          gpu_utilization: gpu_util,
          gpu_temperature: Math.min(85, Math.max(30, prev.system.gpu_temperature + (Math.random() - 0.5) * 2)),
          cpu_utilization: cpu_util,
          cpu_temperature: Math.min(75, Math.max(30, prev.system.cpu_temperature + (Math.random() - 0.5) * 1)),
          memory_usage: Math.min(100, Math.max(0, prev.system.memory_usage + (Math.random() - 0.5) * 3)),
          zpe_zone,
          drift_band
        },
        coherence: Math.max(0, Math.min(1, prev.coherence + (Math.random() - 0.5) * 0.1)),
        homeostasis: Math.max(0, Math.min(1, prev.homeostasis + (Math.random() - 0.5) * 0.05))
      }));
    };

    const interval = setInterval(updatePattern, 1000);
    return () => clearInterval(interval);
  }, []);

  const getZpeZoneColor = (zone: string) => {
    switch (zone) {
      case 'SAFE': return 'text-green-400';
      case 'WARM': return 'text-yellow-400';
      case 'UNSAFE': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-900 rounded-xl shadow-lg text-white">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold flex items-center">
          <Cube className="w-6 h-6 mr-2" />
          Tesseract Pattern Visualizer
        </h2>
        <div className="flex items-center space-x-4">
          <div className={`flex items-center ${getZpeZoneColor(patternState.system.zpe_zone)}`}>
            <Activity className="w-6 h-6 mr-2" />
            <span>ZPE: {patternState.system.zpe_zone}</span>
          </div>
          <div className="flex items-center">
            <Gpu className="w-6 h-6 mr-2" />
            <span>{patternState.system.gpu_utilization.toFixed(1)}%</span>
          </div>
          <div className="flex items-center">
            <Cpu className="w-6 h-6 mr-2" />
            <span>{patternState.system.cpu_utilization.toFixed(1)}%</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="border border-gray-700 rounded-lg p-4">
          <div className="text-lg font-bold mb-4 flex items-center">
            <Grid className="w-5 h-5 mr-2" />
            Dimensional Analysis
          </div>
          <div className="space-y-3">
            {patternState.dimensions.map((value, index) => (
              <div key={index}>
                <div className="text-gray-400 mb-1">Dimension {index + 1}</div>
                <div className="flex items-center">
                  <div className="flex-1">
                    <div className="h-2 bg-gray-700 rounded-full">
                      <div 
                        className="h-2 bg-blue-400 rounded-full transition-all duration-500"
                        style={{ width: `${(value / 15) * 100}%` }}
                      />
                    </div>
                  </div>
                  <span className="ml-2 text-blue-400">{value.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="border border-gray-700 rounded-lg p-4">
          <div className="text-lg font-bold mb-4 flex items-center">
            <Zap className="w-5 h-5 mr-2" />
            System Metrics
          </div>
          <div className="space-y-3">
            <div>
              <div className="text-gray-400 mb-1">Drift Band</div>
              <div className="flex items-center">
                <div className="flex-1">
                  <div className="h-2 bg-gray-700 rounded-full">
                    <div 
                      className={`h-2 rounded-full transition-all duration-500 ${
                        patternState.system.drift_band > 0.4 ? 'bg-red-400' :
                        patternState.system.drift_band > 0.2 ? 'bg-yellow-400' :
                        'bg-green-400'
                      }`}
                      style={{ width: `${patternState.system.drift_band * 100}%` }}
                    />
                  </div>
                </div>
                <span className="ml-2 text-blue-400">
                  {(patternState.system.drift_band * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <div>
              <div className="text-gray-400 mb-1">GPU Temperature</div>
              <div className="flex items-center">
                <div className="flex-1">
                  <div className="h-2 bg-gray-700 rounded-full">
                    <div 
                      className={`h-2 rounded-full transition-all duration-500 ${
                        patternState.system.gpu_temperature > 75 ? 'bg-red-400' :
                        patternState.system.gpu_temperature > 65 ? 'bg-yellow-400' :
                        'bg-green-400'
                      }`}
                      style={{ width: `${(patternState.system.gpu_temperature / 85) * 100}%` }}
                    />
                  </div>
                </div>
                <span className="ml-2 text-blue-400">
                  {patternState.system.gpu_temperature.toFixed(1)}°C
                </span>
              </div>
            </div>
            <div>
              <div className="text-gray-400 mb-1">Memory Usage</div>
              <div className="flex items-center">
                <div className="flex-1">
                  <div className="h-2 bg-gray-700 rounded-full">
                    <div 
                      className={`h-2 rounded-full transition-all duration-500 ${
                        patternState.system.memory_usage > 85 ? 'bg-red-400' :
                        patternState.system.memory_usage > 75 ? 'bg-yellow-400' :
                        'bg-green-400'
                      }`}
                      style={{ width: `${patternState.system.memory_usage}%` }}
                    />
                  </div>
                </div>
                <span className="ml-2 text-blue-400">
                  {patternState.system.memory_usage.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TesseractVisualizer; 