"""
Paradox Visualizer - TPF (Triangle Paradox Fractal) visualization engine.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ParadoxState:
    """Current state of the paradox visualization."""
    phase: int = 0
    stabilized: bool = False
    paradox_visible: bool = False
    trading_mode: bool = False
    glyph_state: str = "INITIALIZING"
    detonation_protocol: bool = False
    timestamp: float = 0.0

@dataclass
class MarketData:
    """Market data for visualization."""
    price: float = 0.0
    volume: float = 0.0
    rsi: float = 50.0
    drift: float = 0.0
    entropy: float = 0.5

class ParadoxVisualizer:
    """Core paradox visualization engine."""
    
    def __init__(self):
        self.state = ParadoxState()
        self.market_data = MarketData()
        self.trading_signals = []
        self.tpf_fractals = []
        self.last_update = datetime.now()
        
    def update_state(self, new_phase: int, market_data: Optional[Dict] = None) -> ParadoxState:
        """
        Update the paradox visualization state.
        """
        self.state.phase = new_phase % 100
        
        # Update market data if provided
        if market_data:
            self.market_data = MarketData(**market_data)
        
        # Paradox detection at phase 30
        if self.state.phase == 30 and not self.state.paradox_visible:
            self.state.paradox_visible = True
            self.state.glyph_state = "PARADOX DETECTED"
        
        # TPF stabilization at phase 70
        if self.state.phase == 70 and not self.state.stabilized:
            self.state.stabilized = True
            self.state.glyph_state = "TPF STABILIZED"
        
        # Reset cycle at phase 99
        if self.state.phase == 99:
            self.state.paradox_visible = False
            self.state.stabilized = False
            self.state.glyph_state = "INITIALIZING"
        
        self.state.timestamp = datetime.now().timestamp()
        return self.state
    
    def trigger_detonation(self) -> List[Dict]:
        """
        Trigger the detonation protocol and generate trading signals.
        """
        self.state.detonation_protocol = True
        self.state.trading_mode = True
        
        # Generate trading signals based on recursive paradox state
        signals = []
        for i in range(5):
            signal = {
                'id': int(datetime.now().timestamp() * 1000) + i,
                'type': 'SELL' if self.market_data.rsi > 70 else 'BUY' if self.market_data.rsi < 30 else 'HOLD',
                'confidence': 0.95 if self.state.stabilized else 0.4 if self.state.paradox_visible else 0.7,
                'price': self.market_data.price + (np.random.random() - 0.5) * 200,
                'timestamp': datetime.now().timestamp()
            }
            signals.append(signal)
        
        self.trading_signals = signals
        return signals
    
    def calculate_tpf_metrics(self) -> Dict:
        """
        Calculate TPF (Triangle Paradox Fractal) metrics.
        """
        return {
            'magnitude': np.sqrt(sum(x**2 for x in [self.market_data.price, self.market_data.volume, self.market_data.rsi])),
            'phase': np.arctan2(self.market_data.drift, self.market_data.entropy),
            'stability_score': 1.0 if self.state.stabilized else 0.5 if self.state.paradox_visible else 0.0,
            'paradox_intensity': 1.0 if self.state.paradox_visible else 0.0,
            'detonation_ready': self.state.detonation_protocol
        }
    
    def get_visualization_data(self) -> Dict:
        """
        Get complete visualization data for frontend rendering.
        """
        return {
            'state': {
                'phase': self.state.phase,
                'stabilized': self.state.stabilized,
                'paradox_visible': self.state.paradox_visible,
                'trading_mode': self.state.trading_mode,
                'glyph_state': self.state.glyph_state,
                'detonation_protocol': self.state.detonation_protocol
            },
            'market_data': {
                'price': self.market_data.price,
                'volume': self.market_data.volume,
                'rsi': self.market_data.rsi,
                'drift': self.market_data.drift,
                'entropy': self.market_data.entropy
            },
            'trading_signals': self.trading_signals,
            'tpf_metrics': self.calculate_tpf_metrics(),
            'timestamp': self.state.timestamp
        } 