"""
Pattern metrics for DLT Waveform Engine
"""

import numpy as np
from typing import List, Tuple
from statistics import stdev

class PatternMetrics:
    def __init__(self):
        self.entropy_window = 20  # Window size for entropy calculation
        self.coherence_threshold = 0.5  # Threshold for coherence calculation
        
    def entropy(self, data: List[float]) -> float:
        """Calculate Shannon entropy of the data"""
        if not data or len(data) < 2:
            return 0.0
            
        # Normalize data to [0,1] range
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        # Create histogram
        hist, _ = np.histogram(data_norm, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
        
    def coherence(self, data: List[float]) -> float:
        """Calculate pattern coherence using autocorrelation"""
        if not data or len(data) < 2:
            return 0.0
            
        # Calculate autocorrelation
        data_norm = (data - np.mean(data)) / np.std(data)
        autocorr = np.correlate(data_norm, data_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize and calculate coherence
        autocorr = autocorr / autocorr[0]
        coherence = np.mean(np.abs(autocorr[:self.entropy_window]))
        
        return coherence
        
    def get_entropy_and_coherence(self, pattern: List[float]) -> Tuple[float, float]:
        """Get both entropy and coherence metrics for a pattern"""
        return self.entropy(pattern), self.coherence(pattern)
        
    def detect_pattern(self, data: List[float], window_size: int = 3) -> List[str]:
        """Detect patterns in the data using sliding window"""
        if not data or len(data) < window_size:
            return []
            
        patterns = []
        for i in range(len(data) - window_size + 1):
            window = data[i:i+window_size]
            pattern = self._encode_pattern(window)
            patterns.append(pattern)
            
        return patterns
        
    def _encode_pattern(self, window: List[float]) -> str:
        """Encode a window of data into a pattern string"""
        if len(window) < 2:
            return "X"
            
        pattern = []
        for i in range(1, len(window)):
            if window[i] > window[i-1]:
                pattern.append("U")
            elif window[i] < window[i-1]:
                pattern.append("D")
            else:
                pattern.append("S")
                
        return "".join(pattern) 