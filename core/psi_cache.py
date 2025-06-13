"""
Ψ Replay Cache
=============

Implements a rolling window cache for Ψ(t) values to enable
hot-start reentry and pattern memory persistence.
"""

from typing import List, Optional
import json
from pathlib import Path
import time

class PsiCache:
    def __init__(self, window_size: int = 100, cache_file: str = "psi_cache.json"):
        self.window_size = window_size
        self.cache_file = Path(__file__).resolve().parent / "cache" / cache_file
        self.cache_file.parent.mkdir(exist_ok=True)
        self.psi_window: List[float] = []
        self._load_cache()
        
    def _load_cache(self):
        """Load cached Ψ values if available"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.psi_window = data.get('psi_values', [])[-self.window_size:]
            except Exception as e:
                print(f"Warning: Could not load psi cache: {e}")
                self.psi_window = []
                
    def _save_cache(self):
        """Save current Ψ window to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'psi_values': self.psi_window
                }, f)
        except Exception as e:
            print(f"Warning: Could not save psi cache: {e}")
            
    def add_psi(self, psi_value: float):
        """Add new Ψ value to window"""
        self.psi_window.append(psi_value)
        if len(self.psi_window) > self.window_size:
            self.psi_window.pop(0)
        self._save_cache()
        
    def get_psi_window(self) -> List[float]:
        """Get current Ψ window"""
        return self.psi_window.copy()
        
    def get_last_psi(self) -> Optional[float]:
        """Get most recent Ψ value"""
        return self.psi_window[-1] if self.psi_window else None
        
    def clear(self):
        """Clear the cache"""
        self.psi_window.clear()
        self._save_cache() 