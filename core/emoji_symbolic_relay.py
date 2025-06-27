# -*- coding: utf-8 -*-
"""
Emoji Symbolic Relay System - 256-bit Ferris RDE Hash Integration

Implements symbolic relay system for connecting multiple states to 256-bit 
Ferris RDE hashes with deterministic path routing and profit vectorization.

Mathematical Foundation:
- Relay path: path_hash = Σ(hash_signature[:4] for each symbol)
- Ferris RDE hash: H_ferris = SHA256(path_hash)
- Symbol registration: relay_key = emoji + hash_signature[:8]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
import hashlib
import logging
import time

from core.asic_logic_gate_foundation import ASICLogicGate, get_asic_gate_manager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RelayPath:
    """Represents a symbolic relay path"""
    path_id: str
    symbols: List[str]
    path_hash: str
    ferris_hash: str
    profit_score: float
    timestamp: float
    active: bool = True


@dataclass
class SymbolRegistration:
    """Represents a registered symbol in the relay system"""
    emoji: str
    logic_gate: ASICLogicGate
    relay_key: str
    registration_time: float
    usage_count: int = 0
    last_used: float = 0.0


class EmojiSymbolicRelay:
    """Symbolic relay system for connecting multiple states to 256 Ferris RDE hash"""
    
    def __init__(self):
        self.symbol_registry: Dict[str, SymbolRegistration] = {}
        self.relay_paths: Dict[str, RelayPath] = {}
        self.ferris_hash_map: Dict[str, str] = {}
        self.active_paths: Set[str] = set()
        self.path_history: List[RelayPath] = []
        
        # Performance tracking
        self.total_relays_created = 0
        self.total_symbols_registered = 0
        self.average_profit_score = 0.0
    
    def register_symbol(self, emoji: str, logic_gate: ASICLogicGate) -> str:
        """Register emoji symbol with logic gate"""
        try:
            relay_key = f"{emoji}_{logic_gate.hash_signature[:8]}"
            
            # Check if already registered
            if relay_key in self.symbol_registry:
                # Update usage count
                registration = self.symbol_registry[relay_key]
                registration.usage_count += 1
                registration.last_used = time.time()
                return relay_key
            
            # Create new registration
            registration = SymbolRegistration(
                emoji=emoji,
                logic_gate=logic_gate,
                relay_key=relay_key,
                registration_time=time.time(),
                usage_count=1,
                last_used=time.time()
            )
            
            self.symbol_registry[relay_key] = registration
            self.total_symbols_registered += 1
            
            logger.info(f"Registered symbol: {emoji} -> {relay_key}")
            return relay_key
            
        except Exception as e:
            logger.error(f"Failed to register symbol {emoji}: {e}")
            return ""
    
    def create_relay_path(self, symbols: List[str]) -> str:
        """Create relay path connecting multiple symbols"""
        try:
            if not symbols:
                logger.warning("No symbols provided for relay path")
                return ""
            
            # Build path hash from registered symbols
            path_hash = ""
            registered_symbols = []
            
            for symbol in symbols:
                # Find symbol in registry
                symbol_found = False
                for relay_key, registration in self.symbol_registry.items():
                    if registration.emoji == symbol:
                        path_hash += registration.logic_gate.hash_signature[:4]
                        registered_symbols.append(symbol)
                        symbol_found = True
                        break
                
                if not symbol_found:
                    logger.warning(f"Symbol {symbol} not found in registry")
            
            if not path_hash:
                logger.error("No valid symbols found for relay path")
                return ""
            
            # Generate 256-bit Ferris RDE hash
            ferris_hash = hashlib.sha256(path_hash.encode()).hexdigest()
            
            # Create relay path
            path_id = f"path_{int(time.time())}_{len(self.relay_paths)}"
            relay_path = RelayPath(
                path_id=path_id,
                symbols=registered_symbols,
                path_hash=path_hash,
                ferris_hash=ferris_hash,
                profit_score=self._calculate_path_profit_score(registered_symbols),
                timestamp=time.time()
            )
            
            # Store relay path
            self.relay_paths[path_id] = relay_path
            self.ferris_hash_map[path_hash] = ferris_hash
            self.active_paths.add(path_id)
            self.path_history.append(relay_path)
            
            self.total_relays_created += 1
            self._update_average_profit_score()
            
            logger.info(f"Created relay path: {path_id} -> {ferris_hash[:16]}...")
            return ferris_hash
            
        except Exception as e:
            logger.error(f"Failed to create relay path: {e}")
            return ""
    
    def _calculate_path_profit_score(self, symbols: List[str]) -> float:
        """Calculate profit score for a relay path"""
        try:
            total_profit = 0.0
            symbol_count = 0
            
            for symbol in symbols:
                for registration in self.symbol_registry.values():
                    if registration.emoji == symbol:
                        total_profit += registration.logic_gate.profit_vector
                        symbol_count += 1
                        break
            
            return total_profit / symbol_count if symbol_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate path profit score: {e}")
            return 0.0
    
    def _update_average_profit_score(self):
        """Update average profit score across all paths"""
        try:
            if self.path_history:
                scores = [path.profit_score for path in self.path_history]
                self.average_profit_score = sum(scores) / len(scores)
        except Exception as e:
            logger.error(f"Failed to update average profit score: {e}")
    
    def get_relay_path(self, path_id: str) -> Optional[RelayPath]:
        """Get relay path by ID"""
        return self.relay_paths.get(path_id)
    
    def get_ferris_hash(self, path_hash: str) -> Optional[str]:
        """Get Ferris hash for a path hash"""
        return self.ferris_hash_map.get(path_hash)
    
    def activate_path(self, path_id: str) -> bool:
        """Activate a relay path"""
        if path_id in self.relay_paths:
            self.relay_paths[path_id].active = True
            self.active_paths.add(path_id)
            logger.info(f"Activated relay path: {path_id}")
            return True
        return False
    
    def deactivate_path(self, path_id: str) -> bool:
        """Deactivate a relay path"""
        if path_id in self.relay_paths:
            self.relay_paths[path_id].active = False
            self.active_paths.discard(path_id)
            logger.info(f"Deactivated relay path: {path_id}")
            return True
        return False
    
    def get_active_paths(self) -> List[RelayPath]:
        """Get all active relay paths"""
        return [self.relay_paths[path_id] for path_id in self.active_paths 
                if path_id in self.relay_paths]
    
    def get_symbol_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered symbols"""
        try:
            emoji_counts = {}
            gate_type_counts = {}
            bit_state_counts = {}
            
            for registration in self.symbol_registry.values():
                emoji = registration.emoji
                gate_type = registration.logic_gate.gate_type.value
                bit_state = registration.logic_gate.bit_state
                
                emoji_counts[emoji] = emoji_counts.get(emoji, 0) + 1
                gate_type_counts[gate_type] = gate_type_counts.get(gate_type, 0) + 1
                bit_state_counts[bit_state] = bit_state_counts.get(bit_state, 0) + 1
            
            return {
                "total_symbols": len(self.symbol_registry),
                "emoji_distribution": emoji_counts,
                "gate_type_distribution": gate_type_counts,
                "bit_state_distribution": bit_state_counts,
                "average_usage_count": sum(r.usage_count for r in self.symbol_registry.values()) / len(self.symbol_registry) if self.symbol_registry else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get symbol statistics: {e}")
            return {}
    
    def get_path_statistics(self) -> Dict[str, Any]:
        """Get statistics about relay paths"""
        try:
            return {
                "total_paths": len(self.relay_paths),
                "active_paths": len(self.active_paths),
                "average_profit_score": self.average_profit_score,
                "total_relays_created": self.total_relays_created,
                "path_length_distribution": {
                    "short": sum(1 for p in self.relay_paths.values() if len(p.symbols) <= 2),
                    "medium": sum(1 for p in self.relay_paths.values() if 3 <= len(p.symbols) <= 5),
                    "long": sum(1 for p in self.relay_paths.values() if len(p.symbols) > 5)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get path statistics: {e}")
            return {}
    
    def clear_inactive_paths(self) -> int:
        """Clear inactive paths and return count of cleared paths"""
        inactive_paths = [path_id for path_id, path in self.relay_paths.items() 
                         if not path.active]
        
        for path_id in inactive_paths:
            del self.relay_paths[path_id]
            self.active_paths.discard(path_id)
        
        logger.info(f"Cleared {len(inactive_paths)} inactive paths")
        return len(inactive_paths)
    
    def export_relay_data(self) -> Dict[str, Any]:
        """Export relay system data for persistence"""
        try:
            return {
                "symbol_registry": {
                    key: {
                        "emoji": reg.emoji,
                        "gate_type": reg.logic_gate.gate_type.value,
                        "bit_state": reg.logic_gate.bit_state,
                        "hash_signature": reg.logic_gate.hash_signature,
                        "profit_vector": reg.logic_gate.profit_vector,
                        "registration_time": reg.registration_time,
                        "usage_count": reg.usage_count,
                        "last_used": reg.last_used
                    }
                    for key, reg in self.symbol_registry.items()
                },
                "relay_paths": {
                    path_id: {
                        "symbols": path.symbols,
                        "path_hash": path.path_hash,
                        "ferris_hash": path.ferris_hash,
                        "profit_score": path.profit_score,
                        "timestamp": path.timestamp,
                        "active": path.active
                    }
                    for path_id, path in self.relay_paths.items()
                },
                "statistics": {
                    "total_symbols_registered": self.total_symbols_registered,
                    "total_relays_created": self.total_relays_created,
                    "average_profit_score": self.average_profit_score
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to export relay data: {e}")
            return {}


# Global Emoji Symbolic Relay instance
emoji_relay = EmojiSymbolicRelay()


def get_emoji_relay() -> EmojiSymbolicRelay:
    """Get global emoji symbolic relay instance"""
    return emoji_relay


def register_symbol_with_relay(emoji: str, logic_gate: ASICLogicGate) -> str:
    """Register a symbol with the relay system"""
    return emoji_relay.register_symbol(emoji, logic_gate)


def create_relay_path(symbols: List[str]) -> str:
    """Create a relay path with the given symbols"""
    return emoji_relay.create_relay_path(symbols)


def get_relay_statistics() -> Dict[str, Any]:
    """Get comprehensive relay system statistics"""
    return {
        "symbols": emoji_relay.get_symbol_statistics(),
        "paths": emoji_relay.get_path_statistics()
    } 