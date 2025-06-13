from __future__ import annotations
import logging
from typing import Dict, Any, Optional
from collections import deque

from .recursive_strategy_handler import RecursiveStrategyHandler, TradeSignal
from .profit_memory_vault import ProfitMemoryVault, PatternMemory

logger = logging.getLogger(__name__)

class VaultReentryPipeline:
    """
    Integrates recursive strategy handling with profit pattern memory.
    Manages the complete trading pipeline from signal processing to execution.
    """
    
    def __init__(
        self,
        memory_window: int = 128,
        min_pattern_confidence: float = 0.8,
        max_position_size: float = 1.0,
        profit_threshold: float = 0.02
    ):
        # Initialize components
        self.strategy_handler = RecursiveStrategyHandler(
            min_confidence=min_pattern_confidence,
            max_position_size=max_position_size
        )
        
        self.profit_vault = ProfitMemoryVault(
            min_profit_threshold=profit_threshold
        )
        
        # State tracking
        self.memory_window = memory_window
        self.psi_buffer = deque(maxlen=memory_window)
        self.entropy_buffer = deque(maxlen=memory_window)
        self.coherence_buffer = deque(maxlen=memory_window)
        self.price_buffer = deque(maxlen=memory_window)
        
        # Active trade tracking
        self.active_trade: Optional[Dict[str, Any]] = None
        self.trade_start_psi: Optional[float] = None
        
        logger.info("VaultReentryPipeline initialized with %d tick memory window", memory_window)

    def process_tick(
        self,
        tick_data: Dict[str, Any]
    ) -> Optional[TradeSignal]:
        """
        Process a market tick through the complete pipeline.
        
        Parameters
        ----------
        tick_data : dict
            Market tick data including price and recursive metrics
            
        Returns
        -------
        Optional[TradeSignal]
            Trading decision if one is generated
        """
        # Extract data
        price = tick_data['price']
        psi = tick_data.get('psi', 0.0)
        entropy = tick_data.get('entropy', 0.0)
        coherence = tick_data.get('coherence', 0.0)
        
        # Update buffers
        self.psi_buffer.append(psi)
        self.entropy_buffer.append(entropy)
        self.coherence_buffer.append(coherence)
        self.price_buffer.append(price)
        
        # Check for active trade
        if self.active_trade is not None:
            return self._handle_active_trade(price, psi, entropy, coherence)
            
        # Look for pattern matches
        similar_patterns = self.profit_vault.find_similar_patterns(
            list(self.psi_buffer),
            list(self.entropy_buffer)
        )
        
        if similar_patterns:
            # We found similar patterns - use them to influence strategy
            best_pattern = similar_patterns[0]
            logger.info(
                "Found similar pattern %s with %.2f%% historical profit",
                best_pattern.pattern_id,
                best_pattern.profit_pct * 100
            )
            
            # Adjust confidence based on pattern similarity
            pattern_confidence = min(0.9, best_pattern.profit_pct * 2)
            
            # Get strategy decision
            signal = self.strategy_handler.process_signal(
                psi=psi,
                entropy=entropy,
                coherence=coherence,
                current_price=price,
                metadata={
                    'recursive_state': tick_data.get('recursive_state', 'TFF'),
                    'pattern_match': best_pattern.pattern_id,
                    'pattern_profit': best_pattern.profit_pct
                }
            )
            
            # If we got a trade signal, start tracking the trade
            if signal.action != 'HOLD':
                self.active_trade = {
                    'entry_price': price,
                    'entry_psi': psi,
                    'pattern_id': best_pattern.pattern_id,
                    'signal': signal
                }
                self.trade_start_psi = psi
                
            return signal
            
        else:
            # No pattern matches - use pure strategy
            return self.strategy_handler.process_signal(
                psi=psi,
                entropy=entropy,
                coherence=coherence,
                current_price=price,
                metadata={'recursive_state': tick_data.get('recursive_state', 'TFF')}
            )

    def _handle_active_trade(
        self,
        current_price: float,
        current_psi: float,
        current_entropy: float,
        current_coherence: float
    ) -> Optional[TradeSignal]:
        """Handle logic for an active trade."""
        if not self.active_trade:
            return None
            
        entry_price = self.active_trade['entry_price']
        entry_psi = self.active_trade['entry_psi']
        pattern_id = self.active_trade['pattern_id']
        
        # Calculate profit
        profit_pct = (current_price - entry_price) / entry_price
        
        # Check for exit conditions
        should_exit = False
        exit_reason = None
        
        # 1. Profit target hit
        if profit_pct >= 0.02:  # 2% profit target
            should_exit = True
            exit_reason = "profit_target"
            
        # 2. Stop loss hit
        elif profit_pct <= -0.01:  # 1% stop loss
            should_exit = True
            exit_reason = "stop_loss"
            
        # 3. Ψ divergence
        elif abs(current_psi - entry_psi) > 0.5:  # Significant Ψ divergence
            should_exit = True
            exit_reason = "psi_divergence"
            
        if should_exit:
            # Store the pattern if profitable
            if profit_pct > 0:
                self.profit_vault.store_pattern(
                    entry_price=entry_price,
                    exit_price=current_price,
                    psi_sequence=list(self.psi_buffer),
                    entropy_sequence=list(self.entropy_buffer),
                    coherence_sequence=list(self.coherence_buffer),
                    duration=len(self.psi_buffer),
                    metadata={
                        'pattern_id': pattern_id,
                        'exit_reason': exit_reason,
                        'profit_pct': profit_pct
                    }
                )
                
            # Clear active trade
            self.active_trade = None
            self.trade_start_psi = None
            
            # Return exit signal
            return TradeSignal(
                action='HOLD',  # Exit is handled by position management
                confidence=1.0,
                metadata={
                    'exit_reason': exit_reason,
                    'profit_pct': profit_pct
                }
            )
            
        return None

    def get_state(self) -> Dict[str, Any]:
        """Get current pipeline state."""
        return {
            'active_trade': self.active_trade is not None,
            'strategy_state': self.strategy_handler.get_state(),
            'vault_stats': self.profit_vault.get_pattern_stats(),
            'buffer_sizes': {
                'psi': len(self.psi_buffer),
                'entropy': len(self.entropy_buffer),
                'coherence': len(self.coherence_buffer),
                'price': len(self.price_buffer)
            }
        } 