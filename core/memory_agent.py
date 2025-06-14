"""
Memory Agent System
==================

Implements the memory agent that stores strategy successes, learns from
profitable patterns, and provides confidence coefficients to the trading logic.
This system connects to the shared memory map and integrates with the
thermal-aware processing pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import logging
import threading
import json
from enum import Enum
from .memory_map import get_memory_map, MemoryMap
from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor, ProfitZoneState, TrajectoryVector
from .thermal_zone_manager import ThermalZoneManager, ThermalState
from .hash_trigger_engine import HashTrigger, HashTriggerEngine

# Setup logging
logger = logging.getLogger(__name__)

class StrategyState(Enum):
    """Strategy execution states"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    LEARNING = "learning"

@dataclass
class StrategyExecution:
    """Container for strategy execution data"""
    strategy_id: str
    execution_id: str
    hash_triggers: List[str]
    entry_price: float
    exit_price: Optional[float]
    profit_loss: Optional[float]
    confidence_used: float
    thermal_state: Optional[str]
    profit_zone: Optional[str]
    execution_time: float
    state: StrategyState
    timestamp: datetime
    metadata: Dict

@dataclass
class AgentMemory:
    """Container for agent memory state"""
    agent_id: str
    strategy_successes: Dict[str, List[StrategyExecution]]
    confidence_coefficients: Dict[str, float]
    learning_weights: Dict[str, float]
    total_executions: int
    successful_executions: int
    average_profit: float
    last_updated: datetime

class MemoryAgent:
    """
    Memory agent that learns from strategy executions and provides
    intelligent confidence coefficients for future trades.
    """
    
    def __init__(self, agent_id: str, 
                 memory_map: Optional[MemoryMap] = None,
                 profit_coprocessor: Optional[ProfitTrajectoryCoprocessor] = None,
                 thermal_manager: Optional[ThermalZoneManager] = None):
        """
        Initialize memory agent
        
        Args:
            agent_id: Unique identifier for this agent
            memory_map: Shared memory map instance
            profit_coprocessor: Profit trajectory coprocessor for context
            thermal_manager: Thermal zone manager for context
        """
        self.agent_id = agent_id
        self.memory_map = memory_map or get_memory_map()
        self.profit_coprocessor = profit_coprocessor
        self.thermal_manager = thermal_manager
        
        # Agent state
        self.agent_memory = self._initialize_agent_memory()
        self.execution_history: List[StrategyExecution] = []
        self.active_executions: Dict[str, StrategyExecution] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_decay = 0.95  # Decay factor for outdated patterns
        self.success_threshold = 0.0  # Minimum profit to be considered success
        self.pattern_similarity_threshold = 0.8
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing data
        self._load_agent_state()
        
    def _initialize_agent_memory(self) -> AgentMemory:
        """Initialize agent memory structure"""
        return AgentMemory(
            agent_id=self.agent_id,
            strategy_successes={},
            confidence_coefficients={},
            learning_weights={},
            total_executions=0,
            successful_executions=0,
            average_profit=0.0,
            last_updated=datetime.now(timezone.utc)
        )
        
    def _load_agent_state(self) -> None:
        """Load agent state from shared memory map"""
        try:
            active_agents = self.memory_map.get("active_agents", {})
            if self.agent_id in active_agents:
                agent_data = active_agents[self.agent_id]
                
                # Reconstruct agent memory
                self.agent_memory.strategy_successes = agent_data.get("strategy_successes", {})
                self.agent_memory.confidence_coefficients = agent_data.get("confidence_coefficients", {})
                self.agent_memory.learning_weights = agent_data.get("learning_weights", {})
                self.agent_memory.total_executions = agent_data.get("total_executions", 0)
                self.agent_memory.successful_executions = agent_data.get("successful_executions", 0)
                self.agent_memory.average_profit = agent_data.get("average_profit", 0.0)
                
                logger.info(f"Loaded agent {self.agent_id} with {self.agent_memory.total_executions} executions")
            else:
                logger.info(f"Initializing new agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")
            
    def _save_agent_state(self) -> None:
        """Save agent state to shared memory map"""
        try:
            with self._lock:
                active_agents = self.memory_map.get("active_agents", {})
                active_agents[self.agent_id] = asdict(self.agent_memory)
                self.memory_map.set("active_agents", active_agents)
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
            
    def start_strategy_execution(self, strategy_id: str, hash_triggers: List[str],
                               entry_price: float, initial_confidence: float) -> str:
        """
        Start tracking a new strategy execution
        
        Args:
            strategy_id: Strategy identifier
            hash_triggers: List of hash trigger IDs that fired
            entry_price: Entry price for the trade
            initial_confidence: Initial confidence score
            
        Returns:
            Execution ID for tracking
        """
        execution_id = f"{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Get current context
        thermal_state = None
        profit_zone = None
        
        if self.thermal_manager and self.thermal_manager.current_state:
            thermal_state = self.thermal_manager.current_state.zone.value
            
        if self.profit_coprocessor and self.profit_coprocessor.last_vector:
            profit_zone = self.profit_coprocessor.last_vector.zone_state.value
            
        # Create execution record
        execution = StrategyExecution(
            strategy_id=strategy_id,
            execution_id=execution_id,
            hash_triggers=hash_triggers.copy(),
            entry_price=entry_price,
            exit_price=None,
            profit_loss=None,
            confidence_used=initial_confidence,
            thermal_state=thermal_state,
            profit_zone=profit_zone,
            execution_time=0.0,
            state=StrategyState.ACTIVE,
            timestamp=datetime.now(timezone.utc),
            metadata={}
        )
        
        with self._lock:
            self.active_executions[execution_id] = execution
            self.execution_history.append(execution)
            
        logger.info(f"Started execution {execution_id} for strategy {strategy_id}")
        return execution_id
        
    def complete_strategy_execution(self, execution_id: str, exit_price: float,
                                  execution_time: float, metadata: Optional[Dict] = None) -> None:
        """
        Complete a strategy execution and learn from the result
        
        Args:
            execution_id: Execution ID from start_strategy_execution
            exit_price: Exit price for the trade
            execution_time: Total execution time in seconds
            metadata: Additional metadata about the execution
        """
        if execution_id not in self.active_executions:
            logger.warning(f"Execution {execution_id} not found in active executions")
            return
            
        with self._lock:
            execution = self.active_executions[execution_id]
            
            # Calculate profit/loss
            profit_loss = exit_price - execution.entry_price
            
            # Update execution record
            execution.exit_price = exit_price
            execution.profit_loss = profit_loss
            execution.execution_time = execution_time
            execution.state = StrategyState.COMPLETED if profit_loss >= self.success_threshold else StrategyState.FAILED
            execution.metadata = metadata or {}
            
            # Remove from active executions
            del self.active_executions[execution_id]
            
            # Learn from this execution
            self._process_execution_result(execution)
            
            # Update agent memory stats
            self.agent_memory.total_executions += 1
            if execution.state == StrategyState.COMPLETED:
                self.agent_memory.successful_executions += 1
                
            # Update average profit
            total_profit = (self.agent_memory.average_profit * (self.agent_memory.total_executions - 1) + 
                          profit_loss)
            self.agent_memory.average_profit = total_profit / self.agent_memory.total_executions
            
            self.agent_memory.last_updated = datetime.now(timezone.utc)
            
            # Save to shared memory if it was a success
            if execution.state == StrategyState.COMPLETED:
                self._echo_success_to_memory(execution)
                
            # Save agent state
            self._save_agent_state()
            
        logger.info(f"Completed execution {execution_id}: "
                   f"P&L={profit_loss:.2f}, State={execution.state.value}")
        
    def _process_execution_result(self, execution: StrategyExecution) -> None:
        """Process execution result and update learning weights"""
        strategy_id = execution.strategy_id
        
        # Initialize strategy tracking if new
        if strategy_id not in self.agent_memory.strategy_successes:
            self.agent_memory.strategy_successes[strategy_id] = []
            self.agent_memory.confidence_coefficients[strategy_id] = 0.5  # Neutral start
            self.agent_memory.learning_weights[strategy_id] = 1.0
            
        # Add to strategy successes
        self.agent_memory.strategy_successes[strategy_id].append(execution)
        
        # Keep only last 100 executions per strategy
        if len(self.agent_memory.strategy_successes[strategy_id]) > 100:
            self.agent_memory.strategy_successes[strategy_id] = \
                self.agent_memory.strategy_successes[strategy_id][-100:]
                
        # Update confidence coefficient based on recent performance
        self._update_confidence_coefficient(strategy_id)
        
    def _update_confidence_coefficient(self, strategy_id: str) -> None:
        """Update confidence coefficient for a strategy based on recent performance"""
        executions = self.agent_memory.strategy_successes[strategy_id]
        if not executions:
            return
            
        # Calculate recent success rate (last 20 executions)
        recent_executions = executions[-20:]
        successful = sum(1 for ex in recent_executions if ex.state == StrategyState.COMPLETED)
        success_rate = successful / len(recent_executions)
        
        # Calculate average profit for successful trades
        successful_profits = [ex.profit_loss for ex in recent_executions 
                            if ex.state == StrategyState.COMPLETED and ex.profit_loss]
        avg_profit = np.mean(successful_profits) if successful_profits else 0.0
        
        # Calculate confidence based on success rate and profit magnitude
        base_confidence = success_rate
        profit_multiplier = 1.0 + np.tanh(avg_profit / 100.0)  # Normalized profit boost
        new_confidence = base_confidence * profit_multiplier
        
        # Apply learning rate for smooth updates
        current_confidence = self.agent_memory.confidence_coefficients[strategy_id]
        updated_confidence = (current_confidence * (1 - self.learning_rate) + 
                            new_confidence * self.learning_rate)
        
        # Clamp to reasonable range
        self.agent_memory.confidence_coefficients[strategy_id] = np.clip(updated_confidence, 0.1, 1.0)
        
        # Update in global memory map
        self.memory_map.update_confidence_coefficient(
            f"{self.agent_id}_{strategy_id}",
            self.agent_memory.confidence_coefficients[strategy_id]
        )
        
    def _echo_success_to_memory(self, execution: StrategyExecution) -> None:
        """Echo successful execution to shared memory map"""
        success_data = {
            "agent_id": self.agent_id,
            "strategy_id": execution.strategy_id,
            "profit_loss": execution.profit_loss,
            "confidence_used": execution.confidence_used,
            "hash_triggers": execution.hash_triggers,
            "thermal_state": execution.thermal_state,
            "profit_zone": execution.profit_zone,
            "execution_time": execution.execution_time,
            "entry_price": execution.entry_price,
            "exit_price": execution.exit_price
        }
        
        self.memory_map.add_strategy_success(execution.strategy_id, success_data)
        
    def get_confidence_coefficient(self, strategy_id: str, 
                                 hash_triggers: Optional[List[str]] = None,
                                 current_context: Optional[Dict] = None) -> float:
        """
        Get confidence coefficient for a strategy execution
        
        Args:
            strategy_id: Strategy identifier
            hash_triggers: Hash triggers that fired (for pattern matching)
            current_context: Current market/system context
            
        Returns:
            Confidence coefficient (0.0 to 1.0)
        """
        base_confidence = self.agent_memory.confidence_coefficients.get(strategy_id, 0.5)
        
        # Adjust based on similar past executions
        if hash_triggers and strategy_id in self.agent_memory.strategy_successes:
            similarity_boost = self._calculate_pattern_similarity_boost(
                strategy_id, hash_triggers, current_context
            )
            base_confidence *= (1.0 + similarity_boost)
            
        # Adjust based on current thermal/profit state
        if current_context:
            context_modifier = self._calculate_context_modifier(current_context)
            base_confidence *= context_modifier
            
        # Apply confidence decay for strategies not used recently
        if strategy_id in self.agent_memory.strategy_successes:
            last_execution = self.agent_memory.strategy_successes[strategy_id][-1]
            days_since = (datetime.now(timezone.utc) - last_execution.timestamp).days
            decay_factor = self.confidence_decay ** days_since
            base_confidence *= decay_factor
            
        return np.clip(base_confidence, 0.1, 1.0)
        
    def _calculate_pattern_similarity_boost(self, strategy_id: str, 
                                          hash_triggers: List[str],
                                          current_context: Optional[Dict]) -> float:
        """Calculate boost based on pattern similarity to past successes"""
        executions = self.agent_memory.strategy_successes[strategy_id]
        successful_executions = [ex for ex in executions 
                               if ex.state == StrategyState.COMPLETED]
        
        if not successful_executions:
            return 0.0
            
        # Calculate similarity to successful patterns
        similarities = []
        for execution in successful_executions[-10:]:  # Last 10 successful
            trigger_similarity = self._calculate_trigger_similarity(
                hash_triggers, execution.hash_triggers
            )
            context_similarity = self._calculate_context_similarity(
                current_context, execution
            )
            
            combined_similarity = (trigger_similarity + context_similarity) / 2.0
            if combined_similarity > self.pattern_similarity_threshold:
                similarities.append(combined_similarity)
                
        if similarities:
            avg_similarity = np.mean(similarities)
            return (avg_similarity - self.pattern_similarity_threshold) * 0.5
        else:
            return -0.1  # Slight penalty for no similar patterns
            
    def _calculate_trigger_similarity(self, triggers1: List[str], 
                                    triggers2: List[str]) -> float:
        """Calculate similarity between two sets of hash triggers"""
        set1 = set(triggers1)
        set2 = set(triggers2)
        
        if not set1 and not set2:
            return 1.0
        elif not set1 or not set2:
            return 0.0
        else:
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union  # Jaccard similarity
            
    def _calculate_context_similarity(self, current_context: Optional[Dict],
                                    execution: StrategyExecution) -> float:
        """Calculate similarity between current context and past execution context"""
        if not current_context:
            return 0.5  # Neutral if no context
            
        similarity_score = 0.0
        factors = 0
        
        # Compare thermal states
        if ("thermal_state" in current_context and execution.thermal_state):
            if current_context["thermal_state"] == execution.thermal_state:
                similarity_score += 1.0
            factors += 1
            
        # Compare profit zones
        if ("profit_zone" in current_context and execution.profit_zone):
            if current_context["profit_zone"] == execution.profit_zone:
                similarity_score += 1.0
            factors += 1
            
        return similarity_score / factors if factors > 0 else 0.5
        
    def _calculate_context_modifier(self, context: Dict) -> float:
        """Calculate context-based confidence modifier"""
        modifier = 1.0
        
        # Thermal state modifier
        thermal_state = context.get("thermal_state")
        if thermal_state:
            thermal_modifiers = {
                "cool": 1.1,
                "normal": 1.0,
                "warm": 0.95,
                "hot": 0.8,
                "critical": 0.5
            }
            modifier *= thermal_modifiers.get(thermal_state, 1.0)
            
        # Profit zone modifier
        profit_zone = context.get("profit_zone")
        if profit_zone:
            profit_modifiers = {
                "surging": 1.2,
                "stable": 1.0,
                "drawdown": 0.8,
                "volatile": 0.9,
                "unknown": 1.0
            }
            modifier *= profit_modifiers.get(profit_zone, 1.0)
            
        return np.clip(modifier, 0.5, 1.5)
        
    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Union[float, int]]:
        """Get performance statistics for a strategy"""
        if strategy_id not in self.agent_memory.strategy_successes:
            return {"error": "strategy_not_found"}
            
        executions = self.agent_memory.strategy_successes[strategy_id]
        successful = [ex for ex in executions if ex.state == StrategyState.COMPLETED]
        
        if not executions:
            return {"error": "no_executions"}
            
        total_profit = sum(ex.profit_loss for ex in executions if ex.profit_loss)
        successful_profit = sum(ex.profit_loss for ex in successful if ex.profit_loss)
        
        return {
            "total_executions": len(executions),
            "successful_executions": len(successful),
            "success_rate": len(successful) / len(executions),
            "total_profit": total_profit,
            "average_profit": total_profit / len(executions),
            "successful_average_profit": successful_profit / len(successful) if successful else 0.0,
            "confidence_coefficient": self.agent_memory.confidence_coefficients[strategy_id],
            "last_execution": executions[-1].timestamp.isoformat() if executions else None
        }
        
    def get_agent_statistics(self) -> Dict[str, Union[float, int, str]]:
        """Get overall agent statistics"""
        return {
            "agent_id": self.agent_id,
            "total_executions": self.agent_memory.total_executions,
            "successful_executions": self.agent_memory.successful_executions,
            "success_rate": (self.agent_memory.successful_executions / 
                           self.agent_memory.total_executions 
                           if self.agent_memory.total_executions > 0 else 0.0),
            "average_profit": self.agent_memory.average_profit,
            "active_executions": len(self.active_executions),
            "tracked_strategies": len(self.agent_memory.strategy_successes),
            "last_updated": self.agent_memory.last_updated.isoformat()
        }
        
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old execution data"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        with self._lock:
            # Clean execution history
            self.execution_history = [
                ex for ex in self.execution_history 
                if ex.timestamp > cutoff_date
            ]
            
            # Clean strategy successes
            for strategy_id in self.agent_memory.strategy_successes:
                self.agent_memory.strategy_successes[strategy_id] = [
                    ex for ex in self.agent_memory.strategy_successes[strategy_id]
                    if ex.timestamp > cutoff_date
                ]
                
            self._save_agent_state()
            
        logger.info(f"Cleaned up data older than {days_to_keep} days")

# Example usage and testing
if __name__ == "__main__":
    from .profit_trajectory_coprocessor import ProfitTrajectoryCoprocessor
    from .thermal_zone_manager import ThermalZoneManager
    
    # Create supporting components
    profit_coprocessor = ProfitTrajectoryCoprocessor()
    thermal_manager = ThermalZoneManager(profit_coprocessor)
    
    # Add some sample profit data
    for i in range(10):
        profit = 1000 + i * 20 + np.random.normal(0, 10)
        profit_coprocessor.update(profit)
        
    thermal_manager.update_thermal_state()
    
    # Create memory agent
    agent = MemoryAgent(
        agent_id="test_agent_001",
        profit_coprocessor=profit_coprocessor,
        thermal_manager=thermal_manager
    )
    
    print("Memory Agent Test:")
    print(f"Agent ID: {agent.agent_id}")
    
    # Test strategy execution
    execution_id = agent.start_strategy_execution(
        strategy_id="test_strategy",
        hash_triggers=["TRIGGER_001", "TRIGGER_002"],
        entry_price=100.0,
        initial_confidence=0.8
    )
    
    print(f"Started execution: {execution_id}")
    
    # Simulate completion
    agent.complete_strategy_execution(
        execution_id=execution_id,
        exit_price=105.0,  # Profitable trade
        execution_time=30.5,
        metadata={"test": True}
    )
    
    print("Completed execution")
    
    # Test confidence coefficient
    confidence = agent.get_confidence_coefficient(
        strategy_id="test_strategy",
        hash_triggers=["TRIGGER_001", "TRIGGER_002"],
        current_context={
            "thermal_state": "normal",
            "profit_zone": "stable"
        }
    )
    
    print(f"Confidence coefficient: {confidence:.3f}")
    
    # Get statistics
    stats = agent.get_agent_statistics()
    print("\nAgent Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    performance = agent.get_strategy_performance("test_strategy")
    print("\nStrategy Performance:")
    for key, value in performance.items():
        print(f"  {key}: {value}") 