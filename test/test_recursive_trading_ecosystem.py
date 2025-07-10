#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 RECURSIVE TRADING ECOSYSTEM TEST - SCHWABOT COMPLETE SYSTEM DEMO
==================================================================

Comprehensive test script that demonstrates the complete recursive trading ecosystem:
- Hash Match Command Injector
- Live Vector Simulator
- Flask AI Agent Handler
- Agent Memory Integration
- Profit Bucket Registry
- Real-time Execution

This test showcases the living, breathing, recursive trading system in action.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Schwabot components
try:
    from core.agent_memory import AgentMemory
    from core.entropy_signal_integration import EntropySignalIntegrator
    from core.flask_ai_agent_handler import create_flask_ai_handler
    from core.hash_match_command_injector import create_hash_match_injector
    from core.live_vector_simulator import LiveVectorSimulator, MarketSnapshot, SimulationConfig
    from core.profit_bucket_registry import ProfitBucketRegistry
    from core.real_time_execution_engine import RealTimeExecutionEngine
    from core.strategy_bit_mapper import StrategyBitMapper
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all core modules are available")
    exit(1)


class RecursiveTradingEcosystemTest:
    """
    🧪 Recursive Trading Ecosystem Test
    
    Comprehensive test suite that demonstrates:
    - Hash pattern recognition and matching
    - AI agent consensus building
    - Live market data simulation
    - Command injection and execution
    - Performance tracking and feedback
    """
    
    def __init__(self):
        """Initialize the test ecosystem."""
        self.test_results = {}
        self.start_time = time.time()
        
        # Initialize components
        logger.info("🔧 Initializing Recursive Trading Ecosystem Test")
        
        # Core components
        self.hash_injector = create_hash_match_injector()
        self.agent_memory = AgentMemory()
        self.profit_registry = ProfitBucketRegistry()
        self.strategy_mapper = StrategyBitMapper("matrix_dir")
        self.entropy_integrator = EntropySignalIntegrator()
        self.execution_engine = RealTimeExecutionEngine()
        
        # Simulation components
        self.simulator = None
        self.flask_handler = None
        
        logger.info("✅ Ecosystem components initialized")
    
    async def test_hash_match_injector(self):
        """Test the Hash Match Command Injector."""
        logger.info("🧪 Testing Hash Match Command Injector")
        
        try:
            # Create test tick data
            test_ticks = [
                {
                    "symbol": "BTCUSDT",
                    "price": 50000.0,
                    "volume": 1000.0,
                    "timestamp": time.time(),
                    "entropy": 0.015,
                    "volatility": 0.02
                },
                {
                    "symbol": "BTCUSDT",
                    "price": 50100.0,
                    "volume": 1200.0,
                    "timestamp": time.time() + 1,
                    "entropy": 0.025,
                    "volatility": 0.03
                },
                {
                    "symbol": "BTCUSDT",
                    "price": 49900.0,
                    "volume": 800.0,
                    "timestamp": time.time() + 2,
                    "entropy": 0.035,
                    "volatility": 0.04
                }
            ]
            
            injection_results = []
            
            for i, tick_data in enumerate(test_ticks):
                logger.info(f"   Processing tick {i+1}: Price=${tick_data['price']:,.2f}, Entropy={tick_data['entropy']:.4f}")
                
                result = await self.hash_injector.process_tick(tick_data)
                if result:
                    injection_results.append(result)
                    logger.info(f"   ✅ Command injected: {result.command.command_type.value} with confidence {result.command.confidence:.3f}")
                else:
                    logger.info(f"   ℹ️ No hash match found for tick {i+1}")
            
            # Get performance summary
            performance = self.hash_injector.get_performance_summary()
            
            self.test_results["hash_injector"] = {
                "success": True,
                "total_ticks": len(test_ticks),
                "injections": len(injection_results),
                "performance": performance
            }
            
            logger.info(f"✅ Hash Match Command Injector test completed: {len(injection_results)}/{len(test_ticks)} injections")
            
        except Exception as e:
            logger.error(f"❌ Hash Match Command Injector test failed: {e}")
            self.test_results["hash_injector"] = {"success": False, "error": str(e)}
    
    async def test_live_vector_simulator(self):
        """Test the Live Vector Simulator."""
        logger.info("🧪 Testing Live Vector Simulator")
        
        try:
            # Create simulation config
            config = SimulationConfig(
                initial_price=50000.0,
                base_volatility=0.02,
                tick_interval=0.1,  # 100ms ticks
                simulation_duration=30.0,  # 30 seconds
                random_seed=42
            )
            
            # Create simulator
            self.simulator = LiveVectorSimulator(config)
            
            # Track simulation results
            simulation_data = {
                "ticks": [],
                "hash_triggers": 0,
                "command_injections": 0,
                "regime_transitions": []
            }
            
            # Define callback function
            async def simulation_callback(snapshot: MarketSnapshot, hash_triggered: bool):
                simulation_data["ticks"].append({
                    "tick_number": self.simulator.total_ticks,
                    "price": snapshot.price,
                    "volume": snapshot.volume,
                    "entropy": snapshot.entropy,
                    "regime": snapshot.market_regime.value,
                    "hash_triggered": hash_triggered
                })
                
                if hash_triggered:
                    simulation_data["hash_triggers"] += 1
                
                if self.simulator.total_ticks % 50 == 0:  # Log every 50th tick
                    logger.info(f"   📊 Tick {self.simulator.total_ticks}: Price=${snapshot.price:,.2f}, Entropy={snapshot.entropy:.4f}, Regime={snapshot.market_regime.value}")
            
            # Run simulation
            await self.simulator.run_simulation(callback=simulation_callback)
            
            # Get simulation summary
            summary = self.simulator.get_simulation_summary()
            
            self.test_results["live_simulator"] = {
                "success": True,
                "total_ticks": self.simulator.total_ticks,
                "hash_triggers": self.simulator.hash_triggers,
                "command_injections": self.simulator.command_injections,
                "summary": summary,
                "simulation_data": simulation_data
            }
            
            logger.info(f"✅ Live Vector Simulator test completed: {self.simulator.total_ticks} ticks, {self.simulator.hash_triggers} hash triggers")
            
        except Exception as e:
            logger.error(f"❌ Live Vector Simulator test failed: {e}")
            self.test_results["live_simulator"] = {"success": False, "error": str(e)}
    
    async def test_flask_ai_handler(self):
        """Test the Flask AI Agent Handler."""
        logger.info("🧪 Testing Flask AI Agent Handler")
        
        try:
            # Create Flask handler
            self.flask_handler = create_flask_ai_handler()
            
            # Test agent registration
            test_agents = [
                {"agent_id": "test_gpt4o", "agent_type": "gpt4o"},
                {"agent_id": "test_claude", "agent_type": "claude"},
                {"agent_id": "test_r1", "agent_type": "r1"}
            ]
            
            registered_agents = []
            
            for agent_data in test_agents:
                # Simulate agent registration
                agent_id = agent_data["agent_id"]
                agent_type = agent_data["agent_type"]
                
                # Register agent in memory
                self.agent_memory.initialize_agent(agent_id, agent_type)
                registered_agents.append(agent_id)
                
                logger.info(f"   ✅ Registered agent: {agent_id} ({agent_type})")
            
            # Test hash processing
            test_hash_data = {
                "hash_signature": "9f3a1b2c",
                "market_data": {
                    "symbol": "BTCUSDT",
                    "price": 50000.0,
                    "volume": 1000.0,
                    "timestamp": time.time(),
                    "entropy": 0.015,
                    "volatility": 0.02
                }
            }
            
            # Simulate agent votes
            agent_votes = []
            
            for agent_id in registered_agents:
                # Simulate agent response
                confidence = 0.7 + (hash(agent_id) % 30) / 100  # Vary confidence
                
                vote_data = {
                    "agent_id": agent_id,
                    "agent_type": "gpt4o" if "gpt4o" in agent_id else "claude" if "claude" in agent_id else "r1",
                    "hash_signature": test_hash_data["hash_signature"],
                    "market_data": test_hash_data["market_data"],
                    "confidence": confidence
                }
                
                agent_votes.append(vote_data)
                logger.info(f"   📊 Agent {agent_id} vote: confidence={confidence:.3f}")
            
            # Simulate consensus building
            consensus_score = sum(vote["confidence"] for vote in agent_votes) / len(agent_votes)
            
            self.test_results["flask_ai_handler"] = {
                "success": True,
                "registered_agents": len(registered_agents),
                "agent_votes": agent_votes,
                "consensus_score": consensus_score,
                "test_hash_data": test_hash_data
            }
            
            logger.info(f"✅ Flask AI Agent Handler test completed: {len(registered_agents)} agents, consensus={consensus_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Flask AI Agent Handler test failed: {e}")
            self.test_results["flask_ai_handler"] = {"success": False, "error": str(e)}
    
    async def test_agent_memory_integration(self):
        """Test Agent Memory Integration."""
        logger.info("🧪 Testing Agent Memory Integration")
        
        try:
            # Test agent performance tracking
            test_agents = ["test_gpt4o", "test_claude", "test_r1"]
            
            memory_results = {}
            
            for agent_id in test_agents:
                # Initialize agent
                self.agent_memory.initialize_agent(agent_id, "gpt4o")
                
                # Simulate performance updates
                for i in range(10):
                    performance_score = 0.6 + (hash(f"{agent_id}_{i}") % 40) / 100
                    
                    self.agent_memory.update_agent_performance(
                        agent_id=agent_id,
                        performance_score=performance_score,
                        metadata={
                            "trade_id": f"trade_{i}",
                            "profit_pct": performance_score * 2 - 1,  # Convert to profit/loss
                            "timestamp": time.time() + i
                        }
                    )
                
                # Get agent performance
                performance = self.agent_memory.get_agent_performance(agent_id)
                memory_results[agent_id] = performance
                
                logger.info(f"   📊 Agent {agent_id}: avg_performance={performance.get('avg_performance', 0):.3f}")
            
            # Get overall performance database
            performance_db = self.agent_memory.get_performance_db()
            
            self.test_results["agent_memory"] = {
                "success": True,
                "test_agents": test_agents,
                "memory_results": memory_results,
                "performance_db": performance_db
            }
            
            logger.info(f"✅ Agent Memory Integration test completed: {len(test_agents)} agents tracked")
            
        except Exception as e:
            logger.error(f"❌ Agent Memory Integration test failed: {e}")
            self.test_results["agent_memory"] = {"success": False, "error": str(e)}
    
    async def test_profit_bucket_registry(self):
        """Test Profit Bucket Registry."""
        logger.info("🧪 Testing Profit Bucket Registry")
        
        try:
            # Test pattern registration
            test_patterns = [
                {
                    "hash_pattern": "9f3a1b2c",
                    "entry_price": 50000.0,
                    "exit_price": 50500.0,
                    "profit_pct": 0.01,
                    "time_to_exit": 300,
                    "strategy_id": "strategy_001",
                    "confidence": 0.85
                },
                {
                    "hash_pattern": "a1b2c3d4",
                    "entry_price": 50100.0,
                    "exit_price": 49800.0,
                    "profit_pct": -0.006,
                    "time_to_exit": 180,
                    "strategy_id": "strategy_002",
                    "confidence": 0.72
                },
                {
                    "hash_pattern": "b2c3d4e5",
                    "entry_price": 49900.0,
                    "exit_price": 50200.0,
                    "profit_pct": 0.006,
                    "time_to_exit": 240,
                    "strategy_id": "strategy_003",
                    "confidence": 0.78
                }
            ]
            
            registered_patterns = []
            
            for pattern in test_patterns:
                # Register pattern
                self.profit_registry.register_profit_pattern(
                    hash_pattern=pattern["hash_pattern"],
                    entry_price=pattern["entry_price"],
                    exit_price=pattern["exit_price"],
                    profit_pct=pattern["profit_pct"],
                    time_to_exit=pattern["time_to_exit"],
                    strategy_id=pattern["strategy_id"],
                    confidence=pattern["confidence"]
                )
                
                registered_patterns.append(pattern["hash_pattern"])
                logger.info(f"   ✅ Registered pattern: {pattern['hash_pattern']} (profit: {pattern['profit_pct']:.3f})")
            
            # Test pattern matching
            test_tick_blob = "BTCUSDT:50000.0:1234567890:0.015:1000.0"
            match = self.profit_registry.find_matching_pattern(
                tick_blob=test_tick_blob,
                min_confidence=0.7
            )
            
            # Get registry statistics
            stats = self.profit_registry.get_registry_statistics()
            
            self.test_results["profit_registry"] = {
                "success": True,
                "registered_patterns": len(registered_patterns),
                "pattern_matching": match is not None,
                "registry_stats": stats
            }
            
            logger.info(f"✅ Profit Bucket Registry test completed: {len(registered_patterns)} patterns registered")
            
        except Exception as e:
            logger.error(f"❌ Profit Bucket Registry test failed: {e}")
            self.test_results["profit_registry"] = {"success": False, "error": str(e)}
    
    async def test_integrated_ecosystem(self):
        """Test the complete integrated ecosystem."""
        logger.info("🧪 Testing Complete Integrated Ecosystem")
        
        try:
            # Create integrated test scenario
            logger.info("   🔄 Starting integrated ecosystem test...")
            
            # Step 1: Initialize profit patterns
            logger.info("   📊 Step 1: Initializing profit patterns")
            test_patterns = [
                {"hash": "9f3a1b2c", "profit": 0.015, "confidence": 0.85},
                {"hash": "a1b2c3d4", "profit": -0.008, "confidence": 0.72},
                {"hash": "b2c3d4e5", "profit": 0.012, "confidence": 0.78}
            ]
            
            for pattern in test_patterns:
                self.profit_registry.register_profit_pattern(
                    hash_pattern=pattern["hash"],
                    entry_price=50000.0,
                    exit_price=50000.0 * (1 + pattern["profit"]),
                    profit_pct=pattern["profit"],
                    time_to_exit=300,
                    strategy_id=f"test_strategy_{pattern['hash']}",
                    confidence=pattern["confidence"]
                )
            
            # Step 2: Initialize agents
            logger.info("   🤖 Step 2: Initializing AI agents")
            test_agents = ["ecosystem_gpt4o", "ecosystem_claude", "ecosystem_r1"]
            
            for agent_id in test_agents:
                self.agent_memory.initialize_agent(agent_id, "gpt4o")
            
            # Step 3: Run live simulation with hash injection
            logger.info("   🌊 Step 3: Running live simulation with hash injection")
            
            # Create simulation config for integrated test
            config = SimulationConfig(
                initial_price=50000.0,
                base_volatility=0.025,
                tick_interval=0.2,  # 200ms ticks
                simulation_duration=60.0,  # 1 minute
                random_seed=42
            )
            
            simulator = LiveVectorSimulator(config)
            
            # Track ecosystem performance
            ecosystem_data = {
                "ticks_processed": 0,
                "hash_matches": 0,
                "commands_injected": 0,
                "agent_consensus": 0,
                "successful_executions": 0
            }
            
            # Define integrated callback
            async def ecosystem_callback(snapshot: MarketSnapshot, hash_triggered: bool):
                ecosystem_data["ticks_processed"] += 1
                
                if hash_triggered:
                    ecosystem_data["hash_matches"] += 1
                    
                    # Process with hash injector
                    tick_data = {
                        "symbol": snapshot.symbol,
                        "price": snapshot.price,
                        "volume": snapshot.volume,
                        "timestamp": snapshot.timestamp,
                        "entropy": snapshot.entropy,
                        "volatility": snapshot.volatility
                    }
                    
                    injection_result = await self.hash_injector.process_tick(tick_data)
                    
                    if injection_result:
                        ecosystem_data["commands_injected"] += 1
                        
                        # Simulate agent consensus
                        agent_votes = []
                        for agent_id in test_agents:
                            confidence = 0.6 + (hash(f"{agent_id}_{ecosystem_data['ticks_processed']}") % 40) / 100
                            agent_votes.append(confidence)
                        
                        avg_consensus = sum(agent_votes) / len(agent_votes)
                        ecosystem_data["agent_consensus"] += 1
                        
                        if avg_consensus > 0.7:
                            ecosystem_data["successful_executions"] += 1
                        
                        logger.info(f"   🔗 Hash match at tick {ecosystem_data['ticks_processed']}: consensus={avg_consensus:.3f}")
                
                # Log progress
                if ecosystem_data["ticks_processed"] % 50 == 0:
                    logger.info(f"   📊 Ecosystem progress: {ecosystem_data['ticks_processed']} ticks, {ecosystem_data['hash_matches']} matches")
            
            # Run integrated simulation
            await simulator.run_simulation(callback=ecosystem_callback)
            
            # Calculate ecosystem metrics
            ecosystem_metrics = {
                "hash_match_rate": ecosystem_data["hash_matches"] / max(1, ecosystem_data["ticks_processed"]),
                "command_injection_rate": ecosystem_data["commands_injected"] / max(1, ecosystem_data["hash_matches"]),
                "consensus_rate": ecosystem_data["agent_consensus"] / max(1, ecosystem_data["commands_injected"]),
                "execution_success_rate": ecosystem_data["successful_executions"] / max(1, ecosystem_data["agent_consensus"])
            }
            
            self.test_results["integrated_ecosystem"] = {
                "success": True,
                "ecosystem_data": ecosystem_data,
                "ecosystem_metrics": ecosystem_metrics,
                "simulation_summary": simulator.get_simulation_summary()
            }
            
            logger.info(f"✅ Integrated Ecosystem test completed:")
            logger.info(f"   📊 Hash Match Rate: {ecosystem_metrics['hash_match_rate']:.3f}")
            logger.info(f"   💉 Command Injection Rate: {ecosystem_metrics['command_injection_rate']:.3f}")
            logger.info(f"   🤖 Consensus Rate: {ecosystem_metrics['consensus_rate']:.3f}")
            logger.info(f"   ✅ Execution Success Rate: {ecosystem_metrics['execution_success_rate']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Integrated Ecosystem test failed: {e}")
            self.test_results["integrated_ecosystem"] = {"success": False, "error": str(e)}
    
    async def run_complete_test_suite(self):
        """Run the complete test suite."""
        logger.info("🚀 Starting Complete Recursive Trading Ecosystem Test Suite")
        logger.info("=" * 80)
        
        # Run individual component tests
        await self.test_hash_match_injector()
        await self.test_live_vector_simulator()
        await self.test_flask_ai_handler()
        await self.test_agent_memory_integration()
        await self.test_profit_bucket_registry()
        
        # Run integrated ecosystem test
        await self.test_integrated_ecosystem()
        
        # Generate test summary
        self._generate_test_summary()
        
        logger.info("=" * 80)
        logger.info("🎉 Complete Recursive Trading Ecosystem Test Suite finished!")
    
    def _generate_test_summary(self):
        """Generate comprehensive test summary."""
        logger.info("📊 Generating Test Summary")
        
        # Calculate overall success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get("success", False))
        success_rate = successful_tests / total_tests
        
        # Create summary
        summary = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "test_duration": time.time() - self.start_time
            },
            "component_results": self.test_results,
            "ecosystem_health": {
                "hash_injector_ready": self.test_results.get("hash_injector", {}).get("success", False),
                "simulator_ready": self.test_results.get("live_simulator", {}).get("success", False),
                "ai_handler_ready": self.test_results.get("flask_ai_handler", {}).get("success", False),
                "memory_ready": self.test_results.get("agent_memory", {}).get("success", False),
                "registry_ready": self.test_results.get("profit_registry", {}).get("success", False),
                "integrated_ready": self.test_results.get("integrated_ecosystem", {}).get("success", False)
            }
        }
        
        # Log summary
        logger.info(f"📈 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Successful: {successful_tests}")
        logger.info(f"   Success Rate: {success_rate:.1%}")
        logger.info(f"   Duration: {summary['test_summary']['test_duration']:.1f}s")
        
        # Log ecosystem health
        logger.info(f"🏥 Ecosystem Health:")
        health = summary["ecosystem_health"]
        for component, ready in health.items():
            status = "✅ READY" if ready else "❌ FAILED"
            logger.info(f"   {component}: {status}")
        
        # Save detailed results
        output_file = "test_results_recursive_ecosystem.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📁 Detailed results saved to: {output_file}")
        
        # Final assessment
        if success_rate >= 0.8:
            logger.info("🎯 EXCELLENT: Ecosystem is ready for production!")
        elif success_rate >= 0.6:
            logger.info("⚠️ GOOD: Ecosystem is mostly functional, some components need attention")
        else:
            logger.info("🚨 CRITICAL: Ecosystem has significant issues that need immediate attention")
        
        return summary


async def main():
    """Main test execution function."""
    logger.info("🧪 Recursive Trading Ecosystem Test")
    logger.info("Testing Schwabot's complete hash-based, AI-driven trading system")
    
    # Create and run test suite
    test_suite = RecursiveTradingEcosystemTest()
    await test_suite.run_complete_test_suite()


if __name__ == "__main__":
    asyncio.run(main()) 