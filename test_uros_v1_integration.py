#!/usr/bin/env python3
"""
UROS v1.0 End-to-End Integration Test.

This test validates the complete UROS v1.0 pipeline:
1. Loads Prophet curve data
2. Posts mock AI commands
3. Validates flow through sequencer → validator → strategy_mapper
4. Tests all mathematical calculations (α, ρ, drift, cost)
5. Verifies memory management and hash registry integration
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import UROS v1.0 modules
try:
    from core.gpt_command_layer import (
        AIAgentType, CommandDomain, CommandPriority, AICommand, CommandResponse
    )
    from core.prophet_connector import (
        ProphetConnector, ProphetCurve, CurveType, compute_alpha_score,
        analyze_curve_alignment
    )
    from core.memory_stack.ai_command_sequencer import (
        AICommandSequencer, sequence_ai_command, update_command_sequence_result
    )
    from core.memory_stack.memory_key_allocator import (
        MemoryKeyAllocator, allocate_memory_key, create_memory_link, KeyType
    )
    from core.memory_stack.execution_validator import (
        ExecutionValidator, simulate_execution_cost, validate_drift, validate_execution
    )
    from core.strategy_mapper import (
        StrategyMapper, map_strategy_enhanced, StrategyMappingResult
    )
    from core.dlt_waveform_engine import DLTWaveformEngine
    from core.hash_registry import HashRegistry, register_hash_entry, update_hash_status
    from core.api_gateway import APIGateway
    from core.fault_bus import FaultBus
    from core.utils.windows_cli_compatibility import safe_print, safe_format_error
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import UROS v1.0 modules: {e}")
    MODULES_AVAILABLE = False


class UROSv1IntegrationTest:
    """Comprehensive integration test for UROS v1.0."""
    
    def __init__(self):
        """Initialize the integration test."""
        self.test_results = []
        self.start_time = time.time()
        
        # Initialize components
        self.prophet_connector = None
        self.command_sequencer = None
        self.memory_allocator = None
        self.execution_validator = None
        self.strategy_mapper = None
        self.waveform_engine = None
        self.hash_registry = None
        self.api_gateway = None
        self.fault_bus = None
        
        safe_print("🧪 UROS v1.0 Integration Test initialized")
    
    async def run_complete_test(self) -> bool:
        """Run the complete integration test suite."""
        try:
            safe_print("🚀 Starting UROS v1.0 Integration Test Suite...")
            
            # Test 1: Component Initialization
            if not await self._test_component_initialization():
                return False
            
            # Test 2: Prophet Curve Loading
            if not await self._test_prophet_curves():
                return False
            
            # Test 3: AI Command Sequencing
            if not await self._test_command_sequencing():
                return False
            
            # Test 4: Memory Management
            if not await self._test_memory_management():
                return False
            
            # Test 5: Execution Validation
            if not await self._test_execution_validation():
                return False
            
            # Test 6: Strategy Mapping
            if not await self._test_strategy_mapping():
                return False
            
            # Test 7: Mathematical Calculations
            if not await self._test_mathematical_calculations():
                return False
            
            # Test 8: Hash Registry Integration
            if not await self._test_hash_registry():
                return False
            
            # Test 9: End-to-End Pipeline
            if not await self._test_end_to_end_pipeline():
                return False
            
            # Test 10: Performance Metrics
            if not await self._test_performance_metrics():
                return False
            
            # Generate final report
            await self._generate_test_report()
            
            safe_print("✅ All UROS v1.0 integration tests passed!")
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "integration_test")
            safe_print(f"❌ Integration test failed: {error_msg}")
            return False
    
    async def _test_component_initialization(self) -> bool:
        """Test component initialization."""
        try:
            safe_print("🔧 Testing component initialization...")
            
            # Initialize all components
            self.prophet_connector = ProphetConnector()
            self.command_sequencer = AICommandSequencer()
            self.memory_allocator = MemoryKeyAllocator()
            self.execution_validator = ExecutionValidator()
            self.strategy_mapper = StrategyMapper()
            self.waveform_engine = DLTWaveformEngine()
            self.hash_registry = HashRegistry()
            self.api_gateway = APIGateway()
            self.fault_bus = FaultBus()
            
            # Verify components are initialized
            components = [
                ("Prophet Connector", self.prophet_connector),
                ("Command Sequencer", self.command_sequencer),
                ("Memory Allocator", self.memory_allocator),
                ("Execution Validator", self.execution_validator),
                ("Strategy Mapper", self.strategy_mapper),
                ("Waveform Engine", self.waveform_engine),
                ("Hash Registry", self.hash_registry),
                ("API Gateway", self.api_gateway),
                ("Fault Bus", self.fault_bus)
            ]
            
            for name, component in components:
                if component is None:
                    safe_print(f"❌ {name} failed to initialize")
                    return False
                safe_print(f"✅ {name} initialized successfully")
            
            self.test_results.append(("Component Initialization", True, "All components initialized"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "component_initialization")
            safe_print(f"❌ Component initialization failed: {error_msg}")
            self.test_results.append(("Component Initialization", False, error_msg))
            return False
    
    async def _test_prophet_curves(self) -> bool:
        """Test Prophet curve loading and analysis."""
        try:
            safe_print("📊 Testing Prophet curve functionality...")
            
            # Test curve loading
            curves = self.prophet_connector.get_curves_by_type(CurveType.BTC_PRICE)
            if not curves:
                safe_print("⚠️ No Prophet curves found, creating test curve...")
                
                # Create test curve
                test_curve = ProphetCurve(
                    curve_id="test_btc_curve",
                    curve_type=CurveType.BTC_PRICE,
                    asset="BTC",
                    timeframe="1h",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    data_points=[
                        {"timestamp": time.time(), "price": 50000.0, "volume": 1000.0}
                    ],
                    confidence_score=0.8
                )
                self.prophet_connector.add_curve(test_curve)
                curves = [test_curve]
            
            # Test alpha calculation
            alpha_result = compute_alpha_score(
                p_actual=0.05,
                p_expected=0.03,
                delta_t=3600.0,
                curve_id=curves[0].curve_id
            )
            
            # Test curve alignment
            alignment_result = analyze_curve_alignment(
                curve_id=curves[0].curve_id,
                current_price=50000.0,
                current_volume=1000.0,
                current_time=datetime.now()
            )
            
            safe_print(f"✅ Prophet curves: {len(curves)} curves loaded")
            safe_print(f"✅ Alpha calculation: {alpha_result.alpha_value:.4f}")
            safe_print(f"✅ Curve alignment: {alignment_result.alignment_score:.3f}")
            
            self.test_results.append(("Prophet Curves", True, f"{len(curves)} curves tested"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "prophet_curves")
            safe_print(f"❌ Prophet curve test failed: {error_msg}")
            self.test_results.append(("Prophet Curves", False, error_msg))
            return False
    
    async def _test_command_sequencing(self) -> bool:
        """Test AI command sequencing."""
        try:
            safe_print("🧠 Testing AI command sequencing...")
            
            # Create test command
            test_command = AICommand(
                command_id="test_cmd_001",
                agent_type=AIAgentType.GPT,
                domain=CommandDomain.STRATEGY,
                priority=CommandPriority.MEDIUM,
                hash_signature="test_hash_123",
                timestamp=datetime.now(),
                payload={
                    "strategy_name": "test_strategy",
                    "parameters": {"test": True},
                    "target_profit": 100.0
                },
                context={"test": True}
            )
            
            # Sequence command
            sequence = await sequence_ai_command(test_command, tick=1000)
            
            # Create test response
            test_response = CommandResponse(
                command_id=test_command.command_id,
                success=True,
                result={"profit": 50.0},
                execution_time=1.5,
                timestamp=datetime.now()
            )
            
            # Update result
            success = await update_command_sequence_result(
                sequence.sequence_id,
                test_response,
                profit_delta=50.0
            )
            
            if not success:
                raise Exception("Failed to update command sequence result")
            
            safe_print(f"✅ Command sequenced: {sequence.sequence_id}")
            safe_print(f"✅ Command result updated: {success}")
            
            self.test_results.append(("Command Sequencing", True, "Command sequenced and updated"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "command_sequencing")
            safe_print(f"❌ Command sequencing test failed: {error_msg}")
            self.test_results.append(("Command Sequencing", False, error_msg))
            return False
    
    async def _test_memory_management(self) -> bool:
        """Test memory key allocation and management."""
        try:
            safe_print("🔑 Testing memory management...")
            
            # Allocate memory keys
            key1 = allocate_memory_key(
                agent_type="gpt",
                domain="strategy",
                hash_signature="abc123def456",
                tick=1000,
                key_type=KeyType.SYMBOLIC,
                alpha_score=0.05,
                profit_delta=50.0
            )
            
            key2 = allocate_memory_key(
                agent_type="claude",
                domain="profit",
                hash_signature="def456ghi789",
                tick=1001,
                key_type=KeyType.HASH_BASED,
                alpha_score=0.03,
                profit_delta=30.0
            )
            
            # Create memory link
            link = create_memory_link(
                source_key=key1.key_id,
                target_key=key2.key_id,
                link_type="profit_correlation",
                alpha_correlation=0.7,
                confidence=0.8
            )
            
            if not link:
                raise Exception("Failed to create memory link")
            
            # Find similar keys
            similar_keys = self.memory_allocator.find_similar_keys(key1.key_id, max_results=5)
            
            safe_print(f"✅ Memory keys allocated: {key1.key_id}, {key2.key_id}")
            safe_print(f"✅ Memory link created: {link.link_id}")
            safe_print(f"✅ Similar keys found: {len(similar_keys)}")
            
            self.test_results.append(("Memory Management", True, f"{len(similar_keys)} similar keys found"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "memory_management")
            safe_print(f"❌ Memory management test failed: {error_msg}")
            self.test_results.append(("Memory Management", False, error_msg))
            return False
    
    async def _test_execution_validation(self) -> bool:
        """Test execution cost simulation and validation."""
        try:
            safe_print("✅ Testing execution validation...")
            
            # Simulate execution cost
            test_payload = {"strategy": "test", "parameters": {"test": True}}
            test_market_data = {"volatility": 0.1, "volume": 1000.0}
            
            execution_cost = simulate_execution_cost(
                command_id="test_cmd_001",
                payload=test_payload,
                market_data=test_market_data,
                complexity_score=1.5
            )
            
            # Validate drift
            expected_time = datetime.now()
            actual_time = datetime.now()
            
            drift_validation = validate_drift(
                command_id="test_cmd_001",
                expected_time=expected_time,
                actual_time=actual_time,
                alpha_score=0.05,
                confidence_score=0.8
            )
            
            # Validate execution
            execution_validation = validate_execution(
                command_id="test_cmd_001",
                execution_cost=execution_cost,
                drift_validation=drift_validation,
                profit_delta=50.0,
                risk_tolerance=0.5
            )
            
            safe_print(f"✅ Execution cost: {execution_cost.total_cost:.2f}")
            safe_print(f"✅ Drift validation: {drift_validation.drift_magnitude:.2f}s")
            safe_print(f"✅ Execution validation: {execution_validation.validation_status.value}")
            
            self.test_results.append(("Execution Validation", True, f"Status: {execution_validation.validation_status.value}"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "execution_validation")
            safe_print(f"❌ Execution validation test failed: {error_msg}")
            self.test_results.append(("Execution Validation", False, error_msg))
            return False
    
    async def _test_strategy_mapping(self) -> bool:
        """Test enhanced strategy mapping."""
        try:
            safe_print("🗺️ Testing strategy mapping...")
            
            # Create test execution packet
            execution_packet = {
                "strategy_type": "momentum",
                "tick": 1000,
                "expected_profit": 100.0,
                "actual_profit": 75.0,
                "parameters": {"test": True}
            }
            
            # Test enhanced strategy mapping
            result = await map_strategy_enhanced(
                execution_packet=execution_packet,
                agent_type=AIAgentType.GPT,
                prophet_curve_id="test_btc_curve",
                market_data={"price": 50000.0, "volume": 1000.0, "volatility": 0.02}
            )
            
            if not result.success:
                raise Exception(f"Strategy mapping failed: {result.metadata.get('error', 'Unknown error')}")
            
            safe_print(f"✅ Strategy mapped: {result.alpha_score:.4f} alpha")
            safe_print(f"✅ Memory key: {result.memory_key}")
            safe_print(f"✅ Validation score: {result.validation_score:.3f}")
            
            self.test_results.append(("Strategy Mapping", True, f"Alpha: {result.alpha_score:.4f}"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "strategy_mapping")
            safe_print(f"❌ Strategy mapping test failed: {error_msg}")
            self.test_results.append(("Strategy Mapping", False, error_msg))
            return False
    
    async def _test_mathematical_calculations(self) -> bool:
        """Test mathematical calculations (α, ρ, drift, cost)."""
        try:
            safe_print("🧮 Testing mathematical calculations...")
            
            # Test DLT waveform engine
            for i in range(50):
                price = 50000.0 + (i * 10) + (i % 3 - 1) * 100  # Simulate price movement
                self.waveform_engine.update_tick_data(price)
            
            # Analyze waveform
            analysis = self.waveform_engine.analyze_current_waveform()
            
            # Extract mathematical measures
            math_measures = analysis.get("mathematical_measures", {})
            rho_coefficient = math_measures.get("rho_coefficient", 0.0)
            resonance_strength = math_measures.get("resonance_strength", 0.0)
            entropy_complexity = math_measures.get("entropy_complexity", 0.0)
            current_acceleration = math_measures.get("current_acceleration", 0.0)
            current_velocity = math_measures.get("current_velocity", 0.0)
            
            # Test alpha calculation
            alpha_result = compute_alpha_score(
                p_actual=0.05,
                p_expected=0.03,
                delta_t=3600.0,
                curve_id="test_btc_curve"
            )
            
            safe_print(f"✅ ρ coefficient: {rho_coefficient:.4f}")
            safe_print(f"✅ Resonance strength: {resonance_strength:.4f}")
            safe_print(f"✅ Entropy complexity: {entropy_complexity:.4f}")
            safe_print(f"✅ Current acceleration: {current_acceleration:.4f}")
            safe_print(f"✅ Current velocity: {current_velocity:.4f}")
            safe_print(f"✅ Alpha score: {alpha_result.alpha_value:.4f}")
            
            self.test_results.append(("Mathematical Calculations", True, f"ρ: {rho_coefficient:.4f}, α: {alpha_result.alpha_value:.4f}"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "mathematical_calculations")
            safe_print(f"❌ Mathematical calculations test failed: {error_msg}")
            self.test_results.append(("Mathematical Calculations", False, error_msg))
            return False
    
    async def _test_hash_registry(self) -> bool:
        """Test hash registry integration."""
        try:
            safe_print("🔗 Testing hash registry...")
            
            # Register hash entry
            hash_id = await register_hash_entry(
                hash_type="command",
                agent_type="gpt",
                domain="strategy",
                payload={"test": True},
                context={"test": True},
                command_id="test_cmd_001",
                confidence_score=0.8
            )
            
            # Update hash status
            success = await update_hash_status(
                hash_id=hash_id,
                status="completed",
                result={"profit": 50.0},
                error_message=None,
                execution_time=1.5
            )
            
            if not success:
                raise Exception("Failed to update hash status")
            
            safe_print(f"✅ Hash registered: {hash_id}")
            safe_print(f"✅ Hash status updated: {success}")
            
            self.test_results.append(("Hash Registry", True, f"Hash ID: {hash_id}"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "hash_registry")
            safe_print(f"❌ Hash registry test failed: {error_msg}")
            self.test_results.append(("Hash Registry", False, error_msg))
            return False
    
    async def _test_end_to_end_pipeline(self) -> bool:
        """Test complete end-to-end pipeline."""
        try:
            safe_print("🔄 Testing end-to-end pipeline...")
            
            # Create comprehensive test scenario
            test_command = AICommand(
                command_id="e2e_test_cmd",
                agent_type=AIAgentType.CLAUDE,
                domain=CommandDomain.STRATEGY,
                priority=CommandPriority.HIGH,
                hash_signature="e2e_hash_123",
                timestamp=datetime.now(),
                payload={
                    "strategy_name": "e2e_test_strategy",
                    "parameters": {"test": True, "complexity": "high"},
                    "target_profit": 200.0
                },
                context={"pipeline_test": True}
            )
            
            # Step 1: Sequence command
            sequence = await sequence_ai_command(test_command, tick=2000)
            
            # Step 2: Allocate memory key
            memory_key = allocate_memory_key(
                agent_type=test_command.agent_type.value,
                domain=test_command.domain.value,
                hash_signature=test_command.hash_signature,
                tick=2000,
                key_type=KeyType.HYBRID,
                alpha_score=0.0,
                metadata={'pipeline_test': True}
            )
            
            # Step 3: Simulate execution cost
            execution_cost = simulate_execution_cost(
                command_id=test_command.command_id,
                payload=test_command.payload,
                market_data={"price": 50000.0, "volume": 1500.0, "volatility": 0.03},
                complexity_score=2.0
            )
            
            # Step 4: Map strategy
            execution_packet = {
                "strategy_type": "high_frequency",
                "tick": 2000,
                "expected_profit": 200.0,
                "actual_profit": 180.0,
                "parameters": test_command.payload
            }
            
            mapping_result = await map_strategy_enhanced(
                execution_packet=execution_packet,
                agent_type=test_command.agent_type,
                prophet_curve_id="test_btc_curve",
                market_data={"price": 50000.0, "volume": 1500.0, "volatility": 0.03}
            )
            
            # Step 5: Update command result
            response = CommandResponse(
                command_id=test_command.command_id,
                success=mapping_result.success,
                result=mapping_result.mapped_strategy,
                execution_time=2.0,
                timestamp=datetime.now()
            )
            
            await update_command_sequence_result(
                sequence.sequence_id,
                response,
                profit_delta=180.0,
                prophet_curve_id="test_btc_curve",
                market_data={"price": 50000.0, "volume": 1500.0, "volatility": 0.03}
            )
            
            safe_print(f"✅ E2E Pipeline: Command {test_command.command_id} processed successfully")
            safe_print(f"✅ E2E Pipeline: Alpha score {mapping_result.alpha_score:.4f}")
            safe_print(f"✅ E2E Pipeline: Memory key {memory_key.key_id}")
            safe_print(f"✅ E2E Pipeline: Execution cost {execution_cost.total_cost:.2f}")
            
            self.test_results.append(("End-to-End Pipeline", True, f"Command {test_command.command_id} completed"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "end_to_end_pipeline")
            safe_print(f"❌ End-to-end pipeline test failed: {error_msg}")
            self.test_results.append(("End-to-End Pipeline", False, error_msg))
            return False
    
    async def _test_performance_metrics(self) -> bool:
        """Test performance metrics collection."""
        try:
            safe_print("📈 Testing performance metrics...")
            
            # Collect metrics from all components
            sequencer_metrics = self.command_sequencer.get_performance_metrics()
            memory_metrics = self.memory_allocator.get_performance_metrics()
            validator_metrics = self.execution_validator.get_performance_metrics()
            mapper_metrics = self.strategy_mapper.get_performance_metrics()
            waveform_metrics = self.waveform_engine.get_waveform_statistics()
            
            # Log key metrics
            safe_print(f"✅ Command sequencer: {sequencer_metrics['total_commands_processed']} commands")
            safe_print(f"✅ Memory allocator: {memory_metrics['total_keys_allocated']} keys")
            safe_print(f"✅ Execution validator: {validator_metrics['total_validations']} validations")
            safe_print(f"✅ Strategy mapper: {mapper_metrics['total_mappings']} mappings")
            safe_print(f"✅ Waveform engine: {waveform_metrics['total_waveforms']} waveforms")
            
            self.test_results.append(("Performance Metrics", True, "All metrics collected successfully"))
            return True
            
        except Exception as e:
            error_msg = safe_format_error(e, "performance_metrics")
            safe_print(f"❌ Performance metrics test failed: {error_msg}")
            self.test_results.append(("Performance Metrics", False, error_msg))
            return False
    
    async def _generate_test_report(self) -> None:
        """Generate comprehensive test report."""
        try:
            end_time = time.time()
            duration = end_time - self.start_time
            
            # Calculate summary
            total_tests = len(self.test_results)
            passed_tests = sum(1 for _, success, _ in self.test_results if success)
            failed_tests = total_tests - passed_tests
            
            # Generate report
            report = {
                "test_summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
                    "duration_seconds": duration
                },
                "test_results": [
                    {
                        "test_name": name,
                        "success": success,
                        "details": details
                    }
                    for name, success, details in self.test_results
                ],
                "timestamp": datetime.now().isoformat(),
                "uros_version": "1.0.0"
            }
            
            # Save report
            report_file = Path("test_reports/uros_v1_integration_report.json")
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Print summary
            safe_print("\n" + "="*60)
            safe_print("🧪 UROS v1.0 INTEGRATION TEST REPORT")
            safe_print("="*60)
            safe_print(f"Total Tests: {total_tests}")
            safe_print(f"Passed: {passed_tests}")
            safe_print(f"Failed: {failed_tests}")
            safe_print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
            safe_print(f"Duration: {duration:.2f} seconds")
            safe_print("="*60)
            
            # Print failed tests
            if failed_tests > 0:
                safe_print("\n❌ FAILED TESTS:")
                for name, success, details in self.test_results:
                    if not success:
                        safe_print(f"  - {name}: {details}")
            
            safe_print(f"\n📄 Full report saved to: {report_file}")
            
        except Exception as e:
            error_msg = safe_format_error(e, "test_report_generation")
            safe_print(f"❌ Failed to generate test report: {error_msg}")


async def main():
    """Main test runner."""
    if not MODULES_AVAILABLE:
        safe_print("❌ UROS v1.0 modules not available. Please check imports.")
        return False
    
    # Run integration test
    test = UROSv1IntegrationTest()
    success = await test.run_complete_test()
    
    return success


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 