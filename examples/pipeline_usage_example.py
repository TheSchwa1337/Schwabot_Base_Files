#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Pipeline Usage Example
===============================

This example demonstrates how to use the cleaned Schwabot pipeline
with Cursor-friendly interfaces while maintaining the full power
of the mathematical framework.

Key Features Demonstrated:
• Symbolic math operations (Ω = ∇ψ(t) * D)
• 2-gram pattern detection
• Hardware optimization (CPU/GPU)
• Context-aware computations
• Performance monitoring
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the unified pipeline manager
try:
    from core.symbolic_math_interface import PhaseValue, SignalField, SymbolicContext, TimeIndex
    from core.unified_pipeline_manager import PipelineContext, UnifiedPipelineManager
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

def example_phase_omega_computation():
    """
    Example: Compute phase omega using the unified pipeline.
    
    This demonstrates the Cursor-friendly way to compute:
    Ω = ∇ψ(t) * D
    """
    if not PIPELINE_AVAILABLE:
        logger.error("Pipeline not available")
        return
    
    # Create pipeline manager
    pipeline = UnifiedPipelineManager()
    
    # Activate the pipeline
    if not pipeline.activate():
        logger.error("Failed to activate pipeline")
        return
    
    logger.info("✅ Pipeline activated successfully")
    
    # Generate sample signal data
    signal_data = np.random.randn(100) * 0.1 + np.sin(np.linspace(0, 4*np.pi, 100))
    
    # Create context for computation
    context = PipelineContext(
        cycle_id=42,
        vault_state="phantom",
        entropy_index=5,
        phantom_layer=True
    )
    
    # Compute phase omega at different time indices
    time_indices = [10, 25, 50, 75, 90]
    
    logger.info("Computing phase omega values...")
    for t_idx in time_indices:
        # This is the Cursor-friendly way to compute Ω = ∇ψ(t) * D
        omega = pipeline.compute_phase_omega(signal_data, t_idx, context)
        
        logger.info(f"Time {t_idx}: Ω = {omega:.6f}")
    
    # Get performance statistics
    stats = pipeline.get_performance_stats()
    logger.info(f"Performance: {stats['operation_count']} operations, "
                f"avg time: {stats['average_processing_time']:.4f}s")

def example_2gram_pattern_detection():
    """
    Example: 2-gram pattern detection with signal feeding.
    
    This demonstrates how to use the 2-gram detector for pattern recognition.
    """
    if not PIPELINE_AVAILABLE:
        logger.error("Pipeline not available")
        return
    
    # Create pipeline manager
    pipeline = UnifiedPipelineManager()
    
    if not pipeline.activate():
        logger.error("Failed to activate pipeline")
        return
    
    logger.info("✅ Pipeline activated for 2-gram detection")
    
    # Create context
    context = PipelineContext(
        cycle_id=123,
        vault_state="normal",
        entropy_index=0,
        phantom_layer=False
    )
    
    # Feed a sequence of signals to detect patterns
    signals = [0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.3, 0.1, 0.2]
    
    logger.info("Feeding signals to 2-gram detector...")
    for i, signal in enumerate(signals):
        pattern = pipeline.feed_signal_to_2gram(signal, context)
        if pattern:
            logger.info(f"Signal {i}: Detected pattern {pattern}")
    
    # Get 2-gram statistics
    stats = pipeline.get_2gram_statistics()
    logger.info(f"2-gram Statistics: {stats}")
    
    # Get top patterns
    top_patterns = pipeline.get_top_2gram_patterns(5)
    logger.info("Top 2-gram patterns:")
    for pattern, frequency in top_patterns:
        logger.info(f"  {pattern}: {frequency} occurrences")
    
    # Get burst patterns
    burst_patterns = pipeline.get_burst_patterns()
    if burst_patterns:
        logger.info("Burst patterns detected:")
        for pattern, timestamp in burst_patterns:
            logger.info(f"  {pattern} at {timestamp}")
    
    # Calculate entropy
    entropy = pipeline.calculate_entropy()
    logger.info(f"Shannon entropy: {entropy:.4f}")

def example_hardware_optimization():
    """
    Example: Hardware optimization and performance monitoring.
    
    This demonstrates how the pipeline automatically optimizes
    for CPU/GPU based on available hardware.
    """
    if not PIPELINE_AVAILABLE:
        logger.error("Pipeline not available")
        return
    
    # Create pipeline manager
    pipeline = UnifiedPipelineManager()
    
    if not pipeline.activate():
        logger.error("Failed to activate pipeline")
        return
    
    # Get hardware information
    hw_info = pipeline.get_hardware_info()
    logger.info("Hardware Information:")
    for key, value in hw_info.items():
        logger.info(f"  {key}: {value}")
    
    # Perform intensive computation to test optimization
    logger.info("Performing intensive computation test...")
    
    # Large signal data for testing
    large_signal = np.random.randn(10000) * 0.1 + np.sin(np.linspace(0, 20*np.pi, 10000))
    
    # Multiple computations to test performance
    start_time = time.time()
    results = []
    
    for i in range(100):
        omega = pipeline.compute_phase_omega(large_signal, i % 1000)
        results.append(omega)
    
    total_time = time.time() - start_time
    
    logger.info(f"Computed {len(results)} phase omega values in {total_time:.2f}s")
    logger.info(f"Average time per computation: {total_time/len(results)*1000:.2f}ms")
    
    # Get final performance stats
    final_stats = pipeline.get_performance_stats()
    logger.info(f"Final performance stats: {final_stats}")

def example_context_aware_computation():
    """
    Example: Context-aware mathematical computations.
    
    This demonstrates how the pipeline adapts computations
    based on context (phantom layer, vault state, etc.).
    """
    if not PIPELINE_AVAILABLE:
        logger.error("Pipeline not available")
        return
    
    # Create pipeline manager
    pipeline = UnifiedPipelineManager()
    
    if not pipeline.activate():
        logger.error("Failed to activate pipeline")
        return
    
    # Same signal data
    signal_data = np.random.randn(100) * 0.1 + np.sin(np.linspace(0, 4*np.pi, 100))
    time_idx = 50
    
    # Different contexts
    contexts = [
        PipelineContext(cycle_id=1, vault_state="normal", entropy_index=0, phantom_layer=False),
        PipelineContext(cycle_id=2, vault_state="phantom", entropy_index=1, phantom_layer=False),
        PipelineContext(cycle_id=3, vault_state="phantom", entropy_index=2, phantom_layer=True),
    ]
    
    logger.info("Computing phase omega with different contexts...")
    
    for i, context in enumerate(contexts):
        omega = pipeline.compute_phase_omega(signal_data, time_idx, context)
        
        logger.info(f"Context {i+1} ({context.vault_state}, phantom_layer={context.phantom_layer}): "
                   f"Ω = {omega:.6f}")

def example_pipeline_status_monitoring():
    """
    Example: Comprehensive pipeline status monitoring.
    
    This demonstrates how to monitor the entire pipeline system.
    """
    if not PIPELINE_AVAILABLE:
        logger.error("Pipeline not available")
        return
    
    # Create pipeline manager
    pipeline = UnifiedPipelineManager()
    
    if not pipeline.activate():
        logger.error("Failed to activate pipeline")
        return
    
    # Get comprehensive status
    status = pipeline.get_status()
    
    logger.info("=== Pipeline Status Report ===")
    logger.info(f"Active: {status['active']}")
    logger.info(f"Initialized: {status['initialized']}")
    logger.info(f"Config loaded: {status['config_loaded']}")
    logger.info(f"Components available: {status['components_available']}")
    
    # Performance stats
    perf_stats = status['performance_stats']
    logger.info(f"Operations: {perf_stats['operation_count']}")
    logger.info(f"Total processing time: {perf_stats['total_processing_time']:.2f}s")
    logger.info(f"Average processing time: {perf_stats['average_processing_time']:.4f}s")
    
    # Component statuses
    if 'symbolic_math_engine' in status:
        logger.info(f"Symbolic math engine: {status['symbolic_math_engine']['active']}")
    
    if 'two_gram_detector' in status:
        logger.info(f"2-gram detector: {status['two_gram_detector']['active']}")
        logger.info(f"Pattern count: {status['two_gram_detector']['pattern_count']}")
    
    if 'math_orchestrator' in status:
        logger.info(f"Math orchestrator: {status['math_orchestrator']['active']}")

def main():
    """
    Main function to run all examples.
    """
    logger.info("🚀 Starting Schwabot Pipeline Examples")
    logger.info("=" * 50)
    
    if not PIPELINE_AVAILABLE:
        logger.error("❌ Pipeline components not available. Please check imports.")
        return
    
    try:
        # Run examples
        logger.info("\n1. Phase Omega Computation Example")
        example_phase_omega_computation()
        
        logger.info("\n2. 2-Gram Pattern Detection Example")
        example_2gram_pattern_detection()
        
        logger.info("\n3. Hardware Optimization Example")
        example_hardware_optimization()
        
        logger.info("\n4. Context-Aware Computation Example")
        example_context_aware_computation()
        
        logger.info("\n5. Pipeline Status Monitoring Example")
        example_pipeline_status_monitoring()
        
        logger.info("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 