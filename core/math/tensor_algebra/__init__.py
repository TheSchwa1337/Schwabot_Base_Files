#!/usr/bin/env python3
"""
Tensor Algebra Module - Mathematical Foundation for AI Vector Operations
=======================================================================

This module provides the mathematical foundation for multi-layer AI vector 
comparison and symbolic memory operations in the Schwabot trading system.

Core Components:
- UnifiedTensorAlgebra: Core tensor operations
- TensorEngine: Advanced tensor processing
- ProfitEngine: Profit surface calculations
- EntropyEngine: Entropy-based signal processing

Mathematical Foundation:
- Tensor Operations: dot products, projections, gradients
- Vector Spaces: multi-dimensional analysis
- Symbolic Memory: pattern recognition and storage
- AI Integration: machine learning tensor operations
"""

from .unified_tensor_algebra import (
    tensor_dot,
    tensor_project,
    tensor_entropy_gradient,
    tensor_normalize,
    tensor_correlation,
    tensor_distance,
    tensor_similarity,
    tensor_convolution,
    tensor_fft,
    tensor_inverse_fft,
    UnifiedTensorAlgebra
)

from .tensor_engine import (
    TensorEngine,
    create_tensor_space,
    analyze_tensor_patterns,
    compute_tensor_statistics,
    tensor_pattern_matching,
    tensor_clustering,
    tensor_dimensionality_reduction
)

from .profit_engine import (
    compute_profit_surface,
    optimize_long_hold_positions,
    calculate_profit_gradient,
    estimate_profit_curves,
    analyze_profit_distribution,
    ProfitEngine
)

from .entropy_engine import (
    entropy_filter,
    calculate_dynamic_entropy,
    entropy_wave_detection,
    entropy_pattern_analysis,
    entropy_based_clustering,
    EntropyEngine
)

__all__ = [
    # Unified Tensor Algebra
    'tensor_dot',
    'tensor_project', 
    'tensor_entropy_gradient',
    'tensor_normalize',
    'tensor_correlation',
    'tensor_distance',
    'tensor_similarity',
    'tensor_convolution',
    'tensor_fft',
    'tensor_inverse_fft',
    'UnifiedTensorAlgebra',
    
    # Tensor Engine
    'TensorEngine',
    'create_tensor_space',
    'analyze_tensor_patterns',
    'compute_tensor_statistics',
    'tensor_pattern_matching',
    'tensor_clustering',
    'tensor_dimensionality_reduction',
    
    # Profit Engine
    'compute_profit_surface',
    'optimize_long_hold_positions',
    'calculate_profit_gradient',
    'estimate_profit_curves',
    'analyze_profit_distribution',
    'ProfitEngine',
    
    # Entropy Engine
    'entropy_filter',
    'calculate_dynamic_entropy',
    'entropy_wave_detection',
    'entropy_pattern_analysis',
    'entropy_based_clustering',
    'EntropyEngine'
]

# Version information
__version__ = "1.0.0"
__author__ = "Schwabot Development Team"
__description__ = "Tensor Algebra Module for Advanced AI Vector Operations" 