"""
Enhanced System Demonstration
============================

Comprehensive demonstration of the enhanced Schwabot system including:
- Mathematical utilities for line rendering
- Standardized configuration loading
- Matrix fault resolution
- System integration and monitoring
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
import time
import random
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demonstrate_config_loading():
    """Demonstrate standardized configuration loading"""
    print("\n🔧 Configuration Loading Demonstration")
    print("=" * 50)
    
    try:
        from config.io_utils import load_config, ensure_config_exists
        from config.matrix_response_schema import MATRIX_RESPONSE_SCHEMA, LINE_RENDER_SCHEMA
        
        print("✅ Configuration utilities imported successfully")
        
        # Show schema information
        print(f"\n📋 Available Schemas:")
        print(f"  Matrix Response Schema: {type(MATRIX_RESPONSE_SCHEMA).__name__}")
        print(f"  Line Render Schema: {type(LINE_RENDER_SCHEMA).__name__}")
        
        # Show default values
        print(f"\n🎛️  Default Matrix Config Keys:")
        for key in MATRIX_RESPONSE_SCHEMA.default_values.keys():
            print(f"    - {key}")
            
        print(f"\n🎨 Default Render Config Keys:")
        for key in LINE_RENDER_SCHEMA.default_values.keys():
            print(f"    - {key}")
            
    except ImportError as e:
        print(f"❌ Configuration utilities not available: {e}")
    except Exception as e:
        print(f"❌ Configuration demo error: {e}")

def demonstrate_mathematical_utilities():
    """Demonstrate mathematical utility functions"""
    print("\n🧮 Mathematical Utilities Demonstration")
    print("=" * 50)
    
    try:
        from core.render_math_utils import (
            calculate_line_score, determine_line_style, calculate_decay,
            adjust_line_thickness, calculate_volatility_score, 
            calculate_trend_strength, determine_line_color
        )
        
        print("✅ Mathematical utilities imported successfully")
        
        # Demonstrate line scoring
        print(f"\n📊 Line Scoring Examples:")
        test_cases = [
            (1.5, 0.2, "High profit, low entropy"),
            (0.3, 0.7, "Low profit, high entropy"),
            (-0.8, 0.4, "Loss, medium entropy"),
            (0.0, 0.5, "Break-even, medium entropy")
        ]
        
        for profit, entropy, description in test_cases:
            score = calculate_line_score(profit, entropy)
            style = determine_line_style(entropy)
            color = determine_line_color(score, entropy)
            print(f"  {description}: score={score:.3f}, style={style}, color={color}")
        
        # Demonstrate decay calculation
        print(f"\n⏰ Time Decay Examples:")
        time_points = [
            (datetime.now() - timedelta(minutes=30), "30 minutes ago"),
            (datetime.now() - timedelta(hours=1), "1 hour ago"),
            (datetime.now() - timedelta(hours=6), "6 hours ago"),
            (datetime.now() - timedelta(days=1), "1 day ago")
        ]
        
        for time_point, description in time_points:
            decay = calculate_decay(time_point, half_life_seconds=3600)
            print(f"  {description}: decay factor={decay:.3f}")
        
        # Demonstrate thickness adjustment
        print(f"\n💾 Resource-Based Thickness Adjustment:")
        memory_levels = [50, 70, 85, 95]
        base_thickness = 4
        
        for memory_pct in memory_levels:
            thickness = adjust_line_thickness(base_thickness, memory_pct)
            print(f"  Memory {memory_pct}%: thickness {base_thickness} → {thickness}")
        
        # Demonstrate volatility calculation
        print(f"\n📈 Volatility Analysis:")
        price_scenarios = [
            ([100, 101, 99, 100, 102], "Stable prices"),
            ([100, 110, 90, 105, 95], "Volatile prices"),
            ([100, 95, 90, 85, 80], "Declining trend"),
            ([100, 105, 110, 115, 120], "Rising trend")
        ]
        
        for prices, description in price_scenarios:
            volatility = calculate_volatility_score(prices)
            trend_strength, trend_direction = calculate_trend_strength(prices)
            print(f"  {description}: volatility={volatility:.4f}, "
                  f"trend={trend_direction} (strength={trend_strength:.3f})")
        
    except ImportError as e:
        print(f"❌ Mathematical utilities not available: {e}")
    except Exception as e:
        print(f"❌ Mathematical demo error: {e}")

def demonstrate_line_render_engine():
    """Demonstrate enhanced LineRenderEngine"""
    print("\n🎨 Line Render Engine Demonstration")
    print("=" * 50)
    
    try:
        from core.line_render_engine import LineRenderEngine
        
        print("✅ LineRenderEngine imported successfully")
        
        # Initialize engine
        engine = LineRenderEngine()
        print(f"  Engine initialized with config keys: {list(engine.config.keys())}")
        
        # Generate sample line data
        sample_lines = []
        for i in range(5):
            # Generate realistic price-like data
            base_price = 100 + random.uniform(-10, 10)
            path = []
            for j in range(20):
                price_change = random.uniform(-2, 2)
                base_price += price_change
                path.append(base_price)
            
            line_data = {
                'path': path,
                'profit': random.uniform(-1, 2),
                'entropy': random.uniform(0.1, 0.9),
                'last_update': datetime.now() - timedelta(minutes=random.randint(5, 120)),
                'type': f'sample_line_{i}',
                'timestamp': datetime.now().isoformat()
            }
            sample_lines.append(line_data)
        
        print(f"\n📊 Rendering {len(sample_lines)} sample lines...")
        
        # Render lines
        start_time = time.time()
        result = engine.render_lines(sample_lines)
        render_time = time.time() - start_time
        
        print(f"  Render completed in {render_time:.3f}s")
        print(f"  Status: {result['status']}")
        print(f"  Lines rendered: {result['lines_rendered_count']}")
        print(f"  System metrics: Memory {result['system_metrics']['memory_percent']:.1f}%, "
              f"CPU {result['system_metrics']['cpu_percent']:.1f}%")
        
        # Show line details
        if result['rendered_lines_details']:
            print(f"\n🔍 Sample Line Details:")
            for i, line_info in enumerate(result['rendered_lines_details'][:3]):
                print(f"  Line {i+1}:")
                print(f"    Score: {line_info['score']:.3f}")
                print(f"    Style: {line_info['style']}")
                print(f"    Color: {line_info['color']}")
                print(f"    Opacity: {line_info['opacity']:.3f}")
                print(f"    Volatility: {line_info['volatility']:.4f}")
                print(f"    Trend: {line_info['trend_direction']} "
                      f"(strength: {line_info['trend_strength']:.3f})")
        
        # Get statistics
        stats = engine.get_line_statistics()
        print(f"\n📈 Engine Statistics:")
        print(f"  Total lines: {stats['total_lines']}")
        print(f"  Active lines: {stats['active_lines']}")
        print(f"  Average score: {stats['average_score']:.3f}")
        print(f"  Average volatility: {stats['average_volatility']:.4f}")
        print(f"  Trend distribution: {stats['trend_distribution']}")
        
    except ImportError as e:
        print(f"❌ LineRenderEngine not available: {e}")
    except Exception as e:
        print(f"❌ LineRenderEngine demo error: {e}")

def demonstrate_matrix_fault_resolver():
    """Demonstrate enhanced MatrixFaultResolver"""
    print("\n🔧 Matrix Fault Resolver Demonstration")
    print("=" * 50)
    
    try:
        from core.matrix_fault_resolver import MatrixFaultResolver
        
        print("✅ MatrixFaultResolver imported successfully")
        
        # Initialize resolver
        resolver = MatrixFaultResolver()
        print(f"  Resolver initialized with retry attempts: {resolver.retry_attempts}")
        
        # Test different fault types
        fault_scenarios = [
            {
                'type': 'data_corruption',
                'severity': 'high',
                'context': {'file': 'matrix_data.bin', 'corruption_level': 0.15}
            },
            {
                'type': 'memory_overflow',
                'severity': 'medium',
                'context': {'memory_usage': '95%', 'operation': 'matrix_multiply'}
            },
            {
                'type': 'computation_error',
                'severity': 'low',
                'context': {'algorithm': 'eigenvalue_decomposition', 'matrix_size': '1000x1000'}
            },
            {
                'type': 'network_timeout',
                'severity': 'medium',
                'context': {'endpoint': 'matrix_api', 'timeout': '5s'}
            },
            {
                'type': 'unknown_error',
                'severity': 'low',
                'context': {'error_code': 'ERR_UNKNOWN_001'}
            }
        ]
        
        print(f"\n🚨 Testing Fault Resolution:")
        
        for i, fault_data in enumerate(fault_scenarios, 1):
            print(f"\n  Fault {i}: {fault_data['type']} (severity: {fault_data['severity']})")
            
            start_time = time.time()
            result = resolver.resolve_faults(fault_data)
            resolution_time = time.time() - start_time
            
            print(f"    Status: {result['status']}")
            print(f"    Method: {result['method']}")
            print(f"    Resolution time: {resolution_time:.3f}s")
            
            if 'action_taken' in result:
                print(f"    Action: {result['action_taken']}")
            
            if result['status'] == 'error':
                print(f"    Error: {result['error']}")
        
        # Get fault statistics
        stats = resolver.get_fault_statistics()
        print(f"\n📊 Fault Resolution Statistics:")
        print(f"  Total resolutions: {stats['total_resolutions']}")
        print(f"  Total errors: {stats['total_errors']}")
        print(f"  Error rate: {stats['error_rate']:.1%}")
        print(f"  Average resolution time: {stats['average_resolution_time']:.3f}s")
        
        if stats['fault_type_distribution']:
            print(f"  Fault type distribution: {stats['fault_type_distribution']}")
        
        if stats['resolution_method_distribution']:
            print(f"  Resolution method distribution: {stats['resolution_method_distribution']}")
        
    except ImportError as e:
        print(f"❌ MatrixFaultResolver not available: {e}")
    except Exception as e:
        print(f"❌ MatrixFaultResolver demo error: {e}")

def demonstrate_system_integration():
    """Demonstrate system integration and monitoring"""
    print("\n🔗 System Integration Demonstration")
    print("=" * 50)
    
    try:
        from core.line_render_engine import LineRenderEngine
        from core.matrix_fault_resolver import MatrixFaultResolver
        
        print("✅ System components imported successfully")
        
        # Initialize components
        engine = LineRenderEngine()
        resolver = MatrixFaultResolver()
        
        print(f"  Components initialized successfully")
        
        # Simulate integrated workflow
        print(f"\n🔄 Simulating Integrated Workflow:")
        
        # Step 1: Generate data with potential issues
        print(f"  1. Generating sample data...")
        sample_data = []
        for i in range(3):
            # Simulate some data corruption
            if random.random() < 0.3:
                print(f"     ⚠️  Simulating data corruption in line {i}")
                fault_result = resolver.resolve_faults({
                    'type': 'data_corruption',
                    'severity': 'medium',
                    'context': {'line_id': i}
                })
                print(f"     ✅ Fault resolved: {fault_result['method']}")
            
            # Generate line data
            path = [100 + random.uniform(-5, 5) for _ in range(10)]
            line_data = {
                'path': path,
                'profit': random.uniform(-0.5, 1.5),
                'entropy': random.uniform(0.2, 0.8),
                'last_update': datetime.now() - timedelta(minutes=random.randint(1, 60)),
                'type': f'integrated_line_{i}'
            }
            sample_data.append(line_data)
        
        # Step 2: Render lines
        print(f"  2. Rendering lines...")
        render_result = engine.render_lines(sample_data)
        print(f"     ✅ Rendered {render_result['lines_rendered_count']} lines")
        
        # Step 3: Monitor system health
        print(f"  3. Monitoring system health...")
        engine_stats = engine.get_line_statistics()
        resolver_stats = resolver.get_fault_statistics()
        
        print(f"     Engine: {engine_stats['active_lines']} active lines, "
              f"avg score: {engine_stats['average_score']:.3f}")
        print(f"     Resolver: {resolver_stats['total_resolutions']} resolutions, "
              f"error rate: {resolver_stats['error_rate']:.1%}")
        
        # Step 4: Cleanup
        print(f"  4. Performing cleanup...")
        cleaned_lines = engine.cleanup_old_lines(max_age_hours=1)
        print(f"     ✅ Cleaned up {cleaned_lines} old lines")
        
        print(f"\n🎉 Integration workflow completed successfully!")
        
    except ImportError as e:
        print(f"❌ System integration not available: {e}")
    except Exception as e:
        print(f"❌ System integration demo error: {e}")

def main():
    """Run all demonstrations"""
    print("Enhanced Schwabot System Demonstration")
    print("=====================================")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all demonstrations
    demonstrate_config_loading()
    demonstrate_mathematical_utilities()
    demonstrate_line_render_engine()
    demonstrate_matrix_fault_resolver()
    demonstrate_system_integration()
    
    print(f"\n🎊 All Demonstrations Complete!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📋 Summary of Enhancements:")
    print(f"  ✅ Standardized YAML configuration loading with schema validation")
    print(f"  ✅ Mathematical utilities for dynamic line rendering")
    print(f"  ✅ Enhanced LineRenderEngine with thermal/profit awareness")
    print(f"  ✅ Robust MatrixFaultResolver with retry logic and monitoring")
    print(f"  ✅ System integration with performance tracking")
    print(f"  ✅ Comprehensive error handling and logging")
    print(f"  ✅ Resource-aware adjustments and cleanup mechanisms")

if __name__ == "__main__":
    main() 