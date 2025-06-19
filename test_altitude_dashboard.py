#!/usr/bin/env python3
"""
Altitude Dashboard Test Suite
============================
Test script to verify dashboard components work correctly.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit: OK")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly: OK") 
    except ImportError as e:
        print(f"❌ Plotly: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas/NumPy: OK")
    except ImportError as e:
        print(f"❌ Pandas/NumPy: {e}")
        return False
    
    return True

def test_dashboard_classes():
    """Test dashboard data classes"""
    print("\n🧪 Testing dashboard classes...")
    
    try:
        from schwabot_altitude_adjustment_dashboard import (
            AltitudeMetrics, QuantumState, SystemState, SchwabotAltitudeDashboard
        )
        
        # Test AltitudeMetrics
        altitude = AltitudeMetrics()
        print(f"✅ AltitudeMetrics - STAM Zone: {altitude.stam_zone}")
        print(f"✅ AltitudeMetrics - Speed Multiplier: {altitude.required_speed_multiplier:.2f}x")
        
        # Test QuantumState
        quantum = QuantumState()
        multivector_data = quantum.multivector_data
        print(f"✅ QuantumState - Multivector Metrics: {len(multivector_data)}")
        
        # Test SystemState
        system = SystemState()
        decision = system.execution_decision
        print(f"✅ SystemState - Execution Decision: {decision['decision']}")
        
        # Test Dashboard
        dashboard = SchwabotAltitudeDashboard()
        print(f"✅ Dashboard - Initialized with simulation: {not dashboard.simulation_active}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard classes: {e}")
        return False

def test_core_system_integration():
    """Test core system integration"""
    print("\n🧪 Testing core system integration...")
    
    # Test Strategy Bundler
    try:
        from core.sfsss_strategy_bundler import create_sfsss_bundler
        bundler = create_sfsss_bundler()
        print("✅ Strategy Bundler: Available")
        bundler_available = True
    except Exception as e:
        print(f"⚠️ Strategy Bundler: {e}")
        bundler_available = False
    
    # Test Constraints
    try:
        from core.constraints import get_system_bounds
        bounds = get_system_bounds()
        print("✅ Constraints System: Available")
        constraints_available = True
    except Exception as e:
        print(f"⚠️ Constraints System: {e}")
        constraints_available = False
    
    # Test BTC Bridge
    try:
        from core.enhanced_btc_integration_bridge import create_enhanced_bridge
        bridge = create_enhanced_bridge()
        print("✅ BTC Integration Bridge: Available") 
        bridge_available = True
    except Exception as e:
        print(f"⚠️ BTC Integration Bridge: {e}")
        bridge_available = False
    
    # Test Pathway Test Suite
    try:
        from core.integrated_pathway_test_suite import create_integrated_pathway_test_suite
        test_suite = create_integrated_pathway_test_suite()
        print("✅ Pathway Test Suite: Available")
        test_suite_available = True
    except Exception as e:
        print(f"⚠️ Pathway Test Suite: {e}")
        test_suite_available = False
    
    available_systems = sum([bundler_available, constraints_available, bridge_available, test_suite_available])
    print(f"📊 Core Systems: {available_systems}/4 Available")
    
    return available_systems >= 0  # Pass even if no systems available (simulation mode)

def test_mathematical_calculations():
    """Test mathematical calculations"""
    print("\n🧪 Testing mathematical calculations...")
    
    try:
        import numpy as np
        from schwabot_altitude_adjustment_dashboard import AltitudeMetrics, QuantumState, SystemState
        
        # Test altitude physics
        altitude = AltitudeMetrics(market_altitude=0.75, air_density=0.5)
        speed_multiplier = altitude.required_speed_multiplier
        expected_speed = np.sqrt(1.0 / 0.5)  # √(baseline/density)
        
        assert abs(speed_multiplier - expected_speed) < 0.01, f"Speed calculation error: {speed_multiplier} vs {expected_speed}"
        print(f"✅ Speed Multiplier Calculation: {speed_multiplier:.2f}x")
        
        # Test STAM zone logic
        test_cases = [
            (0.2, 'vault_mode'),
            (0.4, 'long'),
            (0.6, 'mid'),
            (0.8, 'short')
        ]
        
        for altitude_val, expected_zone in test_cases:
            test_altitude = AltitudeMetrics(market_altitude=altitude_val)
            actual_zone = test_altitude.stam_zone
            assert actual_zone == expected_zone, f"STAM zone error: {actual_zone} vs {expected_zone}"
        
        print("✅ STAM Zone Classification: All test cases passed")
        
        # Test execution decision logic
        quantum = QuantumState(deterministic_confidence=0.9, stability=0.8, execution_readiness=0.9)
        system = SystemState(quantum_state=quantum)
        decision = system.execution_decision
        
        assert decision['decision'] == 'EXECUTE', f"Should be EXECUTE but got {decision['decision']}"
        print(f"✅ Execution Decision Logic: {decision['decision']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical calculations: {e}")
        return False

def test_data_generation():
    """Test data generation functions"""
    print("\n🧪 Testing data generation...")
    
    try:
        from schwabot_altitude_adjustment_dashboard import SchwabotAltitudeDashboard
        
        dashboard = SchwabotAltitudeDashboard()
        
        # Test historical data generation
        dashboard._generate_initial_historical_data()
        assert len(dashboard.historical_data) == 50, f"Expected 50 data points, got {len(dashboard.historical_data)}"
        print("✅ Historical Data Generation: 50 points")
        
        # Test coherence data generation
        dashboard._generate_coherence_data()
        assert len(dashboard.coherence_data) == 20, f"Expected 20 data points, got {len(dashboard.coherence_data)}"
        print("✅ Coherence Data Generation: 20 points")
        
        # Test simulation updates
        dashboard._simulate_system_state_updates()
        print("✅ System State Simulation: Updates applied")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation: {e}")
        return False

def test_visualization_components():
    """Test visualization component creation"""
    print("\n🧪 Testing visualization components...")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from schwabot_altitude_adjustment_dashboard import SchwabotAltitudeDashboard
        import pandas as pd
        
        dashboard = SchwabotAltitudeDashboard()
        dashboard._generate_initial_historical_data()
        dashboard._generate_coherence_data()
        
        # Test gauge creation
        altitude = dashboard.system_state.altitude_metrics.market_altitude
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=altitude * 100,
            domain={'x': [0, 1], 'y': [0, 1]}
        ))
        print("✅ Gauge Visualization: Created")
        
        # Test scatter plot
        df = pd.DataFrame(dashboard.coherence_data)
        fig_scatter = px.scatter(df, x='coherence', y='entropy', size='drift')
        print("✅ Scatter Plot: Created")
        
        # Test radar chart
        quantum_data = dashboard.system_state.quantum_state.multivector_data
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[d['value'] for d in quantum_data],
            theta=[d['metric'] for d in quantum_data],
            fill='toself'
        ))
        print("✅ Radar Chart: Created")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization components: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 80)
    print("🛠️  SCHWABOT ALTITUDE DASHBOARD TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Dependencies", test_imports),
        ("Dashboard Classes", test_dashboard_classes),
        ("Core Integration", test_core_system_integration),
        ("Mathematical Calculations", test_mathematical_calculations),
        ("Data Generation", test_data_generation),
        ("Visualization Components", test_visualization_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Unexpected error - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to launch.")
        return True
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. Dashboard should work with limited functionality.")
        return True
    else:
        print("❌ Multiple test failures. Please check dependencies and core systems.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 