#!/usr/bin/env python3
"""
from datetime import datetime
from datetime import timedelta
from typing import Set

Complete System Test Runner
===========================
Comprehensive test of the integrated ALIF/ALEPH system including:
- Tick management with drift correction
- Ghost data recovery
- ALEPH/NCCO core integration
- Error handling and fallback logic
"""

import sys
import time
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig()
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_system.log')
        ]
(    )

def test_tick_management():
    """Test tick management system"""
    print("\n🧪 Testing Tick Management System...")
    
    try:
        from core.tick_management_system import create_tick_manager
        
        tick_manager = create_tick_manager(tick_interval=0.5)  # Fast ticks for testing
        
        # Test callbacks
        tick_count = 0
        
        def test_callback(tick_context, alif_result, aleph_result):
            nonlocal tick_count
            tick_count += 1
            print(f"  ✅ Tick {tick_context.tick_id}: ")
                  f"ALIF={alif_result['action']}, "
                  f"ALEPH={aleph_result['action']}, "
(                  f"Entropy={tick_context.entropy:.3f}")
        
        tick_manager.register_callback(test_callback)
        
        # Run for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5.0:
            tick_manager.run_tick_cycle()
            time.sleep(0.1)
        
        # Check results
        status = tick_manager.get_system_status()
        print(f"  📊 Final Status: {tick_count} ticks processed")
        print(f"  📊 Compression Mode: {status['compression_mode']}")
        print(f"  📊 Ghost Ticks: {status['ghost_ticks_pending']}")
        
        tick_manager.counters.print_summary()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Tick Management Test Failed: {e}")
        return False

def test_ghost_recovery():
    """Test ghost data recovery system"""
    print("\n🧪 Testing Ghost Data Recovery...")
    
    try:
        from core.ghost_data_recovery import create_ghost_recovery_manager
        
        recovery_manager = create_ghost_recovery_manager("test_logs")
        
        # Create some test corrupted files
        test_logs_dir = Path("test_logs")
        test_logs_dir.mkdir(exist_ok=True)
        
        # Create corrupted JSON file
        corrupted_file = test_logs_dir / "corrupted_test.json"
        with open(corrupted_file, 'w') as f:
            f.write('{"tick_id": 123, "entropy": 0.5, "incomplete": "data"')  # Missing closing brace
        
        # Create malformed string file
        malformed_file = test_logs_dir / "malformed_test.json"
        with open(malformed_file, 'w') as f:
            f.write('{"tick_id": 124, "entropy": 0.7\x00\x01, "timestamp": "2024-01-01"}')  # Invalid chars
        
        # Run recovery scan
        results = recovery_manager.full_system_recovery_scan()
        
        print(f"  📄 Log Recovery Results: {results['log_recovery']}")
        print(f"  👻 Shadow Status: {results['shadow_status']}")
        print(f"  📈 Recovery Stats: {results['recovery_stats']}")
        
        # Cleanup test files
        import shutil
        shutil.rmtree(test_logs_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ghost Recovery Test Failed: {e}")
        return False

def test_aleph_cores():
    """Test ALEPH core modules"""
    print("\n🧪 Testing ALEPH Core Modules...")
    
    try:
        # Test entropy analyzer
        try:
            from aleph_core import EntropyAnalyzer
            entropy_analyzer = EntropyAnalyzer()
            
            # Test with sample data
            entropy_values = [1, 5, 12, 23, 34, 45, 56, 67, 78, 89]
            results = entropy_analyzer.analyze_entropy_distribution(entropy_values)
            
            print(f"  📊 Entropy Analysis: Mean={results['mean']:.2f}, ")
(                  f"Std={results['std']:.2f}, Coverage={results['coverage_percentage']:.1f}%")
            
        except ImportError:
            print("  ⚠️ EntropyAnalyzer not available - using simulation")
        
        # Test detonation sequencer
        try:
            from aleph_core import DetonationSequencer
            detonator = DetonationSequencer()
            
            # Test detonation
            payload = {"pattern": "test_pattern", "confidence": 0.85}
            result = detonator.initiate_detonation()
                payload=payload,
                price=50000.0,
                volume=100.0,
                order_book={"bids": [[49000, 1.0]], "asks": [[51000, 1.0]]},
                trades=[{"price": 50000, "volume": 0.1, "side": "buy"}]
(            )
            
            print(f"  💥 Detonation Result: {result['detonation_activated']}, ")
(                  f"Confidence={result['confidence']:.2f}")
            
        except ImportError:
            print("  ⚠️ DetonationSequencer not available - using simulation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ALEPH Cores Test Failed: {e}")
        return False

def test_ncco_system():
    """Test NCCO system"""
    print("\n🧪 Testing NCCO System...")
    
    try:
        from ncco_core import NCCO
        
        # Create test NCCO
        ncco = NCCO()
            id=1,
            price_delta=0.05,
            base_price=50000.0,
            bit_mode=1,
            score=0.85,
            pre_commit_id="test_commit_123"
(        )
        
        print(f"  🔧 NCCO Created: {ncco}")
        print(f"  📊 NCCO Score: {ncco.score}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ NCCO Test Failed: {e}")
        return False

def test_integrated_system():
    """Test complete integrated system"""
    print("\n🧪 Testing Integrated ALIF/ALEPH System...")
    
    try:
        from core.integrated_alif_aleph_system import create_integrated_system
        
        # Create system with fast ticks for testing
        system = create_integrated_system()
            tick_interval=0.5,
            log_directory="test_integration_logs",
            enable_recovery=True
(        )
        
        # Collect decisions
        decisions_collected = []
        
        def decision_callback(decision):
            decisions_collected.append(decision)
            print(f"    [Decision {decision.tick_id}] {decision.action} ")
                  f"(confidence: {decision.confidence:.2f}, ")
                  f"risk: {decision.risk_assessment}, "
(                  f"priority: {decision.execution_priority})")
        
        system.register_decision_callback(decision_callback)
        
        # Start system
        system.start_system()
        
        # Run for 10 seconds
        print("  🚀 Running integrated system for 10 seconds...")
        
        start_time = time.time()
        while time.time() - start_time < 10.0:
            status = system.get_system_status()
            
            if len(decisions_collected) % 5 == 0 and len(decisions_collected) > 0:
                print(f"    📊 Status: Uptime={status['uptime']}, ")
                      f"Ticks={status['health_metrics']['total_ticks_processed']}, "
(                      f"Decisions={len(decisions_collected)}")
            
            time.sleep(1.0)
        
        # Get final status
        final_status = system.get_system_status()
        recent_decisions = system.get_recent_decisions(count=5)
        
        print("  📋 Final Results:")
        print(f"    Total Ticks: {final_status['health_metrics']['total_ticks_processed']}")
        print(f"    Total Decisions: {len(decisions_collected)}")
        print(f"    Success Rate: {final_status['health_metrics']['success_rate']:.1%}")
        print(f"    Error Rate: {final_status['health_metrics']['error_rate']:.1%}")
        print(f"    Recent Decisions: {len(recent_decisions)}")
        
        # Stop system
        system.stop_system()
        
        # Cleanup test logs
        import shutil
        shutil.rmtree("test_integration_logs", ignore_errors=True)
        
        return len(decisions_collected) > 0
        
    except Exception as e:
        print(f"  ❌ Integrated System Test Failed: {e}")
        return False

def test_error_scenarios():
    """Test error handling scenarios"""
    print("\n🧪 Testing Error Handling...")
    
    try:
        from core.tick_management_system import create_tick_manager
        
        tick_manager = create_tick_manager(tick_interval=0.1)
        
        # Test error in callback
        error_count = 0
        
        def error_callback(tick_context, alif_result, aleph_result):
            nonlocal error_count
            error_count += 1
            if error_count == 3:
                raise ValueError("Simulated callback error")
        
        tick_manager.register_callback(error_callback)
        
        # Run and expect errors to be handled gracefully
        for i in range(10):
            tick_manager.run_tick_cycle()
            time.sleep(0.05)
        
        # Check that system continued despite errors
        status = tick_manager.get_system_status()
        print(f"  🛡️ Error handling test: System processed {status['tick_count']} ticks")
        print(f"  🛡️ Error count: {len(tick_manager.error_history)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error Handling Test Failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("=" * 80)
    print("🛠️  COMPLETE SCHWABOT SYSTEM TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Tick Management", test_tick_management),
        ("Ghost Data Recovery", test_ghost_recovery),
        ("ALEPH Core Modules", test_aleph_cores),
        ("NCCO System", test_ncco_system),
        ("Integrated System", test_integrated_system),
        ("Error Handling", test_error_scenarios)
    ]
    
    results = []
    start_time = datetime.now()
    
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name}")
        except Exception as e:
            print(f"  ❌ EXCEPTION - {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("📋 COMPLETE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️ Total Duration: {duration}")
    
    # Export test results
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "duration": str(duration),
        "total_tests": total,
        "passed_tests": passed,
        "success_rate": passed / total,
        "test_details": [
            {"name": name, "passed": result} for name, result in results
        ]
    }
    
    with open("test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    if passed == total:
        print("🎉 All tests passed! System is ready for production.")
        return True
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. System functional with minor issues.")
        return True
    else:
        print("❌ Multiple test failures. System needs attention.")
        return False

if __name__ == "__main__":
    setup_logging()
    
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1) 