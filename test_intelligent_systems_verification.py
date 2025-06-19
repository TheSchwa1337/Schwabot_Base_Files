#!/usr/bin/env python3
"""
Test verification for properly named intelligent systems in DLT Waveform Engine

Tests the following systems:
- PostFailureRecoveryIntelligenceLoop (formerly Gap 4)
- TemporalExecutionCorrectionLayer (formerly Gap 5) 
- MemoryKeyDiagnosticsPipelineCorrector
- WindowsCliCompatibilityHandler with ASIC text output

This ensures all naming conventions are proper \
    and Windows CLI issues are resolved.
"""

import sys
import os
import platform

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_windows_cli_compatibility():
    """Test Windows CLI compatibility handler"""
    print("Testing Windows CLI Compatibility Handler...")
    
    try:
        from dlt_waveform_engine import WindowsCliCompatibilityHandler
        
        cli_handler = WindowsCliCompatibilityHandler()
        
        # Test platform detection
        is_windows = cli_handler.is_windows_cli()
        print(f"[INFO] Windows CLI detected: {is_windows}")
        print(f"[INFO] Platform: {platform.system()}")
        
        # Test emoji conversion (this was causing the CLI issues)
        test_message_with_emojis = "Processing complete ✅ Launch successful 🚀 Data analyzed 📊"
        safe_message = cli_handler.safe_print(test_message_with_emojis)
        print(f"[INFO] Original: {test_message_with_emojis}")
        print(f"[INFO] Safe output: {safe_message}")
        
        # Test error formatting
        test_error = Exception("Test error message")
        safe_error = cli_handler.safe_format_error(test_error, "test context")
        print(f"[INFO] Safe error format: {safe_error}")
        
        print("[SUCCESS] Windows CLI compatibility handler working correctly")
        return True
        
    except Exception as e:
        print(f"[ERROR] Windows CLI compatibility test failed: {e}")
        return False

def test_intelligent_systems_naming():
    """Test properly named intelligent systems"""
    print("\nTesting Properly Named Intelligent Systems...")
    
    try:
        from dlt_waveform_engine import ()
            DLTWaveformEngine,
            PostFailureRecoveryIntelligenceLoop,
            TemporalExecutionCorrectionLayer,
            MemoryKeyDiagnosticsPipelineCorrector
(        )
        
        # Test engine initialization
        engine = DLTWaveformEngine()
        print("[SUCCESS] DLT Waveform Engine initialized")
        
        # Verify intelligent systems are properly integrated
        assert hasattr(engine, 'post_failure_recovery_intelligence_loop')
        assert hasattr(engine, 'temporal_execution_correction_layer')
        assert hasattr(engine, 'memory_key_diagnostics_pipeline_corrector')
        assert hasattr(engine, 'cli_handler')
        
        print("[SUCCESS] PostFailureRecoveryIntelligenceLoop integrated (formerly Gap 4)")
        print("[SUCCESS] TemporalExecutionCorrectionLayer integrated (formerly Gap 5)")
        print("[SUCCESS] MemoryKeyDiagnosticsPipelineCorrector integrated")
        
        # Test status reporting with proper naming
        status = engine.get_comprehensive_intelligent_status()
        expected_keys = [
            'post_failure_recovery_intelligence_loop',
            'temporal_execution_correction_layer', 
            'memory_key_diagnostics_pipeline_corrector',
            'windows_cli_compatibility'
        ]
        
        for key in expected_keys:
            assert key in status, f"Missing status key: {key}"
            print(f"[SUCCESS] Status reporting includes: {key}")
        
        print("[SUCCESS] All intelligent systems properly named \
            and integrated")
        return True
        
    except Exception as e:
        print(f"[ERROR] Intelligent systems naming test failed: {e}")
        return False

def test_error_handling_integration():
    """Test enhanced error handling with Windows CLI compatibility"""
    print("\nTesting Enhanced Error Handling...")
    
    try:
        from dlt_waveform_engine import DLTWaveformEngine
        
        engine = DLTWaveformEngine()
        
        # Test intelligent failure recovery system
        test_failure_context = {
            'entropy': 4.5,
            'coherence': 0.3,
            'profit': -0.02,
            'failure_type': 'test_failure'
        }
        
        recovery_result = engine.enhanced_intelligent_failure_recovery(test_failure_context)
        print(f"[INFO] Intelligent failure recovery result: {recovery_result}")
        
        # Test temporal execution optimization
        optimal_lane = engine.intelligent_temporal_execution_optimization(2.5, 0.7)
        print(f"[INFO] Optimal execution lane selected: {optimal_lane}")
        
        # Test memory key diagnostics
        test_hash = "test_hash_12345"
        test_context = {'entropy': 3.0, 'coherence': 0.5}
        diagnostic_result = engine.intelligent_memory_key_diagnostics(test_hash, test_context)
        print(f"[INFO] Memory key diagnostic result: {diagnostic_result.get('diagnostic_intelligence_level', 'unknown')}")
        
        print("[SUCCESS] Enhanced error handling working correctly")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error handling integration test failed: {e}")
        return False

def test_comprehensive_log_export():
    """Test comprehensive log export with proper naming"""
    print("\nTesting Comprehensive Log Export...")
    
    try:
        from dlt_waveform_engine import DLTWaveformEngine
        
        engine = DLTWaveformEngine()
        
        # Export comprehensive log
        log_output = engine.export_comprehensive_intelligent_log()
        
        # Verify proper naming is documented in the log
        assert 'PostFailureRecoveryIntelligenceLoop' in log_output
        assert 'TemporalExecutionCorrectionLayer' in log_output
        assert 'MemoryKeyDiagnosticsPipelineCorrector' in log_output
        assert 'formerly_referenced_as' in log_output
        assert 'Gap 4' in log_output
        assert 'Priority 4' in log_output
        assert 'Gap 5' in log_output
        assert 'Priority 5' in log_output
        
        print("[SUCCESS] Comprehensive log export includes proper naming documentation")
        print("[INFO] Log preview (first 200 chars):")
        print(log_output[:200] + "...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Comprehensive log export test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("="*60)
    print("INTELLIGENT SYSTEMS VERIFICATION TEST")
    print("Testing proper naming conventions and Windows CLI compatibility")
    print("="*60)
    
    tests = [
        test_windows_cli_compatibility,
        test_intelligent_systems_naming,
        test_error_handling_integration,
        test_comprehensive_log_export
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[CRITICAL ERROR] Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("VERIFICATION, RESULTS:")
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}, {test.__name__}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\nOverall: {success_count}/{total_count} tests passed")
    
    if success_count == total_count:
        print("\n[SUCCESS] All intelligent systems properly implemented!")
        print("- PostFailureRecoveryIntelligenceLoop (formerly Gap 4)")
        print("- TemporalExecutionCorrectionLayer (formerly Gap 5)")
        print("-, MemoryKeyDiagnosticsPipelineCorrector")
        print("- WindowsCliCompatibilityHandler with ASIC text output")
        print("- Proper naming conventions throughout")
        print("- Windows CLI emoji issues resolved")
        return True
    else:
        print(f"\n[WARNING] {total_count - success_count} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 