#!/usr/bin/env python3
"""
ALIF/ALEPH System Diagnostic Test
================================
Quick diagnostic test to identify specific issues in the integrated ALIF/ALEPH system.
Tests core functionality and provides detailed error reporting for troubleshooting.
"""

import sys
import platform
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent))

# =====================================
# WINDOWS CLI COMPATIBILITY HANDLER
# =====================================

class WindowsCliCompatibilityHandler:
    """
    Handles Windows CLI compatibility issues including emoji rendering
    and ASIC implementation for plain text output explanations
    
    Addresses the CLI error issues mentioned in the comprehensive testing:
    - Emoji characters causing encoding errors on Windows
    - Need for ASIC plain text output
    - Cross-platform compatibility for error messages
    """
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (platform.system() == "Windows" and 
                ("cmd" in os.environ.get("COMSPEC", "").lower() or
                 "powershell" in os.environ.get("PSModulePath", "").lower()))
    
    @staticmethod
    def safe_print(message: str, use_emoji: bool = True) -> str:
        """
        Print message safely with Windows CLI compatibility
        Implements ASIC plain text output for Windows environments
        
        ASIC Implementation: Application-Specific Integrated Circuit approach
        provides specialized text rendering for Windows CLI environments
        """
        if WindowsCliCompatibilityHandler.is_windows_cli() and use_emoji:
            # ASIC plain text markers for Windows CLI compatibility
            emoji_to_asic_mapping = {
                '✅': '[SUCCESS]',    # Success indicator
                '❌': '[ERROR]',      # Error indicator  
                '🔧': '[PROCESSING]', # Processing indicator
                '🚀': '[LAUNCH]',     # Launch/start indicator
                '🎉': '[COMPLETE]',   # Completion indicator
                '💥': '[CRITICAL]',   # Critical alert
                '⚡': '[FAST]',       # Fast execution
                '🔍': '[SEARCH]',     # Search/analysis
                '📊': '[DATA]',       # Data processing
                '🧪': '[TEST]',       # Testing indicator
                '🛠️': '[TOOLS]',      # Tools/utilities
                '⚖️': '[BALANCE]',    # Balance/measurement
                '🔄': '[CYCLE]',      # Cycle/loop
                '🎯': '[TARGET]',     # Target/goal
                '📈': '[PROFIT]',     # Profit indicator
                '🔥': '[HOT]',        # High activity
                '❄️': '[COOL]',       # Cool/low activity
                '⭐': '[STAR]',       # Important/featured
            }
            
            safe_message = message
            for emoji, asic_replacement in emoji_to_asic_mapping.items():
                safe_message = safe_message.replace(emoji, asic_replacement)
            
            return safe_message
        
        return message
    
    @staticmethod
    def log_safe(logger, level: str, message: str):
        """Log message safely with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_print(message)
        try:
            getattr(logger, level.lower())(safe_message)
        except UnicodeEncodeError:
            # Emergency ASCII fallback for Windows CLI
            ascii_message = safe_message.encode('ascii', errors='replace').decode('ascii')
            getattr(logger, level.lower())(ascii_message)
    
    @staticmethod
    def safe_format_error(error: Exception, context: str = "") -> str:
        """Format error messages safely for Windows CLI"""
        error_message = f"Error: {str(error)}"
        if context:
            error_message += f" | Context: {context}"
        
        return WindowsCliCompatibilityHandler.safe_print(error_message)

logger = logging.getLogger(__name__)

class ALIFALEPHSystemDiagnosticTester:
    """
    Quick diagnostic tester for ALIF/ALEPH system
    Provides rapid testing of critical imports and system creation
    """
    
    def __init__(self):
        # Windows CLI compatibility handler
        self.cli_handler = WindowsCliCompatibilityHandler()
        
    def test_critical_imports(self) -> bool:
        """Test all critical module imports"""
        self.cli_handler.log_safe(logger, 'info', "🧪 Testing Critical Imports...")
        
        try:
            from aleph_core import DetonationSequencer, EntropyAnalyzer, PatternMatcher, SmartMoneyAnalyzer
            self.cli_handler.log_safe(logger, 'info', "  ✅ ALEPH core modules imported")
        except Exception as e:
            self.cli_handler.log_safe(logger, 'error', f"  ❌ ALEPH import failed: {e}")
            return False
        
        try:
            from ncco_core import NCCO, generate_nccos, score_nccos
            self.cli_handler.log_safe(logger, 'info', "  ✅ NCCO core modules imported")
        except Exception as e:
            self.cli_handler.log_safe(logger, 'error', f"  ❌ NCCO import failed: {e}")
            return False
        
        try:
            from core.tick_management_system import create_tick_manager
            self.cli_handler.log_safe(logger, 'info', "  ✅ Tick management imported")
        except Exception as e:
            self.cli_handler.log_safe(logger, 'error', f"  ❌ Tick management import failed: {e}")
            return False
        
        try:
            from core.ghost_data_recovery import create_ghost_recovery_manager
            self.cli_handler.log_safe(logger, 'info', "  ✅ Ghost recovery imported")
        except Exception as e:
            self.cli_handler.log_safe(logger, 'error', f"  ❌ Ghost recovery import failed: {e}")
            return False
        
        try:
            from core.integrated_alif_aleph_system import create_integrated_system
            self.cli_handler.log_safe(logger, 'info', "  ✅ Integrated system imported")
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "integrated system import")
            self.cli_handler.log_safe(logger, 'error', f"  ❌ Integrated system import failed: {error_message}")
            traceback.print_exc()
            return False
        
        return True

    def test_system_creation(self) -> bool:
        """Test system creation and status retrieval"""
        self.cli_handler.log_safe(logger, 'info', "\n🧪 Testing System Creation...")
        
        try:
            from core.integrated_alif_aleph_system import create_integrated_system
            
            system = create_integrated_system(
                tick_interval=1.0,
                log_directory="test_diagnostic_logs",
                enable_recovery=True
            )
            
            self.cli_handler.log_safe(logger, 'info', "  ✅ Integrated system created successfully")
            
            # Test system status
            status = system.get_system_status()
            status_keys = list(status.keys())
            self.cli_handler.log_safe(logger, 'info', f"  ✅ System status retrieved: {status_keys}")
            
            return True
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "system creation")
            self.cli_handler.log_safe(logger, 'error', f"  ❌ System creation failed: {error_message}")
            traceback.print_exc()
            return False

    def test_system_start_stop_cycle(self) -> bool:
        """Test complete system start/stop cycle"""
        self.cli_handler.log_safe(logger, 'info', "\n🧪 Testing System Start/Stop...")
        
        try:
            from core.integrated_alif_aleph_system import create_integrated_system
            
            system = create_integrated_system(tick_interval=0.5)
            
            # Start system
            system.start_system()
            self.cli_handler.log_safe(logger, 'info', "  ✅ System started successfully")
            
            # Let it run briefly
            import time
            time.sleep(2.0)
            
            # Check status
            status = system.get_system_status()
            ticks_processed = status['health_metrics']['total_ticks_processed']
            self.cli_handler.log_safe(logger, 'info', f"  ✅ System running - ticks: {ticks_processed}")
            
            # Stop system
            system.stop_system()
            self.cli_handler.log_safe(logger, 'info', "  ✅ System stopped successfully")
            
            return True
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "system start/stop")
            self.cli_handler.log_safe(logger, 'error', f"  ❌ System start/stop failed: {error_message}")
            traceback.print_exc()
            return False

    def run_diagnostic_test_suite(self) -> bool:
        """Run complete diagnostic test suite"""
        self.cli_handler.log_safe(logger, 'info', "🔍 ALIF/ALEPH SYSTEM DIAGNOSTIC TEST")
        self.cli_handler.log_safe(logger, 'info', "=" * 50)
        
        diagnostic_tests = [
            ("Import Test", self.test_critical_imports),
            ("System Creation", self.test_system_creation),
            ("Start/Stop Test", self.test_system_start_stop_cycle)
        ]
        
        test_results = []
        for test_name, test_function in diagnostic_tests:
            try:
                result = test_function()
                test_results.append((test_name, result))
            except Exception as e:
                error_message = self.cli_handler.safe_format_error(e, test_name)
                self.cli_handler.log_safe(logger, 'error', f"  💥 {test_name} crashed: {error_message}")
                traceback.print_exc()
                test_results.append((test_name, False))
        
        # Display results summary
        self.cli_handler.log_safe(logger, 'info', "\n" + "=" * 50)
        self.cli_handler.log_safe(logger, 'info', "📋 DIAGNOSTIC RESULTS")
        self.cli_handler.log_safe(logger, 'info', "=" * 50)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            self.cli_handler.log_safe(logger, 'info', f"{status} - {test_name}")
        
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        self.cli_handler.log_safe(logger, 'info', f"\n📊 Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        return passed_tests == total_tests

def run_alif_aleph_diagnostic_test() -> bool:
    """Main function to run ALIF/ALEPH diagnostic tests"""
    tester = ALIFALEPHSystemDiagnosticTester()
    return tester.run_diagnostic_test_suite()

if __name__ == "__main__":
    # Configure logging for diagnostic output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = run_alif_aleph_diagnostic_test()
    sys.exit(0 if success else 1) 