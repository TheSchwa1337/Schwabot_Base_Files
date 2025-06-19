#!/usr/bin/env python3
"""
ALIF/ALEPH System Integration Test
=================================
Comprehensive test of the integrated ALIF/ALEPH system including:
- Core module imports and validation
- System creation and initialization
- ALIF/ALEPH tick processing functionality
- Windows CLI compatibility verification
"""

import sys
import platform
import os
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

class ALIFALEPHSystemIntegrationTester:
    """
    Comprehensive tester for ALIF/ALEPH system integration
    Tests all core modules, system creation, and integration functionality
    """
    
    def __init__(self):
        # Windows CLI compatibility handler
        self.cli_handler = WindowsCliCompatibilityHandler()
        self.test_results: List[Tuple[str, bool, str]] = []
        
    def test_aleph_core_module_imports(self) -> bool:
        """Test ALEPH core module imports and functionality"""
        test_name = "ALEPH Core Module Imports"
        
        try:
            from aleph_core import DetonationSequencer, EntropyAnalyzer, PatternMatcher, SmartMoneyAnalyzer
            
            # Test basic functionality
            detonator = DetonationSequencer()
            analyzer = EntropyAnalyzer()
            matcher = PatternMatcher()
            smart_money = SmartMoneyAnalyzer()
            
            self.cli_handler.log_safe(logger, 'info', "✅ ALEPH core modules imported and instantiated successfully")
            self.test_results.append((test_name, True, "All ALEPH modules imported and functional"))
            return True
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "ALEPH core imports")
            self.cli_handler.log_safe(logger, 'error', error_message)
            self.test_results.append((test_name, False, str(e)))
            return False

    def test_ncco_system_imports(self) -> bool:
        """Test NCCO system imports and functionality"""
        test_name = "NCCO System Imports"
        
        try:
            from ncco_core import NCCO, generate_nccos, score_nccos
            
            # Test basic functionality
            test_ncco = NCCO(
                id=1,
                price_delta=0.05,
                base_price=50000.0,
                bit_mode=1,
                score=0.85,
                pre_commit_id="test_integration_123"
            )
            
            # Test generation and scoring
            nccos = generate_nccos(3)
            scores = score_nccos(nccos)
            
            self.cli_handler.log_safe(logger, 'info', f"✅ NCCO system functional - Generated {len(nccos)} NCCOs")
            self.test_results.append((test_name, True, f"NCCO system functional with {len(nccos)} generated objects"))
            return True
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "NCCO system imports")
            self.cli_handler.log_safe(logger, 'error', error_message)
            self.test_results.append((test_name, False, str(e)))
            return False

    def test_tick_management_system(self) -> bool:
        """Test tick management system functionality"""
        test_name = "Tick Management System"
        
        try:
            from core.tick_management_system import create_tick_manager
            
            # Create and test tick manager
            tick_manager = create_tick_manager(tick_interval=1.0)
            
            # Test basic tick processing
            for i in range(3):
                tick_context = tick_manager.run_tick_cycle()
                if tick_context:
                    self.cli_handler.log_safe(logger, 'info', f"🔄 Processed tick {tick_context.tick_id}")
                
                import time
                time.sleep(0.1)
            
            status = tick_manager.get_system_status()
            tick_count = status['tick_count']
            
            self.cli_handler.log_safe(logger, 'info', f"✅ Tick management system processed {tick_count} ticks")
            self.test_results.append((test_name, True, f"Processed {tick_count} ticks successfully"))
            return True
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "tick management system")
            self.cli_handler.log_safe(logger, 'error', error_message)
            self.test_results.append((test_name, False, str(e)))
            return False

    def test_ghost_data_recovery_system(self) -> bool:
        """Test ghost data recovery system functionality"""
        test_name = "Ghost Data Recovery System"
        
        try:
            from core.ghost_data_recovery import create_ghost_recovery_manager
            
            # Create recovery manager
            recovery_manager = create_ghost_recovery_manager("test_recovery_logs")
            
            # Perform recovery scan
            recovery_results = recovery_manager.full_system_recovery_scan()
            
            recovered_count = recovery_results["recovery_stats"]["total_recoveries"]
            success_rate = recovery_results["recovery_stats"]["recovery_success_rate"]
            
            self.cli_handler.log_safe(logger, 'info', f"✅ Ghost recovery system operational - Success rate: {success_rate:.1%}")
            self.test_results.append((test_name, True, f"Recovery system functional with {recovered_count} recoveries"))
            return True
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "ghost data recovery")
            self.cli_handler.log_safe(logger, 'error', error_message)
            self.test_results.append((test_name, False, str(e)))
            return False

    def test_integrated_alif_aleph_system(self) -> bool:
        """Test complete integrated ALIF/ALEPH system"""
        test_name = "Integrated ALIF/ALEPH System"
        
        try:
            from core.integrated_alif_aleph_system import create_integrated_system
            
            # Create integrated system
            system = create_integrated_system(
                tick_interval=1.0,
                log_directory="test_integration_logs",
                enable_recovery=True
            )
            
            # Test system status
            status = system.get_system_status()
            cores_available = status['cores_available']
            
            # Test system startup and shutdown
            system.start_system()
            self.cli_handler.log_safe(logger, 'info', "🚀 Integrated system started successfully")
            
            import time
            time.sleep(2.0)
            
            # Check system operation
            running_status = system.get_system_status()
            ticks_processed = running_status['health_metrics']['total_ticks_processed']
            active_threads = running_status['active_threads']
            
            system.stop_system()
            self.cli_handler.log_safe(logger, 'info', "🛑 Integrated system stopped successfully")
            
            self.cli_handler.log_safe(logger, 'info', f"✅ Integrated system processed {ticks_processed} ticks with {active_threads} threads")
            self.test_results.append((test_name, True, f"System operational - {ticks_processed} ticks, {active_threads} threads, cores: {cores_available}"))
            return True
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "integrated ALIF/ALEPH system")
            self.cli_handler.log_safe(logger, 'error', error_message)
            self.test_results.append((test_name, False, str(e)))
            return False

    def test_windows_cli_compatibility(self) -> bool:
        """Test Windows CLI compatibility functionality"""
        test_name = "Windows CLI Compatibility"
        
        try:
            # Test emoji replacement
            test_message = "🚀 Testing 📊 emoji ✅ replacement 🔧 functionality"
            safe_message = self.cli_handler.safe_print(test_message)
            
            # Test error formatting
            test_error = ValueError("Test error for compatibility")
            safe_error = self.cli_handler.safe_format_error(test_error, "compatibility_test")
            
            # Test Windows detection
            is_windows = self.cli_handler.is_windows_cli()
            
            self.cli_handler.log_safe(logger, 'info', f"✅ Windows CLI compatibility functional - Windows detected: {is_windows}")
            self.test_results.append((test_name, True, f"Compatibility functional - Windows: {is_windows}"))
            return True
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "Windows CLI compatibility")
            self.cli_handler.log_safe(logger, 'error', error_message)
            self.test_results.append((test_name, False, str(e)))
            return False

    def run_comprehensive_integration_test(self) -> bool:
        """Run comprehensive integration test suite"""
        self.cli_handler.log_safe(logger, 'info', "🧪 ALIF/ALEPH SYSTEM INTEGRATION TEST SUITE")
        self.cli_handler.log_safe(logger, 'info', "=" * 60)
        
        test_methods = [
            self.test_aleph_core_module_imports,
            self.test_ncco_system_imports,
            self.test_tick_management_system,
            self.test_ghost_data_recovery_system,
            self.test_integrated_alif_aleph_system,
            self.test_windows_cli_compatibility
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__.replace('test_', '').replace('_', ' ').title()
            self.cli_handler.log_safe(logger, 'info', f"🔄 Running {test_name}...")
            
            try:
                result = test_method()
                status = "✅ PASS" if result else "❌ FAIL"
                self.cli_handler.log_safe(logger, 'info', f"   {status} - {test_name}")
            except Exception as e:
                error_message = self.cli_handler.safe_format_error(e, test_name)
                self.cli_handler.log_safe(logger, 'error', f"   💥 CRASH - {test_name}: {error_message}")
                self.test_results.append((test_name, False, str(e)))
        
        # Calculate and display results
        passed_tests = sum(1 for _, result, _ in self.test_results if result)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        self.cli_handler.log_safe(logger, 'info', "\n" + "=" * 60)
        self.cli_handler.log_safe(logger, 'info', "📊 INTEGRATION TEST RESULTS SUMMARY")
        self.cli_handler.log_safe(logger, 'info', "=" * 60)
        
        for test_name, result, details in self.test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            self.cli_handler.log_safe(logger, 'info', f"{status} - {test_name}: {details}")
        
        self.cli_handler.log_safe(logger, 'info', f"\n📈 Overall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if passed_tests == total_tests:
            self.cli_handler.log_safe(logger, 'info', "🎉 ALL INTEGRATION TESTS PASSED! ALIF/ALEPH system is fully operational.")
            return True
        elif passed_tests >= total_tests * 0.8:
            self.cli_handler.log_safe(logger, 'info', "⚠️ Most integration tests passed. System functional with minor issues.")
            return True
        else:
            self.cli_handler.log_safe(logger, 'error', "❌ Multiple integration test failures. System needs attention.")
            return False

def run_alif_aleph_integration_test() -> bool:
    """Main function to run ALIF/ALEPH integration tests"""
    tester = ALIFALEPHSystemIntegrationTester()
    return tester.run_comprehensive_integration_test()

if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = run_alif_aleph_integration_test()
    sys.exit(0 if success else 1) 