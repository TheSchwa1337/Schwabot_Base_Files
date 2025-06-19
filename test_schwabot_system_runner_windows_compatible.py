#!/usr/bin/env python3
"""
Schwabot System Test Runner - Windows Compatible
===============================================
Windows CLI compatible test runner for the complete Schwabot system.
Runs comprehensive system tests with proper encoding and ASIC output formatting.
"""

import sys
import platform
import os
import subprocess
import io
from typing import Dict, List, Optional, Any
import logging

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
                '🔒': '[LOCKED]',     # Security/locked
                '🛡️': '[SHIELD]',     # Protection/shield
                '👻': '[GHOST]',      # Ghost data
                '⚠️': '[WARNING]',    # Warning indicator
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

class SchwabotSystemTestRunner:
    """
    Windows CLI compatible test runner for Schwabot system
    Handles encoding issues and provides ASIC output formatting
    """
    
    def __init__(self):
        # Windows CLI compatibility handler
        self.cli_handler = WindowsCliCompatibilityHandler()
        
        # Set environment variables for proper encoding
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
    def run_complete_system_test(self) -> bool:
        """Run the complete system test with proper encoding and error handling"""
        try:
            self.cli_handler.log_safe(logger, 'info', "🚀 Starting Schwabot Complete System Test...")
            
            # Run the test and capture output
            result = subprocess.run(
                [sys.executable, 'test_complete_system.py'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Process and display output
            self._display_test_results(result)
            
            return result.returncode == 0
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "complete system test")
            self.cli_handler.log_safe(logger, 'error', error_message)
            return False
    
    def run_alif_aleph_integration_test(self) -> bool:
        """Run ALIF/ALEPH integration test with Windows CLI compatibility"""
        try:
            self.cli_handler.log_safe(logger, 'info', "🔧 Starting ALIF/ALEPH Integration Test...")
            
            # Run the integration test
            result = subprocess.run(
                [sys.executable, 'test_alif_aleph_system_integration.py'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Process and display output
            self._display_test_results(result)
            
            return result.returncode == 0
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "ALIF/ALEPH integration test")
            self.cli_handler.log_safe(logger, 'error', error_message)
            return False
    
    def run_diagnostic_test(self) -> bool:
        """Run quick diagnostic test with Windows CLI compatibility"""
        try:
            self.cli_handler.log_safe(logger, 'info', "🔍 Starting System Diagnostic Test...")
            
            # Run the diagnostic test
            result = subprocess.run(
                [sys.executable, 'test_alif_aleph_system_diagnostic.py'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Process and display output
            self._display_test_results(result)
            
            return result.returncode == 0
            
        except Exception as e:
            error_message = self.cli_handler.safe_format_error(e, "diagnostic test")
            self.cli_handler.log_safe(logger, 'error', error_message)
            return False
    
    def _display_test_results(self, result: subprocess.CompletedProcess):
        """Display test results with Windows CLI compatibility"""
        # Print header
        print("=" * 80)
        print(self.cli_handler.safe_print("📊 SCHWABOT SYSTEM TEST RESULTS"))
        print("=" * 80)
        
        if result.stdout:
            # Process output with ASIC emoji replacement
            safe_output = self.cli_handler.safe_print(result.stdout)
            print(safe_output)
        
        if result.stderr:
            print(self.cli_handler.safe_print("\n⚠️ ERRORS:"))
            safe_error = self.cli_handler.safe_print(result.stderr)
            print(safe_error)
        
        # Display completion status
        status_message = "✅ Test completed successfully" if result.returncode == 0 else "❌ Test completed with errors"
        print(f"\n{self.cli_handler.safe_print(status_message)}")
        print(f"Return code: {result.returncode}")
    
    def run_comprehensive_test_suite(self) -> bool:
        """Run all available tests in sequence"""
        self.cli_handler.log_safe(logger, 'info', "🛠️ COMPREHENSIVE SCHWABOT TEST SUITE")
        self.cli_handler.log_safe(logger, 'info', "=" * 60)
        
        test_suite = [
            ("System Diagnostic", self.run_diagnostic_test),
            ("ALIF/ALEPH Integration", self.run_alif_aleph_integration_test),
            ("Complete System Test", self.run_complete_system_test)
        ]
        
        test_results = []
        
        for test_name, test_function in test_suite:
            self.cli_handler.log_safe(logger, 'info', f"🔄 Running {test_name}...")
            
            try:
                result = test_function()
                test_results.append((test_name, result))
                
                status = "✅ PASS" if result else "❌ FAIL"
                self.cli_handler.log_safe(logger, 'info', f"   {status} - {test_name}")
                
            except Exception as e:
                error_message = self.cli_handler.safe_format_error(e, test_name)
                self.cli_handler.log_safe(logger, 'error', f"   💥 CRASH - {test_name}: {error_message}")
                test_results.append((test_name, False))
        
        # Display comprehensive results
        self._display_comprehensive_results(test_results)
        
        # Return overall success
        return all(result for _, result in test_results)
    
    def _display_comprehensive_results(self, test_results: List[tuple]):
        """Display comprehensive test results summary"""
        self.cli_handler.log_safe(logger, 'info', "\n" + "=" * 60)
        self.cli_handler.log_safe(logger, 'info', "📈 COMPREHENSIVE TEST RESULTS SUMMARY")
        self.cli_handler.log_safe(logger, 'info', "=" * 60)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            self.cli_handler.log_safe(logger, 'info', f"{status} - {test_name}")
        
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        self.cli_handler.log_safe(logger, 'info', f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        if passed_tests == total_tests:
            self.cli_handler.log_safe(logger, 'info', "🎉 ALL TESTS PASSED! Schwabot system is fully operational.")
        elif passed_tests >= total_tests * 0.8:
            self.cli_handler.log_safe(logger, 'info', "⚠️ Most tests passed. System functional with minor issues.")
        else:
            self.cli_handler.log_safe(logger, 'error', "❌ Multiple test failures. System needs attention.")

def run_schwabot_test_suite() -> bool:
    """Main function to run Schwabot test suite with Windows CLI compatibility"""
    runner = SchwabotSystemTestRunner()
    return runner.run_comprehensive_test_suite()

if __name__ == "__main__":
    # Configure logging for test runner
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = run_schwabot_test_suite()
    sys.exit(0 if success else 1) 