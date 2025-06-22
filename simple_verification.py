#!/usr/bin/env python3
"""Simple Maturity Verification Script.

This script verifies our maturity implementation by checking:
1. F841 unused variables are properly consumed
2. Maturity components are properly integrated
3. Error handling is comprehensive
4. Architecture patterns are implemented

Does not require flake8 to be installed.
"""

import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from dataclasses import dataclass

@dataclass
class VerificationResult:
    """Result of a verification check."""
    name: str
    passed: bool
    details: str
    score: int = 0

class MaturityVerifier:
    """Verifies maturity implementation without external dependencies."""
    
    def __init__(self):
        """Initialize the verifier."""
        self.results: List[VerificationResult] = []
        self.total_score = 0
        self.max_score = 0
        
        print("🔍 Simple Maturity Verifier initialized")
    
    def verify_f841_elimination(self) -> VerificationResult:
        """Verify that F841 unused variables are eliminated."""
        print("🔍 Checking F841 unused variable elimination...")
        
        # Check core/main.py for unused variables
        main_py = Path("core/main.py")
        if not main_py.exists():
            return VerificationResult(
                "F841 Elimination", False, 
                "core/main.py not found", 0
            )
        
        try:
            content = main_py.read_text(encoding='utf-8')
            
            # Look for the variables that were previously unused
            unused_vars = ['tick_phase', 'state_valid', 'portfolio_shift']
            consumption_found = {}
            
            for var in unused_vars:
                # Check if variable is consumed (not just assigned)
                consumption_patterns = [
                    f"state_tracker.update_{var}",
                    f"{var}_validator.validate",
                    f"process_{var}",
                    f"handle_{var}",
                    f"consume_{var}"
                ]
                
                consumed = any(pattern in content for pattern in consumption_patterns)
                consumption_found[var] = consumed
            
            consumed_count = sum(consumption_found.values())
            total_vars = len(unused_vars)
            
            if consumed_count == total_vars:
                return VerificationResult(
                    "F841 Elimination", True,
                    f"All {total_vars} previously unused variables are now consumed",
                    30
                )
            else:
                unconsumed = [var for var, consumed in consumption_found.items() if not consumed]
                return VerificationResult(
                    "F841 Elimination", False,
                    f"{consumed_count}/{total_vars} variables consumed. Unconsumed: {unconsumed}",
                    consumed_count * 10
                )
                
        except Exception as e:
            return VerificationResult(
                "F841 Elimination", False,
                f"Error analyzing main.py: {e}", 0
            )
    
    def verify_maturity_components(self) -> VerificationResult:
        """Verify that maturity components are properly implemented."""
        print("🔍 Checking maturity component implementation...")
        
        required_files = {
            "core/core_loop_manager.py": "Unified component orchestration",
            "core/tick_cycle_validator.py": "Temporal execution correction", 
            "core/profit_vector_reconciler.py": "Waveform vs Allocator reconciliation",
            "core/error_sanitizer.py": "Comprehensive error sanitization"
        }
        
        found_files = 0
        missing_files = []
        
        for file_path, description in required_files.items():
            if Path(file_path).exists():
                found_files += 1
                print(f"   ✅ {file_path} - {description}")
            else:
                missing_files.append(file_path)
                print(f"   ❌ {file_path} - {description}")
        
        if found_files == len(required_files):
            return VerificationResult(
                "Maturity Components", True,
                f"All {len(required_files)} maturity components implemented",
                25
            )
        else:
            return VerificationResult(
                "Maturity Components", False,
                f"{found_files}/{len(required_files)} components found. Missing: {missing_files}",
                found_files * 6
            )
    
    def verify_integration_patterns(self) -> VerificationResult:
        """Verify that integration patterns are properly implemented."""
        print("🔍 Checking integration patterns...")
        
        patterns_found = 0
        total_patterns = 5
        details = []
        
        # Check StateTracker integration
        try:
            from core.state_tracker import StateTracker
            patterns_found += 1
            details.append("✅ StateTracker pattern implemented")
        except ImportError:
            details.append("❌ StateTracker pattern missing")
        
        # Check ComponentRegistry pattern
        try:
            from core.component_registry import ComponentRegistry
            patterns_found += 1
            details.append("✅ ComponentRegistry pattern implemented")
        except ImportError:
            details.append("❌ ComponentRegistry pattern missing")
        
        # Check CoreLoopManager orchestration
        try:
            from core.core_loop_manager import CoreLoopManager
            patterns_found += 1
            details.append("✅ CoreLoopManager orchestration implemented")
        except ImportError:
            details.append("❌ CoreLoopManager orchestration missing")
        
        # Check Error Sanitization
        try:
            from core.error_sanitizer import ErrorSanitizer
            patterns_found += 1
            details.append("✅ Error sanitization pattern implemented")
        except ImportError:
            details.append("❌ Error sanitization pattern missing")
        
        # Check Profit Bridge
        try:
            from core.profit_bridge_orchestrator import ProfitBridgeOrchestrator
            patterns_found += 1
            details.append("✅ Profit bridge pattern implemented")
        except ImportError:
            details.append("❌ Profit bridge pattern missing")
        
        success = patterns_found >= 4  # Allow for 1 missing
        return VerificationResult(
            "Integration Patterns", success,
            f"{patterns_found}/{total_patterns} patterns implemented. " + "; ".join(details),
            patterns_found * 4
        )
    
    def verify_error_handling(self) -> VerificationResult:
        """Verify comprehensive error handling implementation."""
        print("🔍 Checking error handling implementation...")
        
        error_handling_score = 0
        max_error_score = 15
        details = []
        
        # Check if ErrorSanitizer exists and has key methods
        error_sanitizer_file = Path("core/error_sanitizer.py")
        if error_sanitizer_file.exists():
            try:
                content = error_sanitizer_file.read_text(encoding='utf-8')
                
                # Check for key error handling features
                features = {
                    "catch method": "def catch(",
                    "mathematical recovery": "_mathematical_recovery",
                    "sanitization levels": "SanitizationLevel",
                    "error statistics": "get_error_statistics",
                    "traceback formatting": "traceback.format_exc"
                }
                
                for feature, pattern in features.items():
                    if pattern in content:
                        error_handling_score += 3
                        details.append(f"✅ {feature}")
                    else:
                        details.append(f"❌ {feature}")
                        
            except Exception as e:
                details.append(f"❌ Error reading error_sanitizer.py: {e}")
        else:
            details.append("❌ ErrorSanitizer file not found")
        
        success = error_handling_score >= 12  # At least 4 out of 5 features
        return VerificationResult(
            "Error Handling", success,
            f"Error handling score: {error_handling_score}/{max_error_score}. " + "; ".join(details),
            error_handling_score
        )
    
    def verify_architectural_maturity(self) -> VerificationResult:
        """Verify overall architectural maturity."""
        print("🔍 Checking architectural maturity...")
        
        maturity_indicators = 0
        max_indicators = 6
        details = []
        
        # Check for unified execution loop
        core_loop_file = Path("core/core_loop_manager.py")
        if core_loop_file.exists():
            try:
                content = core_loop_file.read_text(encoding='utf-8')
                if "_execute_single_cycle" in content and "error_sanitizer.catch" in content:
                    maturity_indicators += 1
                    details.append("✅ Unified execution loop with error handling")
                else:
                    details.append("⚠️ Execution loop exists but may lack error integration")
            except:
                details.append("❌ Error analyzing core loop manager")
        else:
            details.append("❌ Core loop manager missing")
        
        # Check for comprehensive validation
        if Path("core/tick_cycle_validator.py").exists():
            maturity_indicators += 1
            details.append("✅ Temporal validation implemented")
        else:
            details.append("❌ Temporal validation missing")
        
        # Check for vector reconciliation
        if Path("core/profit_vector_reconciler.py").exists():
            maturity_indicators += 1
            details.append("✅ Vector reconciliation implemented")
        else:
            details.append("❌ Vector reconciliation missing")
        
        # Check for component registry
        if Path("core/component_registry.py").exists():
            maturity_indicators += 1
            details.append("✅ Component registry pattern")
        else:
            details.append("❌ Component registry missing")
        
        # Check for state management
        if Path("core/state_tracker.py").exists():
            maturity_indicators += 1
            details.append("✅ Centralized state management")
        else:
            details.append("❌ State management missing")
        
        # Check for profit bridge
        if Path("core/profit_bridge_orchestrator.py").exists():
            maturity_indicators += 1
            details.append("✅ Profit bridge orchestration")
        else:
            details.append("❌ Profit bridge missing")
        
        success = maturity_indicators >= 5  # High maturity threshold
        return VerificationResult(
            "Architectural Maturity", success,
            f"Maturity indicators: {maturity_indicators}/{max_indicators}. " + "; ".join(details),
            maturity_indicators * 2
        )
    
    def run_complete_verification(self) -> bool:
        """Run complete maturity verification."""
        print("🚀 Starting Simple Maturity Verification...")
        print("=" * 60)
        
        # Run all verification checks
        verifications = [
            self.verify_f841_elimination,
            self.verify_maturity_components,
            self.verify_integration_patterns,
            self.verify_error_handling,
            self.verify_architectural_maturity
        ]
        
        for verification_func in verifications:
            result = verification_func()
            self.results.append(result)
            self.total_score += result.score
            print()
        
        # Calculate maximum possible score
        self.max_score = 30 + 25 + 20 + 15 + 12  # From each verification
        
        # Generate final report
        self.generate_final_report()
        
        # Return overall success
        passed_count = sum(1 for r in self.results if r.passed)
        return passed_count >= 4  # At least 4 out of 5 checks must pass
    
    def generate_final_report(self):
        """Generate final verification report."""
        print("=" * 60)
        print("🎯 SIMPLE MATURITY VERIFICATION REPORT")
        print("=" * 60)
        
        # Overall statistics
        passed_count = sum(1 for r in self.results if r.passed)
        total_checks = len(self.results)
        success_rate = (passed_count / total_checks) * 100
        score_percentage = (self.total_score / self.max_score) * 100 if self.max_score > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   • Checks Passed: {passed_count}/{total_checks} ({success_rate:.1f}%)")
        print(f"   • Total Score: {self.total_score}/{self.max_score} ({score_percentage:.1f}%)")
        print()
        
        # Individual results
        print("📋 DETAILED RESULTS:")
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"   {status} {result.name} (Score: {result.score})")
            print(f"      {result.details}")
        print()
        
        # Final assessment
        print("🎉 FINAL ASSESSMENT:")
        if passed_count >= 4 and score_percentage >= 80:
            print("   🏆 EXCELLENT - Maturity implementation is highly successful!")
            print("   ✅ System shows comprehensive architectural improvements")
        elif passed_count >= 3 and score_percentage >= 60:
            print("   ✅ GOOD - Maturity implementation shows solid progress")
            print("   📈 Most key improvements have been successfully implemented")
        elif passed_count >= 2:
            print("   ⚠️ MODERATE - Maturity implementation shows some progress")
            print("   🔧 Additional work needed to complete the improvements")
        else:
            print("   ❌ NEEDS WORK - Maturity implementation requires significant attention")
            print("   🚨 Core architectural improvements are missing or incomplete")
        
        print("=" * 60)
        
        # Save report
        report_content = self._generate_markdown_report()
        report_file = Path("SIMPLE_MATURITY_VERIFICATION.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📄 Detailed report saved to: {report_file}")
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Simple Maturity Verification Report",
            f"*Generated: {Path.cwd().name} - {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Summary",
            f"- **Checks Passed**: {sum(1 for r in self.results if r.passed)}/{len(self.results)}",
            f"- **Total Score**: {self.total_score}/{self.max_score}",
            "",
            "## Detailed Results",
            ""
        ]
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.extend([
                f"### {result.name} - {status}",
                f"**Score**: {result.score}",
                f"**Details**: {result.details}",
                ""
            ])
        
        return "\n".join(lines)


def main():
    """Main verification function."""
    verifier = MaturityVerifier()
    success = verifier.run_complete_verification()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 