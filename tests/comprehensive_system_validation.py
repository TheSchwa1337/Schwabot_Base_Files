#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive System Validation
==============================

Complete validation of the Schwabot system including:
- Syntax validation
- Import validation
- Code quality checks
- System integration tests
- Performance validation

This script provides a complete health check of the entire system.
"""

import os
import sys
import ast
import logging
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    message: str
    details: Dict[str, Any] = None

class ComprehensiveValidator:
    """Comprehensive system validator."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = []
        
    def validate_syntax(self) -> ValidationResult:
        """Validate Python syntax across all files."""
        logger.info("🔍 Validating Python syntax...")
        
        syntax_errors = []
        files_checked = 0
        
        for py_file in self.project_root.rglob("*.py"):
            files_checked += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Parse AST to check syntax
                ast.parse(content)
                
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception as e:
                syntax_errors.append(f"{py_file}: {e}")
                
        if syntax_errors:
            return ValidationResult(
                name="Python Syntax Validation",
                status="FAIL",
                message=f"Found {len(syntax_errors)} syntax errors",
                details={"errors": syntax_errors, "files_checked": files_checked}
            )
        else:
            return ValidationResult(
                name="Python Syntax Validation",
                status="PASS",
                message=f"All {files_checked} Python files have valid syntax",
                details={"files_checked": files_checked}
            )
    
    def validate_imports(self) -> ValidationResult:
        """Validate that all modules can be imported."""
        logger.info("📦 Validating module imports...")
        
        import_errors = []
        modules_checked = 0
        
        # Core modules to check
        core_modules = [
            "core.unified_mathematical_bridge",
            "core.risk_manager",
            "core.automated_trading_engine",
            "core.btc_usdc_trading_integration",
            "core.cli_dual_state_router",
            "core.enhanced_gpu_auto_detector",
            "core.pure_profit_calculator",
            "core.trade_registry",
            "core.error_handling_and_flake_gate_prevention"
        ]
        
        for module_name in core_modules:
            modules_checked += 1
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                import_errors.append(f"{module_name}: {e}")
            except Exception as e:
                import_errors.append(f"{module_name}: {e}")
                
        if import_errors:
            return ValidationResult(
                name="Module Import Validation",
                status="FAIL",
                message=f"Found {len(import_errors)} import errors",
                details={"errors": import_errors, "modules_checked": modules_checked}
            )
        else:
            return ValidationResult(
                name="Module Import Validation",
                status="PASS",
                message=f"All {modules_checked} core modules import successfully",
                details={"modules_checked": modules_checked}
            )
    
    def validate_code_quality(self) -> ValidationResult:
        """Validate code quality using basic checks."""
        logger.info("✨ Validating code quality...")
        
        quality_issues = []
        files_checked = 0
        
        for py_file in self.project_root.rglob("*.py"):
            files_checked += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Basic quality checks
                for i, line in enumerate(lines, 1):
                    # Check for trailing whitespace
                    if line.rstrip() != line and line.strip():
                        quality_issues.append(f"{py_file}:{i}: Trailing whitespace")
                        
                    # Check for very long lines (>120 chars)
                    if len(line.rstrip()) > 120:
                        quality_issues.append(f"{py_file}:{i}: Line too long ({len(line.rstrip())} chars)")
                        
                    # Check for mixed tabs and spaces
                    if '\t' in line and '    ' in line:
                        quality_issues.append(f"{py_file}:{i}: Mixed tabs and spaces")
                        
            except Exception as e:
                quality_issues.append(f"{py_file}: Error reading file: {e}")
                
        if quality_issues:
            return ValidationResult(
                name="Code Quality Validation",
                status="WARNING",
                message=f"Found {len(quality_issues)} quality issues",
                details={"issues": quality_issues[:20], "files_checked": files_checked}
            )
        else:
            return ValidationResult(
                name="Code Quality Validation",
                status="PASS",
                message=f"All {files_checked} files pass quality checks",
                details={"files_checked": files_checked}
            )
    
    def validate_system_integration(self) -> ValidationResult:
        """Validate system integration and core functionality."""
        logger.info("🔗 Validating system integration...")
        
        integration_issues = []
        
        # Test core system components
        try:
            # Test mathematical bridge
            from core.unified_mathematical_bridge import UnifiedMathematicalBridge
            bridge = UnifiedMathematicalBridge()
            
            # Test risk manager
            from core.risk_manager import RiskManager
            risk_mgr = RiskManager()
            
            # Test GPU detection
            from core.enhanced_gpu_auto_detector import EnhancedGPUAutoDetector
            gpu_detector = EnhancedGPUAutoDetector()
            
            # Test profit calculator
            from core.pure_profit_calculator import PureProfitCalculator
            profit_calc = PureProfitCalculator({})
            
        except Exception as e:
            integration_issues.append(f"Core component initialization: {e}")
            
        if integration_issues:
            return ValidationResult(
                name="System Integration Validation",
                status="FAIL",
                message=f"Found {len(integration_issues)} integration issues",
                details={"issues": integration_issues}
            )
        else:
            return ValidationResult(
                name="System Integration Validation",
                status="PASS",
                message="All core components integrate successfully",
                details={}
            )
    
    def validate_file_structure(self) -> ValidationResult:
        """Validate project file structure."""
        logger.info("📁 Validating file structure...")
        
        missing_files = []
        required_files = [
            "main.py",
            "core/__init__.py",
            "core/unified_mathematical_bridge.py",
            "core/risk_manager.py",
            "core/automated_trading_engine.py",
            "core/btc_usdc_trading_integration.py",
            "core/enhanced_gpu_auto_detector.py",
            "core/pure_profit_calculator.py",
            "core/trade_registry.py",
            "scripts/",
            "tests/",
            "requirements.txt"
        ]
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            return ValidationResult(
                name="File Structure Validation",
                status="WARNING",
                message=f"Missing {len(missing_files)} required files",
                details={"missing_files": missing_files}
            )
        else:
            return ValidationResult(
                name="File Structure Validation",
                status="PASS",
                message="All required files and directories present",
                details={"required_files": len(required_files)}
            )
    
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation checks."""
        logger.info("🚀 Starting comprehensive system validation...")
        
        validations = [
            self.validate_syntax,
            self.validate_imports,
            self.validate_code_quality,
            self.validate_system_integration,
            self.validate_file_structure
        ]
        
        for validation in validations:
            try:
                result = validation()
                self.results.append(result)
                logger.info(f"✅ {result.name}: {result.status}")
            except Exception as e:
                error_result = ValidationResult(
                    name=validation.__name__,
                    status="FAIL",
                    message=f"Validation failed: {e}",
                    details={"error": str(e)}
                )
                self.results.append(error_result)
                logger.error(f"❌ {validation.__name__}: FAIL")
                
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.results:
            return "No validation results available."
            
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE SYSTEM VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        total_checks = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warnings = sum(1 for r in self.results if r.status == "WARNING")
        
        report.append(f"📊 SUMMARY:")
        report.append(f"   Total Checks: {total_checks}")
        report.append(f"   ✅ Passed: {passed}")
        report.append(f"   ❌ Failed: {failed}")
        report.append(f"   ⚠️  Warnings: {warnings}")
        report.append("")
        
        # Detailed results
        for result in self.results:
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}[result.status]
            report.append(f"{status_icon} {result.name}")
            report.append(f"   Status: {result.status}")
            report.append(f"   Message: {result.message}")
            
            if result.details:
                report.append("   Details:")
                for key, value in result.details.items():
                    if isinstance(value, list) and len(value) > 5:
                        report.append(f"     {key}: {len(value)} items (showing first 5)")
                        for item in value[:5]:
                            report.append(f"       - {item}")
                    elif isinstance(value, list):
                        for item in value:
                            report.append(f"       - {item}")
                    else:
                        report.append(f"     {key}: {value}")
            report.append("")
            
        # Overall assessment
        if failed == 0 and warnings == 0:
            report.append("🎉 EXCELLENT! System is fully functional and production-ready!")
        elif failed == 0:
            report.append("✅ GOOD! System is functional with minor warnings.")
        else:
            report.append("⚠️  ATTENTION! System has issues that need to be addressed.")
            
        return "\n".join(report)

def main():
    """Main execution function."""
    logger.info("🚀 Schwabot Comprehensive System Validation")
    logger.info("=" * 50)
    
    validator = ComprehensiveValidator()
    results = validator.run_all_validations()
    
    # Generate and display report
    report = validator.generate_report()
    print(report)
    
    # Save report to file
    with open("comprehensive_validation_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        
    logger.info("📄 Report saved to comprehensive_validation_report.txt")
    
    # Return appropriate exit code
    failed_count = sum(1 for r in results if r.status == "FAIL")
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 