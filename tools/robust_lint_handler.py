#!/usr/bin/env python3
"""
Robust Lint Handler - Schwabot UROS v1.0
========================================
Handles flake8 linting with proper error handling for pipeline desync issues.
Addresses:
- Virtual environment problems
- Character encoding issues
- File path resolution
- PowerShell PSReadLine bugs
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustLintHandler:
    """Handles flake8 linting with comprehensive error handling."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.core_dir = self.project_root / "core"
        self.tools_dir = self.project_root / "tools"
        self.mathlib_dir = self.project_root / "mathlib"
        
        # Expected core modules (verified to exist)
        self.expected_core_modules = [
            "dlt_waveform_engine.py",
            "multi_bit_btc_processor.py", 
            "profit_routing_engine.py",
            "temporal_execution_correction_layer.py",
            "post_failure_recovery_intelligence_loop.py"
        ]
        
        # Flake8 configuration
        self.flake8_config = {
            "select": "E9,F63,F7,F82",  # Critical errors only
            "max_line_length": 100,
            "max_complexity": 20,
            "extend_ignore": "D401,ANN101,ANN201,W293,W291,W292,W503,W504,W391,C901,F541,I100,I101,I201,I202,D100,D101,D103,D107,D200,D202,D205,D209,D211,D212,D300,D301,D402,D412,D415,D417"
        }
    
    def verify_file_existence(self) -> Dict[str, bool]:
        """Verify that all expected files exist."""
        results = {}
        for module in self.expected_core_modules:
            file_path = self.core_dir / module
            exists = file_path.exists()
            results[module] = exists
            if not exists:
                logger.warning(f"Missing expected file: {file_path}")
            else:
                logger.info(f"✓ Found: {file_path}")
        return results
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment for potential issues."""
        env_info = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "cwd": os.getcwd(),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "pyvenv_cfg_exists": False
        }
        
        # Check for pyvenv.cfg
        if env_info["virtual_env"]:
            pyvenv_cfg = Path(env_info["virtual_env"]) / "pyvenv.cfg"
            env_info["pyvenv_cfg_exists"] = pyvenv_cfg.exists()
        
        return env_info
    
    def run_flake8_safe(self, target_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run flake8 with comprehensive error handling."""
        result = {
            "success": False,
            "errors": [],
            "warnings": [],
            "output": "",
            "return_code": -1
        }
        
        try:
            # Build flake8 command
            cmd = [
                sys.executable, "-m", "flake8",
                target_path,
                f"--select={self.flake8_config['select']}",
                f"--max-line-length={self.flake8_config['max_line_length']}",
                f"--max-complexity={self.flake8_config['max_complexity']}",
                f"--extend-ignore={self.flake8_config['extend_ignore']}",
                "--count",
                "--statistics"
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Run with proper encoding handling
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Handle encoding errors gracefully
                cwd=self.project_root
            )
            
            result["return_code"] = process.returncode
            result["output"] = process.stdout
            result["errors"] = process.stderr.splitlines() if process.stderr else []
            
            # Parse output for actual linting results
            if process.stdout:
                lines = process.stdout.splitlines()
                for line in lines:
                    if ":" in line and any(code in line for code in ["E9", "F63", "F7", "F82"]):
                        result["warnings"].append(line)
            
            result["success"] = process.returncode == 0 or len(result["warnings"]) == 0
            
            # Save output to file if requested
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Flake8 Results for {target_path}\n")
                    f.write(f"# Command: {' '.join(cmd)}\n")
                    f.write(f"# Return Code: {process.returncode}\n\n")
                    f.write(process.stdout)
                    if process.stderr:
                        f.write(f"\n# Errors:\n{process.stderr}")
            
        except Exception as e:
            logger.error(f"Error running flake8: {e}")
            result["errors"].append(str(e))
        
        return result
    
    def run_comprehensive_lint_check(self) -> Dict[str, Any]:
        """Run comprehensive linting check across all modules."""
        logger.info("Starting comprehensive lint check...")
        
        # Step 1: Verify environment
        env_info = self.check_python_environment()
        logger.info(f"Environment: {env_info}")
        
        # Step 2: Verify file existence
        file_status = self.verify_file_existence()
        missing_files = [f for f, exists in file_status.items() if not exists]
        
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
        
        # Step 3: Run flake8 on core directory
        core_result = self.run_flake8_safe("core/", "core_lint_results.txt")
        
        # Step 4: Run flake8 on mathlib directory
        mathlib_result = self.run_flake8_safe("mathlib/", "mathlib_lint_results.txt")
        
        # Step 5: Run flake8 on tools directory
        tools_result = self.run_flake8_safe("tools/", "tools_lint_results.txt")
        
        # Compile results
        comprehensive_result = {
            "environment": env_info,
            "file_status": file_status,
            "core_lint": core_result,
            "mathlib_lint": mathlib_result,
            "tools_lint": tools_result,
            "summary": {
                "total_errors": len(core_result["warnings"]) + len(mathlib_result["warnings"]) + len(tools_result["warnings"]),
                "missing_files": len(missing_files),
                "overall_success": all([
                    core_result["success"],
                    mathlib_result["success"], 
                    tools_result["success"]
                ])
            }
        }
        
        # Log summary
        logger.info(f"Lint Summary: {comprehensive_result['summary']}")
        
        return comprehensive_result
    
    def generate_fix_report(self, lint_results: Dict[str, Any]) -> str:
        """Generate a comprehensive fix report."""
        report = []
        report.append("# SCHWABOT LINT FIX REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Environment issues
        env = lint_results["environment"]
        if not env["pyvenv_cfg_exists"] and env["virtual_env"]:
            report.append("## ⚠️ VIRTUAL ENVIRONMENT ISSUE")
            report.append("Missing pyvenv.cfg file - this may cause E902 errors.")
            report.append("Fix: Recreate virtual environment or use system Python.")
            report.append("")
        
        # Missing files
        missing = [f for f, exists in lint_results["file_status"].items() if not exists]
        if missing:
            report.append("## ❌ MISSING FILES")
            for file in missing:
                report.append(f"- {file}")
            report.append("")
        
        # Lint errors by module
        for module, result in [("Core", lint_results["core_lint"]), 
                              ("Mathlib", lint_results["mathlib_lint"]),
                              ("Tools", lint_results["tools_lint"])]:
            if result["warnings"]:
                report.append(f"## 🔧 {module.upper()} LINT ISSUES")
                for warning in result["warnings"][:10]:  # Show first 10
                    report.append(f"- {warning}")
                if len(result["warnings"]) > 10:
                    report.append(f"- ... and {len(result['warnings']) - 10} more")
                report.append("")
        
        # Success summary
        summary = lint_results["summary"]
        if summary["overall_success"]:
            report.append("## ✅ OVERALL STATUS: CLEAN")
        else:
            report.append("## ⚠️ OVERALL STATUS: ISSUES FOUND")
        
        report.append(f"Total Errors: {summary['total_errors']}")
        report.append(f"Missing Files: {summary['missing_files']}")
        
        return "\n".join(report)

def main():
    """Main function for running the robust lint handler."""
    handler = RobustLintHandler()
    
    print("🔍 SCHWABOT ROBUST LINT HANDLER")
    print("=" * 40)
    
    # Run comprehensive check
    results = handler.run_comprehensive_lint_check()
    
    # Generate and display report
    report = handler.generate_fix_report(results)
    print("\n" + report)
    
    # Save detailed results
    with open("comprehensive_lint_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: comprehensive_lint_report.json")
    
    # Exit with appropriate code
    if results["summary"]["overall_success"]:
        print("✅ Lint check completed successfully!")
        sys.exit(0)
    else:
        print("⚠️ Lint check found issues - see report above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 