# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
except ImportError:
    try:
        from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
    except ImportError:
        def safe_print(message): print(message)
        def info(message): print(f"[INFO] {message}")
        def warn(message): print(f"[WARN] {message}")
        def error(message): print(f"[ERROR] {message}")
        def success(message): print(f"[SUCCESS] {message}")
        def debug(message): print(f"[DEBUG] {message}")
#!/usr/bin/env python3
"""
Demo Connectivity Audit - Schwabot Demo Suite Analysis
=====================================================

This module provides comprehensive analysis of all demo/test/simulator modules
and their connectivity to the real Schwabot codebase. It identifies:

1. Which demo modules exist and their current state
2. What example/placeholder code needs to be replaced
3. How to connect demos to the real trading system
4. What refactoring is needed for full implementation
5. How to enable seamless demo-to-live transitions

This audit ensures all demo functionality is mathematically viable and
fully integrated with the real Schwabot architecture.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import importlib
import inspect

logger = logging.getLogger(__name__)


@dataclass
class DemoModuleInfo:
    """Information about a demo module."""
    module_name: str
    file_path: str
    module_type: str  # "demo", "test", "simulator", "backtest"
    has_real_integration: bool
    uses_example_code: bool
    connects_to_live_system: bool
    mathematical_viability: str  # "full", "partial", "none"
    integration_points: List[str]
    refactoring_needed: List[str]
    priority: int  # 1=high, 2=medium, 3=low


@dataclass
class ConnectivityAnalysis:
    """Analysis of demo suite connectivity."""
    total_modules: int
    modules_with_real_integration: int
    modules_with_example_code: int
    modules_connecting_to_live: int
    mathematically_viable_modules: int
    high_priority_refactors: List[str]
    medium_priority_refactors: List[str]
    low_priority_refactors: List[str]
    integration_gaps: List[str]
    recommendations: List[str]


class DemoConnectivityAudit:
    """
    Comprehensive audit system for demo suite connectivity.

    Analyzes all demo/test/simulator modules to ensure they:
    - Use real mathematical logic (DLT, unified mathematics, etc.)
    - Connect to the actual Schwabot pipeline
    - Can transition seamlessly to live trading
    - Don't contain example or placeholder code
    """

    def __init__(self, core_directory: str = "core"):
        self.core_directory = Path(core_directory)
        self.demo_modules: List[DemoModuleInfo] = []
        self.analysis: Optional[ConnectivityAnalysis] = None

        # Real Schwabot integration points
        self.real_integration_points = {
            "ferris_rde_core": "16-bit BTC price mapping",
            "tick_hash_processor": "Real tick hash generation",
            "unified_mathematics_config": "Unified mathematical operations",
            "integrated_alif_aleph_system": "ALEPH/ALIF dualistic system",
            "mathlib_v4": "DLT waveform integration",
            "real_trading_integration": "Real trading system",
            "multi_bit_btc_processor": "Multi-bit BTC processing",
            "dlt_waveform_engine": "DLT waveform engine",
            "ccxt_execution_manager": "Real exchange execution",
            "ccxt_profit_vectorizer": "Real profit vectorization"
        }

        # Example code patterns to detect
        self.example_patterns = [
            "example", "demo", "test", "dummy", "mock", "fake", "placeholder",
            "TODO", "FIXME", "TEMP", "static", "hardcoded", "sample"
        ]

        logger.info("Demo Connectivity Audit initialized")

    def run_full_audit(self) -> ConnectivityAnalysis:
        """Run comprehensive audit of all demo modules."""
        logger.info("🔍 Starting comprehensive demo connectivity audit")

        # Discover all demo modules
        self._discover_demo_modules()

        # Analyze each module
        self._analyze_all_modules()

        # Generate connectivity analysis
        self.analysis = self._generate_connectivity_analysis()

        # Generate recommendations
        self._generate_recommendations()

        logger.info(f"✅ Audit completed. Found {len(self.demo_modules)} demo modules")

        return self.analysis

    def _discover_demo_modules(self) -> None:
        """Discover all demo/test/simulator modules in the codebase."""
        demo_modules = []

        # Search for demo-related files
        for file_path in self.core_directory.rglob("*.py"):
            file_name = file_path.name.lower()

            # Check if file is demo-related
            if any(keyword in file_name for keyword in ["demo", "test", "simulator", "backtest", "backtrace"]):
                module_type = self._determine_module_type(file_name)
                demo_modules.append((file_path, module_type))

        # Also check for demo-related content in files
        for file_path in self.core_directory.rglob("*.py"):
            if file_path not in [f for f, _ in demo_modules]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in ["demo", "test", "simulator", "backtest"]):
                            module_type = self._determine_module_type(file_path.name.lower())
                            demo_modules.append((file_path, module_type))
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")

        # Create module info objects
        for file_path, module_type in demo_modules:
            module_info = DemoModuleInfo(
                module_name=file_path.stem,
                file_path=str(file_path),
                module_type=module_type,
                has_real_integration=False,
                uses_example_code=False,
                connects_to_live_system=False,
                mathematical_viability="none",
                integration_points=[],
                refactoring_needed=[],
                priority=3
            )
            self.demo_modules.append(module_info)

    def _determine_module_type(self, file_name: str) -> str:
        """Determine the type of demo module."""
        if "demo" in file_name:
            return "demo"
        elif "test" in file_name:
            return "test"
        elif "simulator" in file_name:
            return "simulator"
        elif "backtest" in file_name:
            return "backtest"
        elif "backtrace" in file_name:
            return "backtrace"
        else:
            return "unknown"

    def _analyze_all_modules(self) -> None:
        """Analyze all discovered demo modules."""
        for module_info in self.demo_modules:
            self._analyze_single_module(module_info)

    def _analyze_single_module(self, module_info: DemoModuleInfo) -> None:
        """Analyze a single demo module."""
        try:
            with open(module_info.file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for real integration points
            integration_points = []
            for point, description in self.real_integration_points.items():
                if point in content:
                    integration_points.append(f"{point}: {description}")

            module_info.integration_points = integration_points
            module_info.has_real_integration = len(integration_points) > 0

            # Check for example code
            example_code_found = []
            for pattern in self.example_patterns:
                if pattern in content.lower():
                    example_code_found.append(pattern)

            module_info.uses_example_code = len(example_code_found) > 0

            # Check for live system connectivity
            live_connectivity_patterns = [
                "real_trading_integration",
                "ferris_rde_core",
                "tick_hash_processor",
                "ccxt_execution_manager",
                "live_mode",
                "live_trading"
            ]

            live_connections = []
            for pattern in live_connectivity_patterns:
                if pattern in content:
                    live_connections.append(pattern)

            module_info.connects_to_live_system = len(live_connections) > 0

            # Assess mathematical viability
            math_patterns = {
                "full": ["dlt_waveform", "unified_mathematics", "mathlib_v4", "observer_aware"],
                "partial": ["numpy", "scipy", "mathematical", "calculation"],
                "none": []
            }

            math_score = 0
            for viability, patterns in math_patterns.items():
                for pattern in patterns:
                    if pattern in content.lower():
                        math_score += 1

            if math_score >= 3:
                module_info.mathematical_viability = "full"
            elif math_score >= 1:
                module_info.mathematical_viability = "partial"
            else:
                module_info.mathematical_viability = "none"

            # Determine refactoring needs
            refactoring_needs = []
            if module_info.uses_example_code:
                refactoring_needs.append("Replace example/placeholder code")
            if not module_info.has_real_integration:
                refactoring_needs.append("Add real integration points")
            if not module_info.connects_to_live_system:
                refactoring_needs.append("Enable live system connectivity")
            if module_info.mathematical_viability != "full":
                refactoring_needs.append("Improve mathematical viability")

            module_info.refactoring_needed = refactoring_needs

            # Determine priority
            if len(refactoring_needs) >= 3:
                module_info.priority = 1  # High priority
            elif len(refactoring_needs) >= 1:
                module_info.priority = 2  # Medium priority
            else:
                module_info.priority = 3  # Low priority

        except Exception as e:
            logger.error(f"Error analyzing {module_info.module_name}: {e}")
            module_info.refactoring_needed.append(f"Error during analysis: {e}")
            module_info.priority = 1

    def _generate_connectivity_analysis(self) -> ConnectivityAnalysis:
        """Generate comprehensive connectivity analysis."""
        total_modules = len(self.demo_modules)
        modules_with_real_integration = sum(1 for m in self.demo_modules if m.has_real_integration)
        modules_with_example_code = sum(1 for m in self.demo_modules if m.uses_example_code)
        modules_connecting_to_live = sum(1 for m in self.demo_modules if m.connects_to_live_system)
        mathematically_viable_modules = sum(1 for m in self.demo_modules if m.mathematical_viability == "full")

        # Categorize refactoring needs by priority
        high_priority_refactors = [m.module_name for m in self.demo_modules if m.priority == 1]
        medium_priority_refactors = [m.module_name for m in self.demo_modules if m.priority == 2]
        low_priority_refactors = [m.module_name for m in self.demo_modules if m.priority == 3]

        # Identify integration gaps
        integration_gaps = []
        for module in self.demo_modules:
            if not module.has_real_integration:
                integration_gaps.append(f"{module.module_name}: No real integration points")
            if not module.connects_to_live_system:
                integration_gaps.append(f"{module.module_name}: No live system connectivity")

        return ConnectivityAnalysis(
            total_modules=total_modules,
            modules_with_real_integration=modules_with_real_integration,
            modules_with_example_code=modules_with_example_code,
            modules_connecting_to_live=modules_connecting_to_live,
            mathematically_viable_modules=mathematically_viable_modules,
            high_priority_refactors=high_priority_refactors,
            medium_priority_refactors=medium_priority_refactors,
            low_priority_refactors=low_priority_refactors,
            integration_gaps=integration_gaps,
            recommendations=[]
        )

    def _generate_recommendations(self) -> None:
        """Generate specific recommendations for improvement."""
        recommendations = []

        # High-level recommendations
        if self.analysis.modules_with_example_code > 0:
            recommendations.append("Replace all example/placeholder code with real implementations")

        if self.analysis.modules_connecting_to_live < self.analysis.total_modules:
            recommendations.append("Enable live system connectivity for all demo modules")

        if self.analysis.mathematically_viable_modules < self.analysis.total_modules:
            recommendations.append("Improve mathematical viability across all modules")

        # Specific recommendations for high-priority modules
        for module_name in self.analysis.high_priority_refactors:
            module = next(m for m in self.demo_modules if m.module_name == module_name)
            recommendations.append(f"High priority: Refactor {module_name} - {', '.join(module.refactoring_needed)}")

        # Integration recommendations
        recommendations.append("Ensure all demos use real BTC price hashing and 16-bit mapping")
        recommendations.append("Connect all demos to the ALEPH/ALIF dualistic system")
        recommendations.append("Implement DLT waveform integration in all demo modules")
        recommendations.append("Enable seamless demo-to-live transitions")

        self.analysis.recommendations = recommendations

    def generate_audit_report(self, output_file: str = "demo_connectivity_audit_report.json") -> str:
        """Generate comprehensive audit report."""
        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_modules": self.analysis.total_modules,
                "modules_with_real_integration": self.analysis.modules_with_real_integration,
                "modules_with_example_code": self.analysis.modules_with_example_code,
                "modules_connecting_to_live": self.analysis.modules_connecting_to_live,
                "mathematically_viable_modules": self.analysis.mathematically_viable_modules
            },
            "module_details": [
                {
                    "module_name": m.module_name,
                    "file_path": m.file_path,
                    "module_type": m.module_type,
                    "has_real_integration": m.has_real_integration,
                    "uses_example_code": m.uses_example_code,
                    "connects_to_live_system": m.connects_to_live_system,
                    "mathematical_viability": m.mathematical_viability,
                    "integration_points": m.integration_points,
                    "refactoring_needed": m.refactoring_needed,
                    "priority": m.priority
                }
                for m in self.demo_modules
            ],
            "refactoring_priorities": {
                "high_priority": self.analysis.high_priority_refactors,
                "medium_priority": self.analysis.medium_priority_refactors,
                "low_priority": self.analysis.low_priority_refactors
            },
            "integration_gaps": self.analysis.integration_gaps,
            "recommendations": self.analysis.recommendations
        }

        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return output_file

    def print_audit_summary(self) -> None:
        """Print audit summary to console."""
        if not self.analysis:
            safe_print("❌ No audit analysis available. Run run_full_audit() first.")
            return

        safe_print("\n" + "="*60)
        safe_print("🔍 SCHWABOT DEMO CONNECTIVITY AUDIT SUMMARY")
        safe_print("="*60)

        safe_print(f"\n📊 OVERALL STATISTICS:")
        safe_print(f"   Total Demo Modules: {self.analysis.total_modules}")
        safe_print(f"   With Real Integration: {self.analysis.modules_with_real_integration}")
        safe_print(f"   With Example Code: {self.analysis.modules_with_example_code}")
        safe_print(f"   Connecting to Live: {self.analysis.modules_connecting_to_live}")
        safe_print(f"   Mathematically Viable: {self.analysis.mathematically_viable_modules}")

        safe_print(f"\n🚨 HIGH PRIORITY REFACTORS ({len(self.analysis.high_priority_refactors)}):")
        for module in self.analysis.high_priority_refactors:
            safe_print(f"   • {module}")

        safe_print(f"\n⚠️  MEDIUM PRIORITY REFACTORS ({len(self.analysis.medium_priority_refactors)}):")
        for module in self.analysis.medium_priority_refactors:
            safe_print(f"   • {module}")

        safe_print(f"\n✅ LOW PRIORITY REFACTORS ({len(self.analysis.low_priority_refactors)}):")
        for module in self.analysis.low_priority_refactors:
            safe_print(f"   • {module}")

        safe_print(f"\n🔗 INTEGRATION GAPS ({len(self.analysis.integration_gaps)}):")
        for gap in self.analysis.integration_gaps[:5]:  # Show first 5
            safe_print(f"   • {gap}")
        if len(self.analysis.integration_gaps) > 5:
            safe_print(f"   ... and {len(self.analysis.integration_gaps) - 5} more")

        safe_print(f"\n💡 KEY RECOMMENDATIONS:")
        for rec in self.analysis.recommendations[:5]:  # Show first 5
            safe_print(f"   • {rec}")
        if len(self.analysis.recommendations) > 5:
            safe_print(f"   ... and {len(self.analysis.recommendations) - 5} more")

        safe_print("\n" + "="*60)


def get_demo_connectivity_audit() -> DemoConnectivityAudit:
    """Get singleton instance of demo connectivity audit."""
    if not hasattr(get_demo_connectivity_audit, '_instance'):
        get_demo_connectivity_audit._instance = DemoConnectivityAudit()
    return get_demo_connectivity_audit._instance


def main() -> None:
    """Main function for running demo connectivity audit."""
    logging.basicConfig(level=logging.INFO)

    safe_print("🔍 Starting Schwabot Demo Connectivity Audit")
    safe_print("="*50)

    # Run audit
    audit = get_demo_connectivity_audit()
    analysis = audit.run_full_audit()

    # Print summary
    audit.print_audit_summary()

    # Generate report
    report_file = audit.generate_audit_report()
    safe_print(f"\n📄 Detailed report saved to: {report_file}")

    # Provide next steps
    safe_print(f"\n🎯 NEXT STEPS:")
    safe_print(f"   1. Review high priority refactors ({len(analysis.high_priority_refactors)} modules)")
    safe_print(f"   2. Address integration gaps ({len(analysis.integration_gaps)} issues)")
    safe_print(f"   3. Implement recommendations ({len(analysis.recommendations)} items)")
    safe_print(f"   4. Test demo-to-live transitions")
    safe_print(f"   5. Validate mathematical viability")


if __name__ == "__main__":
    main()
