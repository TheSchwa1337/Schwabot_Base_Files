#!/usr/bin/env python3
"""
Schwabot Architecture Analysis - Building Phase Assessment
========================================================

This script analyzes the current codebase to understand:
1. What components are actually implemented vs stubbed
2. What errors are blocking functionality vs cosmetic
3. What dependencies are missing vs broken
4. What needs to be removed vs implemented

Target Architecture:
- Flask API Server
- GPU/CPU Calculation Engine  
- Cross-platform CLIENTS
- CCXT Integration
- BTC Hashing & Strategy Engine
- External API Integration
"""

import os
import sys
import importlib
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import json


class ArchitectureAnalyzer:
    """Analyze the Schwabot codebase architecture and identify issues."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.analysis = {
            "core_components": {},
            "api_components": {},
            "math_components": {},
            "trading_components": {},
            "ui_components": {},
            "broken_files": [],
            "missing_dependencies": [],
            "stub_files": [],
            "working_files": []
        }

    def analyze_directory_structure(self) -> Dict[str, List[str]]:
        """Analyze the directory structure to understand the architecture."""

        structure = {
            "core": [],
            "api": [],
            "math": [],
            "trading": [],
            "ui": [],
            "utils": [],
            "tests": [],
            "config": [],
            "other": []
        }

        for item in self.root_path.iterdir():
            if item.is_file():
                if item.suffix == '.py':
                    if 'api' in item.name.lower() or 'gateway' in item.name.lower():
                        structure["api"].append(str(item))
                    elif 'math' in item.name.lower() or 'calc' in item.name.lower():
                        structure["math"].append(str(item))
                    elif 'trade' in item.name.lower() or 'btc' in item.name.lower():
                        structure["trading"].append(str(item))
                    elif 'ui' in item.name.lower() or 'gui' in item.name.lower():
                        structure["ui"].append(str(item))
                    elif 'test' in item.name.lower():
                        structure["tests"].append(str(item))
                    elif 'config' in item.name.lower() or 'settings' in item.name.lower():
                        structure["config"].append(str(item))
                    elif 'util' in item.name.lower() or 'helper' in item.name.lower():
                        structure["utils"].append(str(item))
                    else:
                        structure["core"].append(str(item))
            elif item.is_dir():
                if item.name in ['core', 'schwabot']:
                    structure["core"].extend([str(f) for f in item.rglob("*.py")])
                elif item.name in ['api', 'gateway']:
                    structure["api"].extend([str(f) for f in item.rglob("*.py")])
                elif item.name in ['math', 'mathlib', 'calculations']:
                    structure["math"].extend([str(f) for f in item.rglob("*.py")])
                elif item.name in ['trading', 'btc', 'exchange']:
                    structure["trading"].extend([str(f) for f in item.rglob("*.py")])
                elif item.name in ['ui', 'gui', 'frontend']:
                    structure["ui"].extend([str(f) for f in item.rglob("*.py")])
                elif item.name in ['tests', 'test']:
                    structure["tests"].extend([str(f) for f in item.rglob("*.py")])
                elif item.name in ['config', 'settings']:
                    structure["config"].extend([str(f) for f in item.rglob("*.py")])
                elif item.name in ['utils', 'helpers']:
                    structure["utils"].extend([str(f) for f in item.rglob("*.py")])
                else:
                    structure["other"].extend([str(f) for f in item.rglob("*.py")])

        return structure

    def analyze_file_content(self, file_path: str) -> Dict[str, any]:
        """Analyze a single Python file to determine its status."""

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the AST to understand the structure
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {
                    "status": "syntax_error",
                    "file": file_path,
                    "error": "Syntax error - cannot parse"
                }

            # Analyze the content
            analysis = {
                "file": file_path,
                "status": "unknown",
                "has_imports": False,
                "has_functions": False,
                "has_classes": False,
                "has_docstrings": False,
                "stub_indicators": [],
                "error_indicators": [],
                "functionality_indicators": []
            }

            # Check for stub indicators
            stub_patterns = [
                "TODO:", "FIXME:", "pass", "raise NotImplementedError",
                "return None", "return 0", "return []", "return {}",
                "def stub_", "class Stub", "placeholder", "dummy"
            ]

            for pattern in stub_patterns:
                if pattern in content:
                    analysis["stub_indicators"].append(pattern)

            # Check for error indicators
            error_patterns = [
                "ImportError", "ModuleNotFoundError", "NameError",
                "AttributeError", "TypeError", "SyntaxError"
            ]

            for pattern in error_patterns:
                if pattern in content:
                    analysis["error_indicators"].append(pattern)

            # Check for functionality indicators
            functionality_patterns = [
                "def ", "class ", "import ", "from ", "return ",
                "if __name__", "main()", "app.run()", "flask",
                "requests", "ccxt", "numpy", "pandas"
            ]

            for pattern in functionality_patterns:
                if pattern in content:
                    analysis["functionality_indicators"].append(pattern)

            # Determine status
            if analysis["stub_indicators"] and not analysis["functionality_indicators"]:
                analysis["status"] = "stub"
            elif analysis["error_indicators"] and not analysis["functionality_indicators"]:
                analysis["status"] = "broken"
            elif analysis["functionality_indicators"]:
                analysis["status"] = "working"
            else:
                analysis["status"] = "empty"

            return analysis

        except Exception as e:
            return {
                "status": "error",
                "file": file_path,
                "error": str(e)
            }

    def identify_critical_paths(self) -> Dict[str, List[str]]:
        """Identify the critical paths for the target architecture."""

        critical_paths = {
            "flask_api": [
                "app.py", "main.py", "server.py", "api/", "gateway/",
                "flask_app.py", "web_server.py"
            ],
            "gpu_cpu_engine": [
                "mathlib/", "calculations/", "engine/", "processor/",
                "gpu_", "cpu_", "compute_", "calculation_"
            ],
            "cross_platform": [
                "cli/", "client/", "desktop/", "gui/", "ui/",
                "windows_", "mac_", "linux_"
            ],
            "ccxt_integration": [
                "ccxt_", "exchange_", "trading_", "order_",
                "market_", "exchange/"
            ],
            "btc_hashing": [
                "btc_", "hash_", "strategy_", "crypto_",
                "bitcoin_", "blockchain_"
            ],
            "external_apis": [
                "api_", "external_", "whale_", "market_data_",
                "news_", "sentiment_"
            ]
        }

        return critical_paths

    def analyze_import_dependencies(self, file_path: str) -> List[str]:
        """Analyze import dependencies for a file."""

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            imports = []
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}")
            except SyntaxError:
                pass

            return imports

        except Exception:
            return []

    def generate_analysis_report(self) -> Dict[str, any]:
        """Generate a comprehensive analysis report."""

        print("🔍 Analyzing Schwabot Architecture...")

        # 1. Analyze directory structure
        structure = self.analyze_directory_structure()

        # 2. Analyze each file
        all_files = []
        for category, files in structure.items():
            all_files.extend(files)

        file_analyses = {}
        for file_path in all_files:
            if file_path.endswith('.py'):
                analysis = self.analyze_file_content(file_path)
                file_analyses[file_path] = analysis

        # 3. Categorize files
        working_files = []
        stub_files = []
        broken_files = []
        empty_files = []

        for file_path, analysis in file_analyses.items():
            if analysis["status"] == "working":
                working_files.append(file_path)
            elif analysis["status"] == "stub":
                stub_files.append(file_path)
            elif analysis["status"] == "broken":
                broken_files.append(file_path)
            elif analysis["status"] == "empty":
                empty_files.append(file_path)

        # 4. Identify critical missing components
        critical_paths = self.identify_critical_paths()
        missing_critical = {}

        for component, patterns in critical_paths.items():
            missing_critical[component] = []
            for pattern in patterns:
                found = False
                for file_path in all_files:
                    if pattern in file_path:
                        found = True
                        break
                if not found:
                    missing_critical[component].append(pattern)

        # 5. Generate report
        report = {
            "summary": {
                "total_files": len(all_files),
                "working_files": len(working_files),
                "stub_files": len(stub_files),
                "broken_files": len(broken_files),
                "empty_files": len(empty_files)
            },
            "structure": structure,
            "file_analyses": file_analyses,
            "categorized_files": {
                "working": working_files,
                "stubs": stub_files,
                "broken": broken_files,
                "empty": empty_files
            },
            "missing_critical_components": missing_critical,
            "recommendations": self.generate_recommendations(
                working_files, stub_files, broken_files, missing_critical
            )
        }

        return report

    def generate_recommendations(self, working_files: List[str],
                                 stub_files: List[str],
                                 broken_files: List[str],
                                 missing_critical: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Generate actionable recommendations."""

        recommendations = {
            "immediate_fixes": [],
            "stub_implementations": [],
            "missing_components": [],
            "cleanup_tasks": []
        }

        # Immediate fixes for broken files
        for file_path in broken_files:
            recommendations["immediate_fixes"].append(f"Fix syntax errors in {file_path}")

        # Stub implementations
        for file_path in stub_files:
            recommendations["stub_implementations"].append(f"Implement functionality in {file_path}")

        # Missing critical components
        for component, missing in missing_critical.items():
            if missing:
                recommendations["missing_components"].append(f"Create {component} components: {', '.join(missing)}")

        # Cleanup tasks
        if len(broken_files) > len(working_files):
            recommendations["cleanup_tasks"].append("Consider removing broken files that aren't critical")

        return recommendations

    def print_report(self, report: Dict[str, any]):
        """Print a formatted analysis report."""

        print("\n" + "="*80)
        print("🏗️  SCHWABOT ARCHITECTURE ANALYSIS REPORT")
        print("="*80)

        # Summary
        summary = report["summary"]
        print(f"\n📊 SUMMARY:")
        print(f"   Total Python files: {summary['total_files']}")
        print(f"   ✅ Working files: {summary['working_files']}")
        print(f"   🔧 Stub files: {summary['stub_files']}")
        print(f"   ❌ Broken files: {summary['broken_files']}")
        print(f"   📄 Empty files: {summary['empty_files']}")

        # Critical missing components
        print(f"\n🚨 CRITICAL MISSING COMPONENTS:")
        missing = report["missing_critical_components"]
        for component, patterns in missing.items():
            if patterns:
                print(f"   {component.upper()}: Missing {', '.join(patterns)}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recs = report["recommendations"]
        for category, items in recs.items():
            if items:
                print(f"   {category.upper()}:")
                for item in items:
                    print(f"     • {item}")

        # File breakdown by category
        print(f"\n📁 FILE BREAKDOWN:")
        structure = report["structure"]
        for category, files in structure.items():
            if files:
                print(f"   {category.upper()}: {len(files)} files")

        print("\n" + "="*80)


def main():
    """Run the architecture analysis."""

    analyzer = ArchitectureAnalyzer()
    report = analyzer.generate_analysis_report()
    analyzer.print_report(report)

    # Save detailed report
    with open("architecture_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: architecture_report.json")


if __name__ == "__main__":
    main()
