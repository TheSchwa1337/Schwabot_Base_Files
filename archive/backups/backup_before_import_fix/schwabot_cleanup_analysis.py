#!/usr/bin/env python3
"""
Schwabot Directory Cleanup Analysis
===================================

This script analyzes the schwabot/ directory to identify:
1. Files that should be DELETED (stubs, duplicates, non-functional)
2. Files that should be REFACTORED (clean up, fix imports, add docs)
3. Files that need FULL IMPLEMENTATION (math logic in comments)
4. Mathematical logic that differs from core/ and needs implementation

Usage:
    python schwabot_cleanup_analysis.py
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


class SchwabotAnalyzer:
    """Analyze schwabot directory for cleanup and refactoring."""
    
    def __init__(self):
        self.schwabot_dir = Path("schwabot")
        self.core_dir = Path("core")
        self.analysis_results = {
            "DELETE": [],
            "REFACTOR": [],
            "IMPLEMENT": [],
            "MATH_DIFFERENCES": []
        }
        
    def analyze_file(self, file_path: Path) -> Dict[str, any]:
        """Analyze a single file for cleanup decisions."""
        result = {
            "file": str(file_path),
            "size": file_path.stat().st_size,
            "lines": 0,
            "has_math_comments": False,
            "has_imports": False,
            "has_functions": False,
            "has_classes": False,
            "is_stub": False,
            "duplicates_core": False,
            "recommendation": "KEEP"
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                result["lines"] = len(lines)
                
                # Check for various patterns
                result["has_math_comments"] = bool(re.search(r'#.*math|""".*math|def.*math', content, re.IGNORECASE))
                result["has_imports"] = bool(re.search(r'^import|^from', content, re.MULTILINE))
                result["has_functions"] = bool(re.search(r'^def\s+\w+', content, re.MULTILINE))
                result["has_classes"] = bool(re.search(r'^class\s+\w+', content, re.MULTILINE))
                result["is_stub"] = len(content.strip()) < 100 or "TODO" in content or "FIXME" in content
                
                # Check for mathematical logic in comments
                math_patterns = [
                    r'#.*formula|#.*equation|#.*calculation',
                    r'"""[\s\S]*?math[\s\S]*?"""',
                    r'#.*algorithm|#.*tensor|#.*quantum',
                    r'#.*entropy|#.*profit|#.*optimization'
                ]
                
                for pattern in math_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        result["has_math_comments"] = True
                        break
                
        except Exception as e:
            result["error"] = str(e)
            
        return result
    
    def check_core_duplicates(self, file_path: Path) -> List[str]:
        """Check if functionality exists in core/ directory."""
        duplicates = []
        file_stem = file_path.stem
        
        # Common patterns that might indicate duplication
        core_patterns = [
            f"{file_stem}.py",
            f"{file_stem}_core.py",
            f"unified_{file_stem}.py",
            f"{file_stem}_engine.py",
            f"{file_stem}_system.py"
        ]
        
        for pattern in core_patterns:
            core_file = self.core_dir / pattern
            if core_file.exists():
                duplicates.append(str(core_file))
                
        return duplicates
    
    def analyze_mathematical_differences(self, file_path: Path) -> List[str]:
        """Extract mathematical logic that differs from core/."""
        differences = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for mathematical implementations that might be unique
            math_functions = re.findall(r'def\s+(\w+).*?:\s*""".*?math.*?"""', content, re.DOTALL | re.IGNORECASE)
            
            for func in math_functions:
                # Check if this function exists in core
                core_has_function = False
                for core_file in self.core_dir.rglob("*.py"):
                    try:
                        with open(core_file, 'r', encoding='utf-8') as cf:
                            core_content = cf.read()
                            if re.search(rf'def\s+{func}\s*\(', core_content):
                                core_has_function = True
                                break
                    except:
                        continue
                
                if not core_has_function:
                    differences.append(f"Function '{func}' in {file_path.name}")
                    
        except Exception as e:
            differences.append(f"Error analyzing {file_path}: {e}")
            
        return differences
    
    def run_analysis(self):
        """Run complete analysis of schwabot directory."""
        print("🔍 Analyzing Schwabot directory for cleanup...")
        
        for file_path in self.schwabot_dir.rglob("*.py"):
            if file_path.name == "__init__.py":
                continue
                
            print(f"  Analyzing: {file_path}")
            
            # Analyze the file
            analysis = self.analyze_file(file_path)
            
            # Check for duplicates in core
            duplicates = self.check_core_duplicates(file_path)
            analysis["duplicates_core"] = bool(duplicates)
            analysis["core_duplicates"] = duplicates
            
            # Check for mathematical differences
            math_diffs = self.analyze_mathematical_differences(file_path)
            if math_diffs:
                self.analysis_results["MATH_DIFFERENCES"].extend(math_diffs)
            
            # Make recommendation
            if analysis["is_stub"] or analysis["size"] < 500:
                analysis["recommendation"] = "DELETE"
                self.analysis_results["DELETE"].append(analysis)
            elif duplicates or analysis["has_math_comments"]:
                analysis["recommendation"] = "IMPLEMENT"
                self.analysis_results["IMPLEMENT"].append(analysis)
            else:
                analysis["recommendation"] = "REFACTOR"
                self.analysis_results["REFACTOR"].append(analysis)
    
    def generate_report(self):
        """Generate comprehensive cleanup report."""
        print("\n" + "="*80)
        print("SCHWABOT CLEANUP ANALYSIS REPORT")
        print("="*80)
        
        # Files to DELETE
        print(f"\n🗑️  FILES TO DELETE ({len(self.analysis_results['DELETE'])}):")
        print("-" * 40)
        for item in self.analysis_results["DELETE"]:
            print(f"  ❌ {item['file']}")
            print(f"     Size: {item['size']} bytes, Lines: {item['lines']}")
            if item.get('error'):
                print(f"     Error: {item['error']}")
            print()
        
        # Files to IMPLEMENT
        print(f"\n🔧 FILES TO IMPLEMENT ({len(self.analysis_results['IMPLEMENT'])}):")
        print("-" * 40)
        for item in self.analysis_results["IMPLEMENT"]:
            print(f"  ⚡ {item['file']}")
            print(f"     Size: {item['size']} bytes, Lines: {item['lines']}")
            if item["duplicates_core"]:
                print(f"     Duplicates in core: {', '.join(item['core_duplicates'])}")
            if item["has_math_comments"]:
                print(f"     Contains mathematical logic in comments")
            print()
        
        # Files to REFACTOR
        print(f"\n🔄 FILES TO REFACTOR ({len(self.analysis_results['REFACTOR'])}):")
        print("-" * 40)
        for item in self.analysis_results["REFACTOR"]:
            print(f"  🔄 {item['file']}")
            print(f"     Size: {item['size']} bytes, Lines: {item['lines']}")
            print()
        
        # Mathematical differences
        if self.analysis_results["MATH_DIFFERENCES"]:
            print(f"\n🧮 MATHEMATICAL DIFFERENCES FROM CORE:")
            print("-" * 40)
            for diff in self.analysis_results["MATH_DIFFERENCES"]:
                print(f"  📊 {diff}")
            print()
        
        # Summary
        total_files = (len(self.analysis_results["DELETE"]) + 
                      len(self.analysis_results["IMPLEMENT"]) + 
                      len(self.analysis_results["REFACTOR"]))
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total files analyzed: {total_files}")
        print(f"  Files to delete: {len(self.analysis_results['DELETE'])}")
        print(f"  Files to implement: {len(self.analysis_results['IMPLEMENT'])}")
        print(f"  Files to refactor: {len(self.analysis_results['REFACTOR'])}")
        print(f"  Mathematical differences: {len(self.analysis_results['MATH_DIFFERENCES'])}")
        
        return self.analysis_results

def main():
    """Main analysis function."""
    analyzer = SchwabotAnalyzer()
    analyzer.run_analysis()
    results = analyzer.generate_report()
    
    # Save results to file
    with open("schwabot_cleanup_report.txt", "w") as f:
        f.write("SCHWABOT CLEANUP ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        for category, items in results.items():
            f.write(f"{category}:\n")
            f.write("-" * 20 + "\n")
            for item in items:
                if isinstance(item, dict):
                    f.write(f"  {item['file']}\n")
                else:
                    f.write(f"  {item}\n")
            f.write("\n")
    
    print(f"\n📄 Detailed report saved to: schwabot_cleanup_report.txt")

if __name__ == "__main__":
    main() 