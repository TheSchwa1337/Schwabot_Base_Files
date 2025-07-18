#!/usr/bin/env python3
"""
Comprehensive File Analysis for Schwabot Core
Identifies files that need rewriting, deletion, or consolidation.
"""

import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


class FileAnalyzer:
    def __init__(self, core_dir: str = "core"):
        self.core_dir = Path(core_dir)
        self.results = {}
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for various metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic metrics
            lines = content.split('\n')
            line_count = len(lines)
            file_size = len(content.encode('utf-8'))
            
            # Count various patterns
            stub_indicators = {
                'pass_statements': len(re.findall(r'^\s*pass\s*$', content, re.MULTILINE)),
                'not_implemented': len(re.findall(r'raise NotImplementedError', content)),
                'todo_comments': len(re.findall(r'#\s*TODO', content, re.IGNORECASE)),
                'fixme_comments': len(re.findall(r'#\s*FIXME', content, re.IGNORECASE)),
                'stub_comments': len(re.findall(r'#\s*stub|#\s*placeholder', content, re.IGNORECASE)),
                'empty_functions': len(re.findall(r'def\s+\w+[^:]*:\s*\n\s*pass', content)),
            }
            
            # Complexity metrics
            complexity_indicators = {
                'nested_levels': self._count_max_nesting(content),
                'function_count': len(re.findall(r'^\s*def\s+', content, re.MULTILINE)),
                'class_count': len(re.findall(r'^\s*class\s+', content, re.MULTILINE)),
                'import_count': len(re.findall(r'^\s*(?:from|import)\s+', content, re.MULTILINE)),
            }
            
            # Math content analysis
            math_indicators = {
                'math_imports': len(re.findall(r'import\s+(?:numpy|scipy|sympy|math|tensorflow|torch)', content, re.IGNORECASE)),
                'math_functions': len(re.findall(r'\b(?:sin|cos|tan|exp|log|sqrt|pow|abs|max|min|sum|mean|std|var|corr|cov|eigen|svd|qr|lu|cholesky|fft|ifft|convolve|gradient|derivative|integral|limit|solve|optimize|minimize|maximize)\b', content, re.IGNORECASE)),
                'tensor_ops': len(re.findall(r'\b(?:tensor|matrix|vector|array|dot|matmul|transpose|reshape|flatten|concatenate|stack|split|slice|index|gather|scatter|reduce|map|fold|scan|scanl|scanr)\b', content, re.IGNORECASE)),
                'quantum_terms': len(re.findall(r'\b(?:quantum|qubit|qutrit|superposition|entanglement|decoherence|coherence|phase|amplitude|probability|measurement|gate|circuit|algorithm|shor|grover|qft|qpe|vqe|qaoa|qsvm|qnn|qml)\b', content, re.IGNORECASE)),
                'entropy_terms': len(re.findall(r'\b(?:entropy|shannon|von\s+neumann|renyi|tsallis|boltzmann|thermodynamic|information|uncertainty|randomness|chaos|fractal|attractor|bifurcation|lyapunov|kolmogorov|complexity|order|disorder)\b', content, re.IGNORECASE)),
            }
            
            # Calculate total stub score
            total_stub_score = sum(stub_indicators.values())
            
            # Calculate complexity score
            complexity_score = (
                complexity_indicators['nested_levels'] * 10 +
                complexity_indicators['function_count'] * 0.5 +
                complexity_indicators['class_count'] * 2 +
                complexity_indicators['import_count'] * 0.3
            )
            
            # Calculate math richness score
            math_richness = sum(math_indicators.values())
            
            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'line_count': line_count,
                'file_size': file_size,
                'stub_indicators': stub_indicators,
                'complexity_indicators': complexity_indicators,
                'math_indicators': math_indicators,
                'total_stub_score': total_stub_score,
                'complexity_score': complexity_score,
                'math_richness': math_richness,
                'risk_level': self._calculate_risk_level(total_stub_score, complexity_score, line_count, math_richness)
            }
            
        except Exception as e:
            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'error': str(e),
                'risk_level': 'ERROR'
            }
    
    def _count_max_nesting(self, content: str) -> int:
        """Count maximum nesting level in the file."""
        max_nesting = 0
        current_nesting = 0
        
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith(('if ', 'for ', 'while ', 'try:', 'except', 'with ', 'class ', 'def ')):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif stripped.startswith(('else:', 'elif ', 'finally:')):
                # Same level
                pass
            elif stripped == '' or stripped.startswith('#'):
                # Empty or comment
                pass
            else:
                # Check if this line reduces nesting
                if current_nesting > 0 and not line.startswith(' ' * (current_nesting * 4)):
                    current_nesting = max(0, current_nesting - 1)
        
        return max_nesting
    
    def _calculate_risk_level(self, stub_score: int, complexity_score: float, line_count: int, math_richness: int) -> str:
        """Calculate risk level for the file."""
        if stub_score > 10:
            return 'HIGH_STUB'
        elif complexity_score > 100 or line_count > 1000:
            return 'HIGH_COMPLEXITY'
        elif line_count < 50 and math_richness < 5:
            return 'LOW_VALUE'
        elif math_richness > 50:
            return 'MATH_RICH'
        else:
            return 'NORMAL'
    
    def analyze_all_files(self) -> Dict[str, Any]:
        """Analyze all Python files in the core directory."""
        python_files = list(self.core_dir.glob('*.py'))
        
        for file_path in python_files:
            self.results[file_path.name] = self.analyze_file(file_path)
        
        return self.results
    
    def generate_recommendations(self) -> Dict[str, List[str]]:
        """Generate recommendations based on analysis."""
        recommendations = {
            'DELETE': [],
            'REWRITE': [],
            'CONSOLIDATE': [],
            'KEEP': [],
            'OPTIMIZE': []
        }
        
        # Group files by risk level
        high_stub_files = []
        high_complexity_files = []
        low_value_files = []
        math_rich_files = []
        normal_files = []
        
        for file_name, analysis in self.results.items():
            if 'error' in analysis:
                recommendations['DELETE'].append(f"{file_name} (ERROR: {analysis['error']})")
                continue
                
            risk_level = analysis['risk_level']
            
            if risk_level == 'HIGH_STUB':
                high_stub_files.append((file_name, analysis))
            elif risk_level == 'HIGH_COMPLEXITY':
                high_complexity_files.append((file_name, analysis))
            elif risk_level == 'LOW_VALUE':
                low_value_files.append((file_name, analysis))
            elif risk_level == 'MATH_RICH':
                math_rich_files.append((file_name, analysis))
            else:
                normal_files.append((file_name, analysis))
        
        # Recommendations for high stub files
        for file_name, analysis in high_stub_files:
            if analysis['total_stub_score'] > 20:
                recommendations['DELETE'].append(f"{file_name} (Too many stubs: {analysis['total_stub_score']})")
            else:
                recommendations['REWRITE'].append(f"{file_name} (Stubs: {analysis['total_stub_score']})")
        
        # Recommendations for high complexity files
        for file_name, analysis in high_complexity_files:
            if analysis['line_count'] > 1500:
                recommendations['REWRITE'].append(f"{file_name} (Too large: {analysis['line_count']} lines)")
            elif analysis['complexity_score'] > 200:
                recommendations['OPTIMIZE'].append(f"{file_name} (High complexity: {analysis['complexity_score']:.1f})")
            else:
                recommendations['CONSOLIDATE'].append(f"{file_name} (Complex: {analysis['complexity_score']:.1f})")
        
        # Recommendations for low value files
        for file_name, analysis in low_value_files:
            if analysis['line_count'] < 30 and analysis['math_richness'] < 3:
                recommendations['DELETE'].append(f"{file_name} (Low value: {analysis['line_count']} lines, {analysis['math_richness']} math terms)")
            else:
                recommendations['CONSOLIDATE'].append(f"{file_name} (Small: {analysis['line_count']} lines)")
        
        # Keep math-rich files
        for file_name, analysis in math_rich_files:
            recommendations['KEEP'].append(f"{file_name} (Math-rich: {analysis['math_richness']} terms)")
        
        # Normal files
        for file_name, analysis in normal_files:
            recommendations['KEEP'].append(f"{file_name} (Normal)")
        
        return recommendations
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        recommendations = self.generate_recommendations()
        
        report = []
        report.append("=" * 80)
        report.append("SCHWABOT CORE FILE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_files = len(self.results)
        error_files = len([f for f in self.results.values() if 'error' in f])
        report.append(f"Total files analyzed: {total_files}")
        report.append(f"Files with errors: {error_files}")
        report.append("")
        
        # File size distribution
        sizes = [f['file_size'] for f in self.results.values() if 'error' not in f]
        if sizes:
            report.append(f"File size statistics:")
            report.append(f"  Largest: {max(sizes):,} bytes")
            report.append(f"  Smallest: {min(sizes):,} bytes")
            report.append(f"  Average: {sum(sizes)//len(sizes):,} bytes")
            report.append("")
        
        # Recommendations by category
        for category, files in recommendations.items():
            if files:
                report.append(f"{category} ({len(files)} files):")
                for file_info in files[:10]:  # Show first 10
                    report.append(f"  - {file_info}")
                if len(files) > 10:
                    report.append(f"  ... and {len(files) - 10} more")
                report.append("")
        
        # Top problematic files
        problematic_files = []
        for file_name, analysis in self.results.items():
            if 'error' not in analysis:
                score = analysis['total_stub_score'] + analysis['complexity_score'] / 10
                problematic_files.append((file_name, score, analysis))
        
        problematic_files.sort(key=lambda x: x[1], reverse=True)
        
        report.append("TOP 10 PROBLEMATIC FILES:")
        for i, (file_name, score, analysis) in enumerate(problematic_files[:10], 1):
            report.append(f"{i:2d}. {file_name}")
            report.append(f"     Score: {score:.1f}, Lines: {analysis['line_count']}, Stubs: {analysis['total_stub_score']}, Complexity: {analysis['complexity_score']:.1f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main analysis function."""
    analyzer = FileAnalyzer()
    print("Analyzing core files...")
    analyzer.analyze_all_files()
    
    # Generate and print report
    report = analyzer.generate_summary_report()
    print(report)
    
    # Save detailed results
    with open('file_analysis_results.json', 'w') as f:
        json.dump(analyzer.results, f, indent=2)
    
    # Save recommendations
    recommendations = analyzer.generate_recommendations()
    with open('file_recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print("\nDetailed results saved to 'file_analysis_results.json'")
    print("Recommendations saved to 'file_recommendations.json'")

if __name__ == "__main__":
    main() 