#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symbolic Math Registry for Schwabot
===================================
Dynamic registry of symbolic mathematical operators with CLI interface:
• Live registry of behavior for symbolic layer (Ω, ∇ψ, λ, Φ, etc.)
• CLI commands for operator explanation and usage
• Fallback definitions and dynamic behavior mapping
• Integration with symbolic math interface
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class SymbolType(Enum):
    """Symbol types for classification."""
    GRADIENT = "gradient"
    PHASE = "phase"
    ENTROPY = "entropy"
    OPERATOR = "operator"
    CONSTANT = "constant"
    FUNCTION = "function"

@dataclass
class SymbolDefinition:
    """Definition of a symbolic operator."""
    symbol: str
    name: str
    symbol_type: SymbolType
    description: str
    mathematical_definition: str
    python_implementation: str
    usage_examples: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    ascii_representation: str = ""
    unicode_code: str = ""
    category: str = ""

class SymbolicRegistry:
    """Dynamic registry for symbolic mathematical operators."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize symbolic registry."""
        self.logger = logging.getLogger(__name__)
        self.symbols: Dict[str, SymbolDefinition] = {}
        self.config_file = config_file
        
        # Initialize with default symbols
        self._initialize_default_symbols()
        
        # Load custom definitions if config file provided
        if config_file:
            self.load_symbols_from_file(config_file)
        
        self.logger.info("✅ Symbolic Registry initialized")
    
    def _initialize_default_symbols(self):
        """Initialize with Schwabot's core symbolic operators."""
        default_symbols = [
            # Core gradient operators
            SymbolDefinition(
                symbol="∇",
                name="Gradient Operator",
                symbol_type=SymbolType.GRADIENT,
                description="Computes gradient of signal field",
                mathematical_definition="∇ψ(t) = ∂ψ/∂t",
                python_implementation="np.gradient(signal_field)",
                usage_examples=[
                    "∇ψ(t) - Compute gradient at time t",
                    "∇ψ(t) | context - Gradient with context awareness"
                ],
                dependencies=["numpy"],
                ascii_representation="grad",
                unicode_code="U+2207",
                category="gradient"
            ),
            
            # Phase operators
            SymbolDefinition(
                symbol="Ω",
                name="Phase Omega",
                symbol_type=SymbolType.PHASE,
                description="Phase computation and momentum signals",
                mathematical_definition="Ω = ∇ψ(t) * D",
                python_implementation="gradient_value * drift_coefficient",
                usage_examples=[
                    "Ω = ∇ψ(t) * D - Basic phase omega",
                    "Ω = (∇ψ(t) * D) / Σnoise - Stable phase omega"
                ],
                dependencies=["numpy"],
                ascii_representation="Omega",
                unicode_code="U+03A9",
                category="phase"
            ),
            
            # Signal field operators
            SymbolDefinition(
                symbol="ψ",
                name="Signal Psi",
                symbol_type=SymbolType.FUNCTION,
                description="Signal field operations and state management",
                mathematical_definition="ψ = signal_field[entropy_index]",
                python_implementation="signal_field[entropy_index:entropy_index+1]",
                usage_examples=[
                    "ψ = signal_field[entropy_index] - Extract signal field",
                    "λ = entropy_weight(ψ, t) - Compute entropy weight"
                ],
                dependencies=["numpy"],
                ascii_representation="psi",
                unicode_code="U+03C8",
                category="signal"
            ),
            
            # Lambda functions
            SymbolDefinition(
                symbol="λ",
                name="Lambda Function",
                symbol_type=SymbolType.FUNCTION,
                description="Entropy weight computation",
                mathematical_definition="λ = entropy_weight(ψ, t)",
                python_implementation="compute_entropy_weight(signal_field, time_idx)",
                usage_examples=[
                    "λ = entropy_weight(ψ, t) - Entropy weight",
                    "λ = -Σ(p * log2(p)) - Shannon entropy"
                ],
                dependencies=["numpy"],
                ascii_representation="lambda",
                unicode_code="U+03BB",
                category="entropy"
            ),
            
            # Entropy delta
            SymbolDefinition(
                symbol="ΔS",
                name="Entropy Delta",
                symbol_type=SymbolType.ENTROPY,
                description="Change in entropy over time",
                mathematical_definition="ΔS = S_final - S_initial",
                python_implementation="final_entropy - initial_entropy",
                usage_examples=[
                    "ΔS = S_final - S_initial - Entropy change",
                    "ΔS > 0 - Increasing entropy"
                ],
                dependencies=["numpy"],
                ascii_representation="dS",
                unicode_code="U+0394S",
                category="entropy"
            ),
            
            # Phi operator
            SymbolDefinition(
                symbol="Φ",
                name="Phi Operator",
                symbol_type=SymbolType.OPERATOR,
                description="Phase rotation and harmonic analysis",
                mathematical_definition="Φ = cos(θ) + i*sin(θ)",
                python_implementation="np.cos(phase) + 1j * np.sin(phase)",
                usage_examples=[
                    "Φ = cos(θ) + i*sin(θ) - Complex phase",
                    "Φ = exp(i*θ) - Exponential form"
                ],
                dependencies=["numpy"],
                ascii_representation="Phi",
                unicode_code="U+03A6",
                category="phase"
            ),
            
            # Summation operator
            SymbolDefinition(
                symbol="Σ",
                name="Summation",
                symbol_type=SymbolType.OPERATOR,
                description="Summation over signal components",
                mathematical_definition="Σ(x_i) from i=1 to n",
                python_implementation="np.sum(array)",
                usage_examples=[
                    "Σ(x_i) - Sum of components",
                    "Σnoise - Sum of noise components"
                ],
                dependencies=["numpy"],
                ascii_representation="Sigma",
                unicode_code="U+03A3",
                category="operator"
            ),
            
            # Collapse function
            SymbolDefinition(
                symbol="C(t)",
                name="Collapse Function",
                symbol_type=SymbolType.FUNCTION,
                description="Quantum state collapse over time",
                mathematical_definition="C(t) = |ψ(t)|²",
                python_implementation="np.abs(signal_field)**2",
                usage_examples=[
                    "C(t) = |ψ(t)|² - Probability density",
                    "C(t) = collapse_probability(signal, t)"
                ],
                dependencies=["numpy"],
                ascii_representation="C(t)",
                unicode_code="C(t)",
                category="quantum"
            ),
            
            # Epsilon entropy
            SymbolDefinition(
                symbol="𝓔",
                name="Epsilon Entropy",
                symbol_type=SymbolType.ENTROPY,
                description="Small-scale entropy fluctuations",
                mathematical_definition="𝓔 = ε * H(signal)",
                python_implementation="epsilon * shannon_entropy(signal)",
                usage_examples=[
                    "𝓔 = ε * H(signal) - Small entropy",
                    "𝓔 < threshold - Negligible entropy"
                ],
                dependencies=["numpy"],
                ascii_representation="E",
                unicode_code="U+1D4D3",
                category="entropy"
            ),
            
            # Partial derivative
            SymbolDefinition(
                symbol="∂",
                name="Partial Derivative",
                symbol_type=SymbolType.OPERATOR,
                description="Partial derivative operator",
                mathematical_definition="∂f/∂x",
                python_implementation="np.gradient(f, x)",
                usage_examples=[
                    "∂ψ/∂t - Time derivative",
                    "∂f/∂x - Spatial derivative"
                ],
                dependencies=["numpy"],
                ascii_representation="d",
                unicode_code="U+2202",
                category="operator"
            ),
            
            # Kappa mapper
            SymbolDefinition(
                symbol="κ",
                name="Kappa Mapper",
                symbol_type=SymbolType.FUNCTION,
                description="Signal mapping and transformation",
                mathematical_definition="κ: X → Y",
                python_implementation="mapping_function(input_signal)",
                usage_examples=[
                    "κ: X → Y - Signal mapping",
                    "κ(signal) = transformed_signal"
                ],
                dependencies=["numpy"],
                ascii_representation="kappa",
                unicode_code="U+03BA",
                category="mapping"
            ),
            
            # Conductance logic
            SymbolDefinition(
                symbol="℧",
                name="Conductance Logic",
                symbol_type=SymbolType.OPERATOR,
                description="Signal conductance and flow control",
                mathematical_definition="℧ = 1/R",
                python_implementation="1.0 / resistance",
                usage_examples=[
                    "℧ = 1/R - Conductance",
                    "℧(signal) = conductance_factor"
                ],
                dependencies=["numpy"],
                ascii_representation="mho",
                unicode_code="U+2127",
                category="conductance"
            ),
            
            # Xi field
            SymbolDefinition(
                symbol="Ξ",
                name="Xi Field",
                symbol_type=SymbolType.FUNCTION,
                description="Cross-asset correlation field",
                mathematical_definition="Ξ = correlation_matrix(assets)",
                python_implementation="np.corrcoef(asset_signals)",
                usage_examples=[
                    "Ξ = correlation_matrix(assets) - Asset correlations",
                    "Ξ[i,j] - Correlation between assets i and j"
                ],
                dependencies=["numpy"],
                ascii_representation="Xi",
                unicode_code="U+039E",
                category="correlation"
            ),
            
            # Ghost tick
            SymbolDefinition(
                symbol="𝕏",
                name="Ghost Tick",
                symbol_type=SymbolType.FUNCTION,
                description="Phantom signal detection",
                mathematical_definition="𝕏 = phantom_detector(signal)",
                python_implementation="detect_phantom_signal(signal)",
                usage_examples=[
                    "𝕏 = phantom_detector(signal) - Phantom detection",
                    "𝕏 > threshold - Phantom signal detected"
                ],
                dependencies=["numpy"],
                ascii_representation="X",
                unicode_code="U+1D54F",
                category="phantom"
            )
        ]
        
        for symbol_def in default_symbols:
            self.register_symbol(symbol_def)
    
    def register_symbol(self, symbol_def: SymbolDefinition) -> bool:
        """Register a new symbolic operator."""
        try:
            self.symbols[symbol_def.symbol] = symbol_def
            self.logger.debug(f"Registered symbol: {symbol_def.symbol} - {symbol_def.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error registering symbol {symbol_def.symbol}: {e}")
            return False
    
    def get_symbol(self, symbol: str) -> Optional[SymbolDefinition]:
        """Get symbol definition by symbol."""
        return self.symbols.get(symbol)
    
    def get_symbols_by_type(self, symbol_type: SymbolType) -> List[SymbolDefinition]:
        """Get all symbols of a specific type."""
        return [s for s in self.symbols.values() if s.symbol_type == symbol_type]
    
    def get_symbols_by_category(self, category: str) -> List[SymbolDefinition]:
        """Get all symbols in a specific category."""
        return [s for s in self.symbols.values() if s.category == category]
    
    def search_symbols(self, query: str) -> List[SymbolDefinition]:
        """Search symbols by name, description, or mathematical definition."""
        query_lower = query.lower()
        results = []
        
        for symbol_def in self.symbols.values():
            if (query_lower in symbol_def.name.lower() or
                query_lower in symbol_def.description.lower() or
                query_lower in symbol_def.mathematical_definition.lower() or
                query_lower in symbol_def.ascii_representation.lower()):
                results.append(symbol_def)
        
        return results
    
    def explain_symbol(self, symbol: str) -> Optional[str]:
        """Generate detailed explanation of a symbol."""
        symbol_def = self.get_symbol(symbol)
        if not symbol_def:
            return None
        
        explanation = f"""
Symbol: {symbol_def.symbol} ({symbol_def.ascii_representation})
Name: {symbol_def.name}
Type: {symbol_def.symbol_type.value}
Category: {symbol_def.category}

Description:
{symbol_def.description}

Mathematical Definition:
{symbol_def.mathematical_definition}

Python Implementation:
{symbol_def.python_implementation}

Usage Examples:
"""
        
        for i, example in enumerate(symbol_def.usage_examples, 1):
            explanation += f"{i}. {example}\n"
        
        if symbol_def.dependencies:
            explanation += f"\nDependencies: {', '.join(symbol_def.dependencies)}"
        
        if symbol_def.unicode_code:
            explanation += f"\nUnicode: {symbol_def.unicode_code}"
        
        return explanation
    
    def list_all_symbols(self) -> str:
        """List all registered symbols."""
        if not self.symbols:
            return "No symbols registered."
        
        result = "Registered Symbols:\n"
        result += "=" * 50 + "\n"
        
        # Group by category
        categories = {}
        for symbol_def in self.symbols.values():
            category = symbol_def.category
            if category not in categories:
                categories[category] = []
            categories[category].append(symbol_def)
        
        for category, symbols in sorted(categories.items()):
            result += f"\n{category.upper()}:\n"
            for symbol_def in symbols:
                result += f"  {symbol_def.symbol} ({symbol_def.ascii_representation}) - {symbol_def.name}\n"
        
        return result
    
    def export_symbols(self, filename: str) -> bool:
        """Export symbols to JSON file."""
        try:
            export_data = {}
            for symbol, definition in self.symbols.items():
                export_data[symbol] = {
                    'name': definition.name,
                    'symbol_type': definition.symbol_type.value,
                    'description': definition.description,
                    'mathematical_definition': definition.mathematical_definition,
                    'python_implementation': definition.python_implementation,
                    'usage_examples': definition.usage_examples,
                    'dependencies': definition.dependencies,
                    'ascii_representation': definition.ascii_representation,
                    'unicode_code': definition.unicode_code,
                    'category': definition.category
                }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported symbols to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting symbols: {e}")
            return False
    
    def load_symbols_from_file(self, filename: str) -> bool:
        """Load symbols from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            for symbol, data in import_data.items():
                symbol_def = SymbolDefinition(
                    symbol=symbol,
                    name=data['name'],
                    symbol_type=SymbolType(data['symbol_type']),
                    description=data['description'],
                    mathematical_definition=data['mathematical_definition'],
                    python_implementation=data['python_implementation'],
                    usage_examples=data.get('usage_examples', []),
                    dependencies=data.get('dependencies', []),
                    ascii_representation=data.get('ascii_representation', ''),
                    unicode_code=data.get('unicode_code', ''),
                    category=data.get('category', '')
                )
                self.register_symbol(symbol_def)
            
            self.logger.info(f"Loaded symbols from {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading symbols from {filename}: {e}")
            return False

def main():
    """CLI interface for symbolic registry."""
    parser = argparse.ArgumentParser(description="Schwabot Symbolic Math Registry CLI")
    parser.add_argument('command', choices=['explain', 'list', 'search', 'export', 'import'],
                       help='Command to execute')
    parser.add_argument('--symbol', '-s', help='Symbol to explain')
    parser.add_argument('--query', '-q', help='Search query')
    parser.add_argument('--file', '-f', help='File for export/import')
    parser.add_argument('--config', '-c', help='Registry config file')
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = SymbolicRegistry(args.config)
    
    if args.command == 'explain':
        if not args.symbol:
            print("Error: --symbol required for explain command")
            sys.exit(1)
        
        explanation = registry.explain_symbol(args.symbol)
        if explanation:
            print(explanation)
        else:
            print(f"Symbol '{args.symbol}' not found")
    
    elif args.command == 'list':
        print(registry.list_all_symbols())
    
    elif args.command == 'search':
        if not args.query:
            print("Error: --query required for search command")
            sys.exit(1)
        
        results = registry.search_symbols(args.query)
        if results:
            print(f"Search results for '{args.query}':")
            print("=" * 40)
            for symbol_def in results:
                print(f"{symbol_def.symbol} - {symbol_def.name}")
                print(f"  {symbol_def.description}")
                print()
        else:
            print(f"No symbols found matching '{args.query}'")
    
    elif args.command == 'export':
        if not args.file:
            print("Error: --file required for export command")
            sys.exit(1)
        
        if registry.export_symbols(args.file):
            print(f"Symbols exported to {args.file}")
        else:
            print("Export failed")
            sys.exit(1)
    
    elif args.command == 'import':
        if not args.file:
            print("Error: --file required for import command")
            sys.exit(1)
        
        if registry.load_symbols_from_file(args.file):
            print(f"Symbols imported from {args.file}")
        else:
            print("Import failed")
            sys.exit(1)

if __name__ == "__main__":
    main() 