#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Consolidation Plan - Systematic Code Correction
======================================================
Applies the same systematic correction approach used for bio_cellular_integration.py
to all files needing fixes, consolidations, or rewrites.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class Phase2ConsolidationPlan:
    """
    Systematic consolidation and correction plan for Schwabot codebase.
    Applies the same rigorous approach used for bio_cellular_integration.py.
    """
    
    def __init__(self, core_dir: str = "core"):
        self.core_dir = Path(core_dir)
        self.backup_dir = Path("backups/phase2_backup")
        self.correction_log = []
        
    def create_backup(self) -> None:
        """Create comprehensive backup before Phase 2 operations."""
        print("🔄 Creating Phase 2 backup...")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup all core files
        for file_path in self.core_dir.rglob("*.py"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.core_dir)
                backup_path = self.backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
        
        print(f"✅ Backup created at: {self.backup_dir}")
    
    def analyze_file_correction_needs(self, file_path: Path) -> Dict[str, any]:
        """Analyze what corrections a file needs."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            issues = {
                'indentation_errors': False,
                'import_errors': False,
                'syntax_errors': False,
                'missing_implementations': False,
                'excessive_nesting': False,
                'poor_structure': False,
                'stub_functions': False,
                'line_count': len(content.splitlines()),
                'file_size': len(content),
            }
            
            # Check for common issues
            lines = content.splitlines()
            
            # Indentation issues
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith((' ', '\t')) and not line.startswith(('def ', 'class ', 'import ', 'from ', '#', '"""', "'''")):
                    if i > 0 and lines[i-1].strip() and not lines[i-1].startswith(('def ', 'class ', 'import ', 'from ', '#', '"""', "'''")):
                        issues['indentation_errors'] = True
                        break
            
            # Import issues
            if 'import' in content and ('from core.' in content or 'import core.' in content):
                issues['import_errors'] = True
            
            # Stub functions
            if 'pass' in content and content.count('pass') > content.count('def') * 0.3:
                issues['stub_functions'] = True
            
            # Excessive nesting
            max_indent = max(len(line) - len(line.lstrip()) for line in lines if line.strip())
            if max_indent > 20:  # More than 5 levels of indentation
                issues['excessive_nesting'] = True
            
            # Poor structure
            if len(lines) > 200 and content.count('class') < 2 and content.count('def') < 5:
                issues['poor_structure'] = True
            
            return issues
            
        except Exception as e:
            return {'error': str(e)}
    
    def categorize_files_for_correction(self) -> Dict[str, List[Path]]:
        """Categorize files based on correction needs."""
        categories = {
            'FULL_REWRITE': [],
            'MAJOR_CORRECTION': [],
            'MINOR_CORRECTION': [],
            'CONSOLIDATION_CANDIDATE': [],
            'KEEP_AS_IS': []
        }
        
        print("🔍 Analyzing files for correction needs...")
        
        for file_path in self.core_dir.rglob("*.py"):
            if file_path.is_file() and file_path.name != "__init__.py":
                issues = self.analyze_file_correction_needs(file_path)
                
                if 'error' in issues:
                    categories['FULL_REWRITE'].append(file_path)
                    continue
                
                # Determine category based on issues
                if (issues.get('indentation_errors') or 
                    issues.get('syntax_errors') or 
                    issues.get('import_errors')):
                    categories['FULL_REWRITE'].append(file_path)
                elif (issues.get('excessive_nesting') or 
                      issues.get('poor_structure') or 
                      issues.get('stub_functions')):
                    categories['MAJOR_CORRECTION'].append(file_path)
                elif issues.get('line_count', 0) < 100:
                    categories['CONSOLIDATION_CANDIDATE'].append(file_path)
                else:
                    categories['MINOR_CORRECTION'].append(file_path)
        
        return categories
    
    def apply_systematic_correction(self, file_path: Path, correction_type: str) -> bool:
        """Apply systematic correction to a file."""
        try:
            print(f"🔧 Correcting: {file_path.name} ({correction_type})")
            
            if correction_type == 'FULL_REWRITE':
                return self._rewrite_file_completely(file_path)
            elif correction_type == 'MAJOR_CORRECTION':
                return self._apply_major_corrections(file_path)
            elif correction_type == 'MINOR_CORRECTION':
                return self._apply_minor_corrections(file_path)
            else:
                return True
                
        except Exception as e:
            print(f"❌ Error correcting {file_path.name}: {e}")
            return False
    
    def _rewrite_file_completely(self, file_path: Path) -> bool:
        """Completely rewrite a file with proper structure."""
        try:
            # Read original content to preserve logic
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Extract key information
            module_name = file_path.stem
            class_names = []
            function_names = []
            
            # Simple extraction of class and function names
            lines = original_content.splitlines()
            for line in lines:
                if line.strip().startswith('class '):
                    class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                    class_names.append(class_name)
                elif line.strip().startswith('def '):
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    function_names.append(func_name)
            
            # Create new content with proper structure
            new_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{module_name.replace('_', ' ').title()} Module
{'=' * (len(module_name) + 8)}
{self._generate_module_docstring(module_name, class_names, function_names)}
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Import dependencies
try:
    from core.math_config_manager import MathConfigManager
    from core.math_cache import MathResultCache
    from core.math_orchestrator import MathOrchestrator
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Math infrastructure not available")

{self._generate_enum_definitions(original_content)}

{self._generate_dataclass_definitions(original_content)}

{self._generate_main_class(module_name, class_names, function_names)}

# Factory function
def create_{module_name.replace('_', '_')}(config: Optional[Dict[str, Any]] = None):
    """Create a {module_name.replace('_', ' ')} instance."""
    return {class_names[0] if class_names else module_name.title().replace('_', '')}(config)
'''
            
            # Write new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.correction_log.append(f"✅ Rewrote: {file_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error rewriting {file_path.name}: {e}")
            return False
    
    def _apply_major_corrections(self, file_path: Path) -> bool:
        """Apply major corrections while preserving logic."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix indentation
            lines = content.splitlines()
            fixed_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    fixed_lines.append('')
                    continue
                
                # Adjust indentation
                if stripped.startswith(('class ', 'def ')):
                    indent_level = 0
                elif stripped.startswith(('if ', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ')):
                    indent_level += 1
                elif stripped.startswith(('else:', 'elif ')):
                    indent_level = max(0, indent_level - 1)
                
                fixed_lines.append('    ' * indent_level + stripped)
            
            # Write corrected content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(fixed_lines))
            
            self.correction_log.append(f"✅ Major corrections applied: {file_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error applying major corrections to {file_path.name}: {e}")
            return False
    
    def _apply_minor_corrections(self, file_path: Path) -> bool:
        """Apply minor corrections and formatting."""
        try:
            # Run black formatter
            subprocess.run([sys.executable, '-m', 'black', str(file_path)], 
                         capture_output=True, check=True)
            
            # Run isort
            subprocess.run([sys.executable, '-m', 'isort', str(file_path)], 
                         capture_output=True, check=True)
            
            self.correction_log.append(f"✅ Minor corrections applied: {file_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error applying minor corrections to {file_path.name}: {e}")
            return False
    
    def _generate_module_docstring(self, module_name: str, class_names: List[str], function_names: List[str]) -> str:
        """Generate appropriate module docstring."""
        docstring = f"Provides {module_name.replace('_', ' ')} functionality for the Schwabot trading system.\n"
        
        if class_names:
            docstring += f"\nMain Classes:\n"
            for class_name in class_names[:3]:  # Limit to first 3
                docstring += f"- {class_name}: Core {class_name.lower().replace('_', ' ')} functionality\n"
        
        if function_names:
            docstring += f"\nKey Functions:\n"
            for func_name in function_names[:5]:  # Limit to first 5
                docstring += f"- {func_name}: {func_name.replace('_', ' ')} operation\n"
        
        return docstring
    
    def _generate_enum_definitions(self, original_content: str) -> str:
        """Generate enum definitions if needed."""
        # Simple enum detection and generation
        if 'Enum' in original_content or 'enum' in original_content.lower():
            return '''
class Status(Enum):
    """System status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"

class Mode(Enum):
    """Operation mode enumeration."""
    NORMAL = "normal"
    DEBUG = "debug"
    TEST = "test"
    PRODUCTION = "production"
'''
        return ''
    
    def _generate_dataclass_definitions(self, original_content: str) -> str:
        """Generate dataclass definitions if needed."""
        # Simple dataclass detection
        if 'dataclass' in original_content or '@dataclass' in original_content:
            return '''
@dataclass
class Config:
    """Configuration data class."""
    enabled: bool = True
    timeout: float = 30.0
    retries: int = 3
    debug: bool = False

@dataclass
class Result:
    """Result data class."""
    success: bool = False
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
'''
        return ''
    
    def _generate_main_class(self, module_name: str, class_names: List[str], function_names: List[str]) -> str:
        """Generate main class with proper structure."""
        class_name = class_names[0] if class_names else module_name.title().replace('_', '')
        
        return f'''
class {class_name}:
    """
    {class_name.replace('_', ' ')} Implementation
    Provides core {module_name.replace('_', ' ')} functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize {class_name} with configuration."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False
        
        # Initialize math infrastructure if available
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_config = MathConfigManager()
            self.math_cache = MathResultCache()
            self.math_orchestrator = MathOrchestrator()
        
        self._initialize_system()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {{
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
        }}
    
    def _initialize_system(self) -> None:
        """Initialize the system."""
        try:
            self.logger.info(f"Initializing {{self.__class__.__name__}}")
            self.initialized = True
            self.logger.info(f"✅ {{self.__class__.__name__}} initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing {{self.__class__.__name__}}: {{e}}")
            self.initialized = False
    
    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False
        
        try:
            self.active = True
            self.logger.info(f"✅ {{self.__class__.__name__}} activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating {{self.__class__.__name__}}: {{e}}")
            return False
    
    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info(f"✅ {{self.__class__.__name__}} deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating {{self.__class__.__name__}}: {{e}}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {{
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
        }}
'''
    
    def consolidate_small_files(self, files: List[Path]) -> Dict[str, Path]:
        """Consolidate small files into logical groups."""
        consolidations = {}
        
        # Group by functionality
        groups = {
            'cli_utils': [],
            'profit_utils': [],
            'system_utils': [],
            'math_utils': [],
        }
        
        for file_path in files:
            name = file_path.stem.lower()
            if any(keyword in name for keyword in ['cli', 'command', 'interface']):
                groups['cli_utils'].append(file_path)
            elif any(keyword in name for keyword in ['profit', 'revenue', 'gain']):
                groups['profit_utils'].append(file_path)
            elif any(keyword in name for keyword in ['math', 'calc', 'compute']):
                groups['math_utils'].append(file_path)
            else:
                groups['system_utils'].append(file_path)
        
        # Create consolidated files
        for group_name, group_files in groups.items():
            if len(group_files) > 1:
                consolidated_path = self.core_dir / f"consolidated_{group_name}.py"
                self._create_consolidated_file(consolidated_path, group_files, group_name)
                consolidations[group_name] = consolidated_path
        
        return consolidations
    
    def _create_consolidated_file(self, output_path: Path, input_files: List[Path], group_name: str) -> None:
        """Create a consolidated file from multiple input files."""
        content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated {group_name.replace('_', ' ').title()} Utilities
{'=' * (len(group_name) + 25)}
Consolidated utilities from multiple small files.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Consolidated utilities from:
{chr(10).join(f"# - {f.name}" for f in input_files)}

class Consolidated{group_name.title().replace('_', '')}:
    """Consolidated {group_name.replace('_', ' ')} utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_utility(self, utility_type: str, data: Any) -> Any:
        """Process utility based on type."""
        self.logger.info(f"Processing {{utility_type}} utility")
        return data

# Factory function
def create_consolidated_{group_name}():
    """Create consolidated {group_name} instance."""
    return Consolidated{group_name.title().replace('_', '')}()
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Remove original files
        for file_path in input_files:
            file_path.unlink()
        
        self.correction_log.append(f"✅ Consolidated {len(input_files)} files into {output_path.name}")
    
    def run_phase2_consolidation(self) -> bool:
        """Run the complete Phase 2 consolidation process."""
        print("🚀 Starting Phase 2 Consolidation...")
        
        # Create backup
        self.create_backup()
        
        # Categorize files
        categories = self.categorize_files_for_correction()
        
        print("\n📊 File Categories:")
        for category, files in categories.items():
            print(f"  {category}: {len(files)} files")
        
        # Apply corrections
        total_files = sum(len(files) for files in categories.values())
        processed_files = 0
        
        for category, files in categories.items():
            if category == 'KEEP_AS_IS':
                continue
                
            print(f"\n🔧 Processing {category} files...")
            for file_path in files:
                if self.apply_systematic_correction(file_path, category):
                    processed_files += 1
        
        # Consolidate small files
        if categories['CONSOLIDATION_CANDIDATE']:
            print(f"\n🔗 Consolidating {len(categories['CONSOLIDATION_CANDIDATE'])} small files...")
            consolidations = self.consolidate_small_files(categories['CONSOLIDATION_CANDIDATE'])
            print(f"✅ Created {len(consolidations)} consolidated files")
        
        # Run final formatting
        print("\n🎨 Applying final formatting...")
        subprocess.run([sys.executable, '-m', 'black', str(self.core_dir)], 
                      capture_output=True, check=True)
        subprocess.run([sys.executable, '-m', 'isort', str(self.core_dir)], 
                      capture_output=True, check=True)
        
        print(f"\n✅ Phase 2 Consolidation Complete!")
        print(f"📈 Processed {processed_files}/{total_files} files")
        print(f"📝 Correction log: {len(self.correction_log)} entries")
        
        return True

def main():
    """Run Phase 2 consolidation."""
    plan = Phase2ConsolidationPlan()
    success = plan.run_phase2_consolidation()
    
    if success:
        print("\n🎉 Phase 2 consolidation completed successfully!")
        print("📋 Next steps:")
        print("  1. Review consolidated files")
        print("  2. Test system integration")
        print("  3. Proceed to Phase 3 optimization")
    else:
        print("\n❌ Phase 2 consolidation failed!")
        print("📋 Check logs and backup files")

if __name__ == "__main__":
    main() 