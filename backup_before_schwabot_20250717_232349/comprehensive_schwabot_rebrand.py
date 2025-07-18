#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Schwabot Comprehensive Rebranding Script

This script completely transforms the system from schwabot_ai to Schwabot,
updating all references, configurations, visual elements, and system branding.

Features:
- Complete schwabot_ai → Schwabot rebranding
- Visual layer updates with Schwabot branding
- Configuration file updates
- Test system updates
- Documentation updates
- GUI and web interface rebranding
"""

import os
import sys
import re
import json
import yaml
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchwabotRebrander:
    """Comprehensive rebranding system for Schwabot."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_before_rebrand"
        self.rebrand_stats = {
            'files_processed': 0,
            'files_updated': 0,
            'errors': [],
            'warnings': []
        }
        
        # Schwabot branding configuration
        self.schwabot_branding = {
            'system_name': 'Schwabot AI',
            'system_description': 'Advanced AI-Powered Trading System',
            'version': '2.0.0',
            'author': 'Schwabot Development Team',
            'website': 'https://schwabot.ai',
            'support_email': 'support@schwabot.ai',
            'logo_ascii': '''
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    🚀 SCHWABOT AI 🚀                        ║
    ║                                                              ║
    ║              Advanced AI-Powered Trading System              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
            ''',
            'colors': {
                'primary': '#1E3A8A',      # Blue
                'secondary': '#059669',    # Green
                'accent': '#DC2626',       # Red
                'warning': '#D97706',      # Orange
                'success': '#059669',      # Green
                'info': '#2563EB'          # Blue
            }
        }
        
        # File patterns to process
        self.file_patterns = [
            '*.py', '*.json', '*.yaml', '*.yml', '*.md', '*.txt',
            '*.html', '*.css', '*.js', '*.cfg', '*.ini', '*.toml'
        ]
        
        # Directories to exclude
        self.exclude_dirs = {
            '.git', '__pycache__', 'node_modules', 'venv', 'env',
            'backup_before_rebrand', 'archive', 'logs'
        }
        
        # Rebranding mappings
        self.rebrand_mappings = {
            # System names
            'schwabot_ai': 'schwabot_ai',
            'Schwabot AI': 'SchwabotAI',
            'Schwabot AI': 'Schwabot AI',
            'schwabot': 'schwabot',
            'Schwabot': 'Schwabot',
            
            # Class names
            'Schwabot AIIntegration': 'SchwabotAIIntegration',
            'Schwabot AIBridge': 'SchwabotAIBridge',
            'Schwabot AIEnhancedInterface': 'SchwabotAIEnhancedInterface',
            'Schwabot AISetup': 'SchwabotAISetup',
            'Schwabot AIMaster': 'SchwabotAIMaster',
            
            # File names
            'schwabot_ai_integration.py': 'schwabot_ai_integration.py',
            'schwabot_ai_bridge.py': 'schwabot_ai_bridge.py',
            'setup_schwabot_ai.py': 'setup_schwabot_ai.py',
            'schwabot_ai_config.json': 'schwabot_ai_config.json',
            
            # URLs and endpoints
            'http://localhost:8080': 'http://localhost:8080',
            'schwabot_ai-windows': 'schwabot-ai-windows',
            'schwabot_ai-linux': 'schwabot-ai-linux',
            'schwabot_ai-macos': 'schwabot-ai-macos',
            
            # Configuration keys
            'schwabot_path': 'schwabot_ai_path',
            'schwabot_port': 'schwabot_ai_port',
            'schwabot_model': 'schwabot_ai_model',
            'schwabot_config': 'schwabot_ai_config',
            
            # UI elements
            'Schwabot AI Interface': 'Schwabot AI Interface',
            'Schwabot AI Server': 'Schwabot AI Server',
            'Schwabot AI Model': 'Schwabot AI Model',
            'Schwabot AI Analysis': 'Schwabot AI Analysis',
            
            # Documentation
            'Schwabot AI Setup': 'Schwabot AI Setup',
            'Schwabot AI Integration': 'Schwabot AI Integration',
            'Schwabot AI Configuration': 'Schwabot AI Configuration'
        }
    
    def create_backup(self) -> bool:
        """Create backup of current system before rebranding."""
        try:
            logger.info("📦 Creating backup before rebranding...")
            
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            # Copy all files except backup and cache directories
            shutil.copytree(
                self.project_root,
                self.backup_dir,
                ignore=shutil.ignore_patterns(
                    'backup_before_rebrand',
                    '__pycache__',
                    '*.pyc',
                    '.git',
                    'logs',
                    'cache'
                )
            )
            
            logger.info(f"✅ Backup created at: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return False
    
    def find_files_to_process(self) -> List[Path]:
        """Find all files that need to be processed for rebranding."""
        files_to_process = []
        
        for pattern in self.file_patterns:
            for file_path in self.project_root.rglob(pattern):
                # Skip excluded directories
                if any(exclude in file_path.parts for exclude in self.exclude_dirs):
                    continue
                
                # Skip backup directory
                if self.backup_dir in file_path.parents:
                    continue
                
                files_to_process.append(file_path)
        
        logger.info(f"📁 Found {len(files_to_process)} files to process")
        return files_to_process
    
    def update_file_content(self, file_path: Path) -> bool:
        """Update content of a single file with rebranding."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply rebranding mappings
            for old_text, new_text in self.rebrand_mappings.items():
                content = content.replace(old_text, new_text)
            
            # Update specific patterns
            content = self._update_specific_patterns(content, file_path)
            
            # Write updated content if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.rebrand_stats['files_updated'] += 1
                logger.info(f"✅ Updated: {file_path}")
                return True
            else:
                logger.debug(f"⏭️  No changes needed: {file_path}")
                return False
                
        except Exception as e:
            error_msg = f"Failed to update {file_path}: {e}"
            self.rebrand_stats['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def _update_specific_patterns(self, content: str, file_path: Path) -> str:
        """Update specific patterns based on file type."""
        file_suffix = file_path.suffix.lower()
        
        # Python files
        if file_suffix == '.py':
            content = self._update_python_patterns(content)
        
        # Configuration files
        elif file_suffix in ['.yaml', '.yml', '.json']:
            content = self._update_config_patterns(content)
        
        # Documentation files
        elif file_suffix in ['.md', '.txt']:
            content = self._update_doc_patterns(content)
        
        # HTML/CSS files
        elif file_suffix in ['.html', '.css']:
            content = self._update_web_patterns(content)
        
        return content
    
    def _update_python_patterns(self, content: str) -> str:
        """Update Python-specific patterns."""
        # Update import statements
        content = re.sub(
            r'from\s+core\.schwabot_ai_',
            'from core.schwabot_ai_',
            content
        )
        content = re.sub(
            r'import\s+core\.schwabot_ai_',
            'import core.schwabot_ai_',
            content
        )
        
        # Update class references
        content = re.sub(
            r'Schwabot AI[A-Za-z]*',
            lambda m: self.rebrand_mappings.get(m.group(0), m.group(0)),
            content
        )
        
        # Update function names
        content = re.sub(
            r'def\s+schwabot_ai_',
            'def schwabot_ai_',
            content
        )
        
        return content
    
    def _update_config_patterns(self, content: str) -> str:
        """Update configuration-specific patterns."""
        # Update configuration keys
        for old_key, new_key in [
            ('schwabot_ai', 'schwabot_ai'),
            ('schwabot_path', 'schwabot_ai_path'),
            ('schwabot_port', 'schwabot_ai_port'),
            ('schwabot_model', 'schwabot_ai_model'),
            ('schwabot_config', 'schwabot_ai_config'),
        ]:
            content = content.replace(old_key, new_key)
        
        return content
    
    def _update_doc_patterns(self, content: str) -> str:
        """Update documentation-specific patterns."""
        # Update titles and headers
        content = re.sub(
            r'#\s*Schwabot AI',
            '# Schwabot AI',
            content
        )
        
        # Update system descriptions
        content = content.replace(
            'Schwabot AI trading system',
            'Schwabot AI trading system'
        )
        content = content.replace(
            'Schwabot AI integration',
            'Schwabot AI integration'
        )
        
        return content
    
    def _update_web_patterns(self, content: str) -> str:
        """Update web-specific patterns."""
        # Update titles
        content = re.sub(
            r'<title>.*?Schwabot AI.*?</title>',
            f'<title>Schwabot AI - {self.schwabot_branding["system_description"]}</title>',
            content
        )
        
        # Update branding
        content = content.replace('Schwabot AI', 'Schwabot AI')
        content = content.replace('schwabot_ai', 'schwabot-ai')
        
        return content
    
    def rename_files(self) -> bool:
        """Rename files that contain schwabot_ai in their names."""
        try:
            logger.info("🔄 Renaming files with schwabot_ai references...")
            
            renamed_count = 0
            for file_path in self.project_root.rglob('*'):
                if file_path.is_file() and 'schwabot_ai' in file_path.name.lower():
                    new_name = file_path.name.replace('schwabot_ai', 'schwabot_ai')
                    new_path = file_path.parent / new_name
                    
                    try:
                        file_path.rename(new_path)
                        renamed_count += 1
                        logger.info(f"✅ Renamed: {file_path.name} → {new_name}")
                    except Exception as e:
                        error_msg = f"Failed to rename {file_path}: {e}"
                        self.rebrand_stats['errors'].append(error_msg)
                        logger.error(f"❌ {error_msg}")
            
            logger.info(f"✅ Renamed {renamed_count} files")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to rename files: {e}")
            return False
    
    def update_config_files(self) -> bool:
        """Update configuration files with Schwabot branding."""
        try:
            logger.info("⚙️  Updating configuration files...")
            
            config_files = [
                'config/schwabot_config.yaml',
                'config/schwabot_config.json',
                'config/integrations.yaml',
                'config/master_integration.yaml',
                'config/unified_settings.yaml'
            ]
            
            for config_file in config_files:
                config_path = self.project_root / config_file
                if config_path.exists():
                    self._update_config_file(config_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update config files: {e}")
            return False
    
    def _update_config_file(self, config_path: Path) -> None:
        """Update a specific configuration file."""
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Update configuration
                config = self._update_config_dict(config)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                    
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Update configuration
                config = self._update_config_dict(config)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"✅ Updated config: {config_path}")
            
        except Exception as e:
            error_msg = f"Failed to update config {config_path}: {e}"
            self.rebrand_stats['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
    
    def _update_config_dict(self, config: Any) -> Any:
        """Recursively update configuration dictionary."""
        if isinstance(config, dict):
            updated_config = {}
            for key, value in config.items():
                # Update keys
                new_key = key
                for old_key, new_key_name in [
                    ('schwabot_ai', 'schwabot_ai'),
                    ('schwabot_path', 'schwabot_ai_path'),
                    ('schwabot_port', 'schwabot_ai_port'),
                    ('schwabot_model', 'schwabot_ai_model'),
                ]:
                    if old_key in key:
                        new_key = key.replace(old_key, new_key_name)
                        break
                
                # Update values
                updated_config[new_key] = self._update_config_dict(value)
            
            return updated_config
        
        elif isinstance(config, list):
            return [self._update_config_dict(item) for item in config]
        
        elif isinstance(config, str):
            # Update string values
            for old_text, new_text in self.rebrand_mappings.items():
                if old_text in config:
                    config = config.replace(old_text, new_text)
            return config
        
        else:
            return config
    
    def create_schwabot_branding_files(self) -> bool:
        """Create Schwabot-specific branding files."""
        try:
            logger.info("🎨 Creating Schwabot branding files...")
            
            # Create Schwabot logo file
            logo_content = self.schwabot_branding['logo_ascii']
            logo_file = self.project_root / 'static' / 'schwabot_logo.txt'
            logo_file.parent.mkdir(exist_ok=True)
            
            with open(logo_file, 'w', encoding='utf-8') as f:
                f.write(logo_content)
            
            # Create Schwabot branding configuration
            branding_config = {
                'system_name': self.schwabot_branding['system_name'],
                'version': self.schwabot_branding['version'],
                'description': self.schwabot_branding['system_description'],
                'author': self.schwabot_branding['author'],
                'website': self.schwabot_branding['website'],
                'support_email': self.schwabot_branding['support_email'],
                'colors': self.schwabot_branding['colors'],
                'created_date': datetime.now().isoformat(),
                'rebranded_from': 'schwabot_ai'
            }
            
            branding_file = self.project_root / 'config' / 'schwabot_branding.json'
            with open(branding_file, 'w', encoding='utf-8') as f:
                json.dump(branding_config, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Created Schwabot branding files")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create branding files: {e}")
            return False
    
    def update_test_files(self) -> bool:
        """Update test files to use Schwabot branding."""
        try:
            logger.info("🧪 Updating test files...")
            
            test_files = [
                'test_system.py',
                'test_imports.py',
                'simple_test.py',
                'comprehensive_mathematical_restoration_test.py'
            ]
            
            for test_file in test_files:
                test_path = self.project_root / test_file
                if test_path.exists():
                    self.update_file_content(test_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update test files: {e}")
            return False
    
    def update_documentation(self) -> bool:
        """Update documentation files."""
        try:
            logger.info("📚 Updating documentation...")
            
            doc_files = [
                'README.md',
                'SYSTEM_STATUS_REPORT.md',
                'FINAL_INTEGRATION_STATUS_REPORT.md',
                'QUICK_SAFE_FIXES_SUMMARY.md'
            ]
            
            for doc_file in doc_files:
                doc_path = self.project_root / doc_file
                if doc_path.exists():
                    self.update_file_content(doc_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update documentation: {e}")
            return False
    
    def run_comprehensive_rebrand(self) -> Dict[str, Any]:
        """Run the complete rebranding process."""
        logger.info("🚀 Starting comprehensive Schwabot rebranding...")
        logger.info(self.schwabot_branding['logo_ascii'])
        
        # Create backup
        if not self.create_backup():
            return {'success': False, 'error': 'Failed to create backup'}
        
        # Find files to process
        files_to_process = self.find_files_to_process()
        
        # Process each file
        for file_path in files_to_process:
            self.rebrand_stats['files_processed'] += 1
            self.update_file_content(file_path)
        
        # Rename files
        self.rename_files()
        
        # Update configuration files
        self.update_config_files()
        
        # Create branding files
        self.create_schwabot_branding_files()
        
        # Update test files
        self.update_test_files()
        
        # Update documentation
        self.update_documentation()
        
        # Generate report
        report = self.generate_rebrand_report()
        
        logger.info("✅ Comprehensive rebranding completed!")
        return report
    
    def generate_rebrand_report(self) -> Dict[str, Any]:
        """Generate a comprehensive rebranding report."""
        report = {
            'rebranding_completed': True,
            'timestamp': datetime.now().isoformat(),
            'system_name': self.schwabot_branding['system_name'],
            'version': self.schwabot_branding['version'],
            'statistics': self.rebrand_stats.copy(),
            'backup_location': str(self.backup_dir),
            'next_steps': [
                'Run system tests to verify functionality',
                'Update any remaining manual configurations',
                'Test the visual layer and GUI components',
                'Verify all API endpoints work correctly',
                'Update deployment scripts if needed'
            ]
        }
        
        # Save report
        report_file = self.project_root / 'SCHWABOT_REBRAND_REPORT.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report

def main():
    """Main function to run the rebranding process."""
    rebrander = SchwabotRebrander()
    
    try:
        report = rebrander.run_comprehensive_rebrand()
        
        if report['rebranding_completed']:
            print("\n" + "="*60)
            print("🎉 SCHWABOT REBRANDING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📊 Files processed: {report['statistics']['files_processed']}")
            print(f"✅ Files updated: {report['statistics']['files_updated']}")
            print(f"❌ Errors: {len(report['statistics']['errors'])}")
            print(f"⚠️  Warnings: {len(report['statistics']['warnings'])}")
            print(f"📦 Backup location: {report['backup_location']}")
            print("\n📋 Next Steps:")
            for step in report['next_steps']:
                print(f"   • {step}")
            print("\n🚀 Your system is now fully rebranded as Schwabot AI!")
            
        else:
            print("❌ Rebranding failed!")
            if 'error' in report:
                print(f"Error: {report['error']}")
            
    except Exception as e:
        logger.error(f"❌ Rebranding process failed: {e}")
        print(f"❌ Rebranding failed: {e}")

if __name__ == "__main__":
    main() 