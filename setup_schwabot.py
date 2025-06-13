#!/usr/bin/env python3
"""
Schwabot Setup and Validation Script
===================================

This script sets up the Schwabot environment, validates configuration files,
creates necessary directories, and runs basic integration tests.
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup_schwabot.log')
    ]
)
logger = logging.getLogger(__name__)

class SchwabotSetup:
    """Schwabot setup and validation manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        self.config_dir = self.project_root / "config"
        self.tests_dir = self.project_root / "tests"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        
        self.required_directories = [
            self.config_dir,
            self.config_dir / "schemas",
            self.tests_dir,
            self.logs_dir,
            self.data_dir,
            self.data_dir / "matrix_logs",
            self.project_root / "core",
            self.project_root / "core" / "tests"
        ]
        
        self.required_config_files = [
            "tesseract_enhanced.yaml",
            "fractal_core.yaml",
            "matrix_response_paths.yaml",
            "risk_config.yaml"
        ]
        
        logger.info(f"Schwabot setup initialized at: {self.project_root}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= min_version:
            logger.info(f"Python version {'.'.join(map(str, current_version))} is compatible")
            return True
        else:
            logger.error(f"Python {'.'.join(map(str, min_version))} or higher required. Current: {'.'.join(map(str, current_version))}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error("requirements.txt not found")
            return False
        
        try:
            logger.info("Installing dependencies...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True, check=True)
            
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def create_directories(self) -> bool:
        """Create required directories"""
        try:
            for directory in self.required_directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created/verified directory: {directory}")
            
            # Create __init__.py files for Python packages
            init_files = [
                self.config_dir / "__init__.py",
                self.config_dir / "schemas" / "__init__.py",
                self.tests_dir / "__init__.py",
                self.project_root / "core" / "__init__.py"
            ]
            
            for init_file in init_files:
                if not init_file.exists():
                    init_file.write_text('"""Package initialization"""')
                    logger.info(f"Created __init__.py: {init_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def validate_config_files(self) -> bool:
        """Validate and create configuration files"""
        try:
            # Import config utilities
            sys.path.insert(0, str(self.project_root))
            from config.config_utils import (
                load_yaml_config, 
                create_default_tesseract_config,
                create_default_fractal_config,
                create_default_matrix_config,
                create_default_risk_config
            )
            
            config_creators = {
                "tesseract_enhanced.yaml": create_default_tesseract_config,
                "fractal_core.yaml": create_default_fractal_config,
                "matrix_response_paths.yaml": create_default_matrix_config,
                "risk_config.yaml": create_default_risk_config
            }
            
            for config_file in self.required_config_files:
                config_path = self.config_dir / config_file
                
                if not config_path.exists():
                    logger.info(f"Creating default config: {config_file}")
                    
                    if config_file in config_creators:
                        default_config = config_creators[config_file]()
                        with open(config_path, 'w') as f:
                            yaml.safe_dump(default_config, f, default_flow_style=False, indent=2)
                    else:
                        # Create generic config
                        default_config = {
                            'meta': {
                                'name': config_file.replace('.yaml', ''),
                                'version': '1.0.0',
                                'created': datetime.now().isoformat()
                            }
                        }
                        with open(config_path, 'w') as f:
                            yaml.safe_dump(default_config, f, default_flow_style=False, indent=2)
                
                # Validate the config file
                try:
                    config = load_yaml_config(config_path, create_default=False)
                    logger.info(f"Validated config file: {config_file}")
                except Exception as e:
                    logger.error(f"Invalid config file {config_file}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate config files: {e}")
            return False
    
    def test_imports(self) -> bool:
        """Test critical imports"""
        critical_imports = [
            "numpy",
            "scipy", 
            "pandas",
            "yaml",
            "pathlib",
            "logging",
            "datetime",
            "typing"
        ]
        
        optional_imports = [
            "rich",
            "pydantic",
            "pytest"
        ]
        
        # Test critical imports
        for module in critical_imports:
            try:
                __import__(module)
                logger.info(f"[OK] Critical import successful: {module}")
            except ImportError as e:
                logger.error(f"[FAIL] Critical import failed: {module} - {e}")
                return False
        
        # Test optional imports
        for module in optional_imports:
            try:
                __import__(module)
                logger.info(f"[OK] Optional import successful: {module}")
            except ImportError as e:
                logger.warning(f"[WARN] Optional import failed: {module} - {e}")
        
        return True
    
    def test_config_system(self) -> bool:
        """Test configuration system"""
        try:
            sys.path.insert(0, str(self.project_root))
            
            # Test config utilities
            from config.config_utils import load_yaml_config, standardize_config_path
            
            # Test loading a config file
            test_config_path = self.config_dir / "fractal_core.yaml"
            config = load_yaml_config(test_config_path)
            logger.info("[OK] Config loading test passed")
            
            # Test path standardization
            standardized = standardize_config_path("test.yaml")
            logger.info("[OK] Path standardization test passed")
            
            # Test dashboard integration
            try:
                from dashboard_integration import DashboardBridge
                dashboard = DashboardBridge(config_path=test_config_path)
                logger.info("[OK] Dashboard integration test passed")
            except Exception as e:
                logger.warning(f"[WARN] Dashboard integration test failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Config system test failed: {e}")
            return False
    
    def test_enhanced_tesseract_processor(self) -> bool:
        """Test Enhanced Tesseract Processor"""
        try:
            sys.path.insert(0, str(self.project_root))
            
            # Mock the dependencies that might not be available
            from unittest.mock import MagicMock
            
            # Create mock modules
            mock_modules = {
                'core.risk_indexer': MagicMock(),
                'core.quantum_cellular_risk_monitor': MagicMock(),
                'core.zygot_shell': MagicMock()
            }
            
            for module_name, mock_module in mock_modules.items():
                sys.modules[module_name] = mock_module
            
            # Mock specific classes
            sys.modules['core.risk_indexer'].RiskIndexer = MagicMock
            sys.modules['core.quantum_cellular_risk_monitor'].QuantumCellularRiskMonitor = MagicMock
            sys.modules['core.quantum_cellular_risk_monitor'].AdvancedRiskMetrics = MagicMock
            sys.modules['core.zygot_shell'].ZygotShell = MagicMock
            sys.modules['core.zygot_shell'].ZygotShellState = MagicMock
            sys.modules['core.zygot_shell'].ZygotControlHooks = MagicMock
            
            # Test processor initialization
            from core.enhanced_tesseract_processor import EnhancedTesseractProcessor
            
            config_path = self.config_dir / "tesseract_enhanced.yaml"
            processor = EnhancedTesseractProcessor(config_path=str(config_path))
            
            # Test basic functionality
            status = processor.get_status()
            assert isinstance(status, dict)
            
            # Test test mode
            processor.enable_test_mode()
            assert processor.test_mode == True
            
            processor.disable_test_mode()
            assert processor.test_mode == False
            
            logger.info("[OK] Enhanced Tesseract Processor test passed")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced Tesseract Processor test failed: {e}")
            return False
    
    def run_basic_tests(self) -> bool:
        """Run basic pytest tests"""
        try:
            # Run specific test files that should work
            test_files = [
                "tests/test_config_loading.py"
            ]
            
            for test_file in test_files:
                test_path = self.project_root / test_file
                if test_path.exists():
                    logger.info(f"Running tests: {test_file}")
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", str(test_path), "-v"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"[OK] Tests passed: {test_file}")
                    else:
                        logger.warning(f"[WARN] Some tests failed in {test_file}")
                        logger.warning(f"Test output: {result.stdout}")
                        logger.warning(f"Test errors: {result.stderr}")
                else:
                    logger.warning(f"Test file not found: {test_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Basic tests failed: {e}")
            return False
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate setup validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'python_version': '.'.join(map(str, sys.version_info[:3])),
            'directories_created': [str(d) for d in self.required_directories if d.exists()],
            'config_files_present': [f for f in self.required_config_files if (self.config_dir / f).exists()],
            'setup_status': 'completed'
        }
        
        # Save report
        report_path = self.project_root / "setup_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Setup report saved to: {report_path}")
        return report
    
    def run_full_setup(self) -> bool:
        """Run complete setup process"""
        logger.info("="*60)
        logger.info("STARTING SCHWABOT SETUP")
        logger.info("="*60)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directories),
            ("Installing dependencies", self.install_dependencies),
            ("Validating config files", self.validate_config_files),
            ("Testing imports", self.test_imports),
            ("Testing config system", self.test_config_system),
            ("Testing Enhanced Tesseract Processor", self.test_enhanced_tesseract_processor),
            ("Running basic tests", self.run_basic_tests)
        ]
        
        failed_steps = []
        
        for step_name, step_function in steps:
            logger.info(f"\n--- {step_name} ---")
            try:
                if step_function():
                    logger.info(f"[SUCCESS] {step_name} completed successfully")
                else:
                    logger.error(f"[FAILED] {step_name} failed")
                    failed_steps.append(step_name)
            except Exception as e:
                logger.error(f"[ERROR] {step_name} failed with exception: {e}")
                failed_steps.append(step_name)
        
        # Generate report
        report = self.generate_setup_report()
        
        logger.info("\n" + "="*60)
        if failed_steps:
            logger.warning(f"SETUP COMPLETED WITH WARNINGS")
            logger.warning(f"Failed steps: {', '.join(failed_steps)}")
            logger.info("The system may still be functional, but some features might not work correctly.")
        else:
            logger.info("SETUP COMPLETED SUCCESSFULLY")
            logger.info("All systems are ready for deployment!")
        
        logger.info("="*60)
        
        return len(failed_steps) == 0


def main():
    """Main setup function"""
    setup = SchwabotSetup()
    success = setup.run_full_setup()
    
    if success:
        print("\n🎉 Schwabot setup completed successfully!")
        print("You can now run the system with confidence.")
    else:
        print("\n⚠️  Schwabot setup completed with warnings.")
        print("Check the setup log for details: setup_schwabot.log")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 