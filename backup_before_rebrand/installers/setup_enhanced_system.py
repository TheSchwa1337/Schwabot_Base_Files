#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Schwabot Trading System Setup Script
=============================================

This script helps users set up the enhanced Schwabot trading system with:
- API key configuration
- GPU acceleration setup
- Environment configuration
- System validation
- Performance optimization

Usage:
    python setup_enhanced_system.py [--interactive] [--config-file CONFIG_FILE]
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
import platform
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedSystemSetup:
    """Enhanced Schwabot trading system setup manager."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the setup manager."""
        self.config_file = config_file or "config/enhanced_trading_config.yaml"
        self.env_template = "config/enhanced.env.template"
        self.env_file = ".env"
        self.project_root = Path(__file__).parent
        self.config = {}
        self.env_vars = {}
        
    def run_setup(self, interactive: bool = True) -> bool:
        """Run the complete setup process."""
        try:
            logger.info("🚀 Starting Enhanced Schwabot Trading System Setup")
            logger.info("=" * 60)
            
            # Step 1: System requirements check
            if not self._check_system_requirements():
                logger.error("❌ System requirements not met")
                return False
            
            # Step 2: Load configuration
            if not self._load_configuration():
                logger.error("❌ Failed to load configuration")
                return False
            
            # Step 3: Interactive setup (if requested)
            if interactive:
                if not self._interactive_setup():
                    logger.error("❌ Interactive setup failed")
                    return False
            
            # Step 4: Create environment file
            if not self._create_environment_file():
                logger.error("❌ Failed to create environment file")
                return False
            
            # Step 5: Install dependencies
            if not self._install_dependencies():
                logger.error("❌ Failed to install dependencies")
                return False
            
            # Step 6: GPU setup
            if not self._setup_gpu_acceleration():
                logger.error("❌ GPU setup failed")
                return False
            
            # Step 7: Validate configuration
            if not self._validate_configuration():
                logger.error("❌ Configuration validation failed")
                return False
            
            # Step 8: Create directories
            if not self._create_directories():
                logger.error("❌ Failed to create directories")
                return False
            
            # Step 9: Test system
            if not self._test_system():
                logger.error("❌ System test failed")
                return False
            
            logger.info("✅ Enhanced Schwabot Trading System Setup Complete!")
            logger.info("🎯 Next steps:")
            logger.info("   1. Edit .env file with your API keys")
            logger.info("   2. Run: python main.py --system-status")
            logger.info("   3. Run: python main.py --test-api-connections")
            logger.info("   4. Start trading: python main.py --start-trading")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def _check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check operating system
        os_name = platform.system()
        logger.info(f"✅ Operating System: {os_name}")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 4:
                logger.warning(f"⚠️  Low memory: {memory_gb:.1f} GB (4+ GB recommended)")
            else:
                logger.info(f"✅ Memory: {memory_gb:.1f} GB")
        except ImportError:
            logger.warning("⚠️  psutil not available, skipping memory check")
        
        # Check disk space
        try:
            disk = psutil.disk_usage('.')
            disk_gb = disk.free / (1024**3)
            if disk_gb < 10:
                logger.warning(f"⚠️  Low disk space: {disk_gb:.1f} GB (10+ GB recommended)")
            else:
                logger.info(f"✅ Disk space: {disk_gb:.1f} GB available")
        except ImportError:
            logger.warning("⚠️  psutil not available, skipping disk check")
        
        return True
    
    def _load_configuration(self) -> bool:
        """Load configuration from YAML file."""
        try:
            config_path = self.project_root / self.config_file
            if not config_path.exists():
                logger.error(f"❌ Configuration file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"✅ Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            return False
    
    def _interactive_setup(self) -> bool:
        """Run interactive setup process."""
        logger.info("🎯 Starting interactive setup...")
        
        try:
            # API Keys setup
            self._setup_api_keys()
            
            # Trading parameters
            self._setup_trading_parameters()
            
            # GPU configuration
            self._setup_gpu_configuration()
            
            # Strategy configuration
            self._setup_strategy_configuration()
            
            # Monitoring setup
            self._setup_monitoring()
            
            logger.info("✅ Interactive setup completed")
            return True
            
        except KeyboardInterrupt:
            logger.info("⚠️  Setup interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Interactive setup failed: {e}")
            return False
    
    def _setup_api_keys(self):
        """Setup API keys interactively."""
        logger.info("\n🔑 API Keys Setup")
        logger.info("-" * 30)
        
        # Market Data APIs
        logger.info("Market Data APIs:")
        
        # CoinMarketCap
        cmc_key = input("CoinMarketCap API Key (optional, press Enter to skip): ").strip()
        if cmc_key:
            self.env_vars['COINMARKETCAP_API_KEY'] = cmc_key
            logger.info("✅ CoinMarketCap API key configured")
        
        # Alpha Vantage
        av_key = input("Alpha Vantage API Key (optional, press Enter to skip): ").strip()
        if av_key:
            self.env_vars['ALPHA_VANTAGE_API_KEY'] = av_key
            logger.info("✅ Alpha Vantage API key configured")
        
        # Glassnode
        gn_key = input("Glassnode API Key (optional, press Enter to skip): ").strip()
        if gn_key:
            self.env_vars['GLASSNODE_API_KEY'] = gn_key
            logger.info("✅ Glassnode API key configured")
        
        # Exchange APIs
        logger.info("\nExchange APIs:")
        
        # Binance
        use_binance = input("Use Binance exchange? (y/n): ").strip().lower()
        if use_binance == 'y':
            binance_key = input("Binance API Key: ").strip()
            binance_secret = input("Binance Secret Key: ").strip()
            if binance_key and binance_secret:
                self.env_vars['BINANCE_API_KEY'] = binance_key
                self.env_vars['BINANCE_SECRET_KEY'] = binance_secret
                self.env_vars['BINANCE_TESTNET'] = 'true'
                logger.info("✅ Binance API configured (testnet mode)")
        
        # Coinbase Pro
        use_coinbase = input("Use Coinbase Pro exchange? (y/n): ").strip().lower()
        if use_coinbase == 'y':
            coinbase_key = input("Coinbase API Key: ").strip()
            coinbase_secret = input("Coinbase Secret Key: ").strip()
            coinbase_passphrase = input("Coinbase Passphrase: ").strip()
            if coinbase_key and coinbase_secret and coinbase_passphrase:
                self.env_vars['COINBASE_API_KEY'] = coinbase_key
                self.env_vars['COINBASE_SECRET_KEY'] = coinbase_secret
                self.env_vars['COINBASE_PASSPHRASE'] = coinbase_passphrase
                self.env_vars['COINBASE_SANDBOX'] = 'true'
                logger.info("✅ Coinbase Pro API configured (sandbox mode)")
        
        # Kraken
        use_kraken = input("Use Kraken exchange? (y/n): ").strip().lower()
        if use_kraken == 'y':
            kraken_key = input("Kraken API Key: ").strip()
            kraken_secret = input("Kraken Secret Key: ").strip()
            if kraken_key and kraken_secret:
                self.env_vars['KRAKEN_API_KEY'] = kraken_key
                self.env_vars['KRAKEN_SECRET_KEY'] = kraken_secret
                self.env_vars['KRAKEN_SANDBOX'] = 'true'
                logger.info("✅ Kraken API configured (sandbox mode)")
    
    def _setup_trading_parameters(self):
        """Setup trading parameters interactively."""
        logger.info("\n📊 Trading Parameters Setup")
        logger.info("-" * 30)
        
        # Trading mode
        trading_mode = input("Trading mode (sandbox/live/backtest) [sandbox]: ").strip() or "sandbox"
        self.env_vars['SCHWABOT_TRADING_MODE'] = trading_mode
        
        # Risk management
        max_position = input("Max position size % [10.0]: ").strip() or "10.0"
        self.env_vars['SCHWABOT_MAX_POSITION_SIZE_PCT'] = max_position
        
        max_exposure = input("Max total exposure % [30.0]: ").strip() or "30.0"
        self.env_vars['SCHWABOT_MAX_TOTAL_EXPOSURE_PCT'] = max_exposure
        
        stop_loss = input("Stop loss % [2.0]: ").strip() or "2.0"
        self.env_vars['SCHWABOT_STOP_LOSS_PCT'] = stop_loss
        
        take_profit = input("Take profit % [5.0]: ").strip() or "5.0"
        self.env_vars['SCHWABOT_TAKE_PROFIT_PCT'] = take_profit
        
        logger.info("✅ Trading parameters configured")
    
    def _setup_gpu_configuration(self):
        """Setup GPU configuration interactively."""
        logger.info("\n🚀 GPU Acceleration Setup")
        logger.info("-" * 30)
        
        # Check for CUDA
        cuda_available = self._check_cuda_availability()
        
        if cuda_available:
            enable_gpu = input("Enable GPU acceleration? (y/n) [y]: ").strip().lower() or "y"
            if enable_gpu == 'y':
                self.env_vars['SCHWABOT_ENABLE_GPU_ACCELERATION'] = 'true'
                
                # GPU device selection
                device_id = input("GPU device ID [0]: ").strip() or "0"
                self.env_vars['SCHWABOT_CUDA_DEVICE_ID'] = device_id
                
                # Memory limit
                memory_limit = input("GPU memory limit (GB) [4.0]: ").strip() or "4.0"
                self.env_vars['SCHWABOT_GPU_MEMORY_LIMIT_GB'] = memory_limit
                
                logger.info("✅ GPU acceleration configured")
            else:
                self.env_vars['SCHWABOT_ENABLE_GPU_ACCELERATION'] = 'false'
                logger.info("✅ GPU acceleration disabled")
        else:
            logger.info("⚠️  CUDA not available, GPU acceleration disabled")
            self.env_vars['SCHWABOT_ENABLE_GPU_ACCELERATION'] = 'false'
    
    def _setup_strategy_configuration(self):
        """Setup strategy configuration interactively."""
        logger.info("\n🎯 Strategy Configuration Setup")
        logger.info("-" * 30)
        
        # Strategy selection
        logger.info("Available strategies:")
        strategies = [
            "volume_weighted_hash_oscillator",
            "zygot_zalgo_entropy_dual_key_gate", 
            "multi_phase_strategy_weight_tensor",
            "quantum_strategy_calculator",
            "entropy_enhanced_trading_executor"
        ]
        
        for i, strategy in enumerate(strategies, 1):
            logger.info(f"  {i}. {strategy}")
        
        selected = input("Select strategies to enable (comma-separated, e.g., 1,2,3) [all]: ").strip()
        if not selected or selected.lower() == 'all':
            selected_strategies = strategies
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selected.split(',')]
                selected_strategies = [strategies[i] for i in indices if 0 <= i < len(strategies)]
            except (ValueError, IndexError):
                selected_strategies = strategies
        
        self.env_vars['SCHWABOT_ACTIVE_STRATEGIES'] = ','.join(selected_strategies)
        
        # Confidence threshold
        confidence = input("Confidence threshold [0.7]: ").strip() or "0.7"
        self.env_vars['SCHWABOT_CONFIDENCE_THRESHOLD'] = confidence
        
        logger.info("✅ Strategy configuration completed")
    
    def _setup_monitoring(self):
        """Setup monitoring configuration interactively."""
        logger.info("\n📈 Monitoring Setup")
        logger.info("-" * 30)
        
        # Logging level
        log_level = input("Log level (DEBUG/INFO/WARNING/ERROR) [INFO]: ").strip() or "INFO"
        self.env_vars['SCHWABOT_LOG_LEVEL'] = log_level
        
        # Enable alerts
        enable_alerts = input("Enable alerts? (y/n) [y]: ").strip().lower() or "y"
        if enable_alerts == 'y':
            self.env_vars['SCHWABOT_ENABLE_ALERTS'] = 'true'
            
            # Alert channels
            channels = input("Alert channels (console,file,email,slack) [console,file]: ").strip() or "console,file"
            self.env_vars['SCHWABOT_ALERT_CHANNELS'] = channels
        else:
            self.env_vars['SCHWABOT_ENABLE_ALERTS'] = 'false'
        
        # Performance tracking
        enable_performance = input("Enable performance tracking? (y/n) [y]: ").strip().lower() or "y"
        self.env_vars['SCHWABOT_ENABLE_PERFORMANCE_TRACKING'] = 'true' if enable_performance == 'y' else 'false'
        
        logger.info("✅ Monitoring configuration completed")
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _create_environment_file(self) -> bool:
        """Create environment file from template and user input."""
        try:
            template_path = self.project_root / self.env_template
            env_path = self.project_root / self.env_file
            
            if not template_path.exists():
                logger.error(f"❌ Environment template not found: {template_path}")
                return False
            
            # Read template
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Replace placeholders with user values
            for key, value in self.env_vars.items():
                placeholder = f"your_{key.lower()}_here"
                template_content = template_content.replace(placeholder, value)
            
            # Write environment file
            with open(env_path, 'w') as f:
                f.write(template_content)
            
            logger.info(f"✅ Environment file created: {env_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create environment file: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Install required dependencies."""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Check if requirements.txt exists
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                logger.info("Installing from requirements.txt...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                             check=True, capture_output=True)
                logger.info("✅ Dependencies installed from requirements.txt")
            else:
                logger.info("Installing core dependencies...")
                
                # Core dependencies
                core_deps = [
                    "numpy>=1.21.0",
                    "pandas>=1.3.0",
                    "aiohttp>=3.8.0",
                    "pyyaml>=6.0",
                    "python-dotenv>=0.19.0",
                    "ccxt>=2.0.0",
                    "asyncio-mqtt>=0.11.0",
                    "websockets>=10.0",
                    "psutil>=5.8.0"
                ]
                
                for dep in core_deps:
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                 check=True, capture_output=True)
                
                # GPU dependencies (optional)
                if self.env_vars.get('SCHWABOT_ENABLE_GPU_ACCELERATION') == 'true':
                    logger.info("Installing GPU dependencies...")
                    gpu_deps = [
                        "cupy-cuda11x>=11.0.0",  # Adjust based on CUDA version
                        "numba>=0.56.0"
                    ]
                    
                    for dep in gpu_deps:
                        try:
                            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                         check=True, capture_output=True)
                        except subprocess.CalledProcessError:
                            logger.warning(f"⚠️  Failed to install {dep}, GPU acceleration may not work")
                
                logger.info("✅ Core dependencies installed")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Dependency installation error: {e}")
            return False
    
    def _setup_gpu_acceleration(self) -> bool:
        """Setup GPU acceleration."""
        if self.env_vars.get('SCHWABOT_ENABLE_GPU_ACCELERATION') != 'true':
            logger.info("⏭️  GPU acceleration disabled, skipping setup")
            return True
        
        logger.info("🚀 Setting up GPU acceleration...")
        
        try:
            # Test CUDA installation
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ nvidia-smi not available, GPU setup failed")
                return False
            
            logger.info("✅ CUDA detected")
            
            # Test CuPy installation
            try:
                import cupy as cp
                logger.info(f"✅ CuPy version: {cp.__version__}")
                
                # Test GPU memory
                device_id = int(self.env_vars.get('SCHWABOT_CUDA_DEVICE_ID', '0'))
                cp.cuda.Device(device_id).use()
                
                # Allocate test memory
                test_array = cp.zeros((1000, 1000), dtype=cp.float32)
                del test_array
                
                logger.info("✅ GPU memory test passed")
                
            except ImportError:
                logger.error("❌ CuPy not installed, GPU acceleration will not work")
                return False
            except Exception as e:
                logger.error(f"❌ GPU test failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU setup failed: {e}")
            return False
    
    def _validate_configuration(self) -> bool:
        """Validate the configuration."""
        logger.info("🔍 Validating configuration...")
        
        try:
            # Check environment file
            env_path = self.project_root / self.env_file
            if not env_path.exists():
                logger.error("❌ Environment file not found")
                return False
            
            # Check required directories
            required_dirs = ['logs', 'data', 'backups', 'config']
            for dir_name in required_dirs:
                dir_path = self.project_root / dir_name
                if not dir_path.exists():
                    logger.warning(f"⚠️  Directory not found: {dir_name}")
            
            # Validate API keys (basic check)
            with open(env_path, 'r') as f:
                env_content = f.read()
            
            if 'your_coinmarketcap_api_key_here' in env_content:
                logger.warning("⚠️  CoinMarketCap API key not configured")
            
            if 'your_alpha_vantage_api_key_here' in env_content:
                logger.warning("⚠️  Alpha Vantage API key not configured")
            
            logger.info("✅ Configuration validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def _create_directories(self) -> bool:
        """Create necessary directories."""
        logger.info("📁 Creating directories...")
        
        try:
            directories = [
                'logs',
                'data',
                'backups',
                'config',
                'test_data',
                'reports'
            ]
            
            for dir_name in directories:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True)
                logger.info(f"✅ Created directory: {dir_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def _test_system(self) -> bool:
        """Test the system setup."""
        logger.info("🧪 Testing system setup...")
        
        try:
            # Test imports
            test_imports = [
                'numpy',
                'pandas',
                'aiohttp',
                'yaml',
                'asyncio'
            ]
            
            for module in test_imports:
                try:
                    __import__(module)
                    logger.info(f"✅ {module} import successful")
                except ImportError:
                    logger.error(f"❌ {module} import failed")
                    return False
            
            # Test GPU if enabled
            if self.env_vars.get('SCHWABOT_ENABLE_GPU_ACCELERATION') == 'true':
                try:
                    import cupy as cp
                    logger.info("✅ GPU import successful")
                except ImportError:
                    logger.warning("⚠️  GPU import failed, but continuing")
            
            # Test configuration loading
            try:
                from dotenv import load_dotenv
                load_dotenv(self.project_root / self.env_file)
                logger.info("✅ Environment variables loaded")
            except ImportError:
                logger.warning("⚠️  python-dotenv not available")
            
            logger.info("✅ System test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ System test failed: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Schwabot Trading System Setup")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Run interactive setup")
    parser.add_argument("--config-file", type=str, default="config/enhanced_trading_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Run non-interactive setup")
    
    args = parser.parse_args()
    
    # Override interactive flag
    if args.non_interactive:
        args.interactive = False
    
    # Run setup
    setup = EnhancedSystemSetup(args.config_file)
    success = setup.run_setup(interactive=args.interactive)
    
    if success:
        logger.info("🎉 Setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 