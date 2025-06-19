#!/usr/bin/env python3
"""
Schwabot Core Package - Mathematical Trading System
=================================================

Core package initialization for the Schwabot mathematical trading framework.
Provides unified access to all core components with proper dependency management.

Key Components:
- Mathematical libraries (mathlib, mathlib_v2, mathlib_v3)
- Trading system components (strategy, execution, risk management)
- Real-time processing (tick processor, market data)
- Advanced features (GAN filtering, quantum operations, visualization)
- System management (configuration, monitoring, logging)

Integration Points:
- Windows CLI compatibility with emoji fallbacks
- Flake8 compliance and type annotations
- Performance optimization and error handling
- Comprehensive logging and monitoring

Windows CLI compatible with flake8 compliance.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Any, Optional, List

# Version information
__version__ = "0.46.0-dev"
__author__ = "Schwabot Development Team"
__description__ = "Advanced Mathematical Trading System"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import Windows CLI compatibility handler
try:
    from core.enhanced_windows_cli_compatibility import (
        EnhancedWindowsCliCompatibilityHandler as CLIHandler,
        safe_print, safe_log
    )
    CLI_COMPATIBILITY_AVAILABLE = True
except ImportError:
    CLI_COMPATIBILITY_AVAILABLE = False
    # Fallback CLI handler
    class CLIHandler:
        @staticmethod
        def safe_emoji_print(message: str, force_ascii: bool = False) -> str:
            emoji_mapping = {
                '✅': '[SUCCESS]', '❌': '[ERROR]', '⚠️': '[WARNING]', '🚨': '[ALERT]',
                '🎉': '[COMPLETE]', '🔄': '[PROCESSING]', '⏳': '[WAITING]', '⭐': '[STAR]',
                '🚀': '[LAUNCH]', '🔧': '[TOOLS]', '🛠️': '[REPAIR]', '⚡': '[FAST]',
                '🔍': '[SEARCH]', '🎯': '[TARGET]', '🔥': '[HOT]', '❄️': '[COOL]',
                '📊': '[DATA]', '📈': '[PROFIT]', '📉': '[LOSS]', '💰': '[MONEY]',
                '🧪': '[TEST]', '⚖️': '[BALANCE]', '🌡️': '[TEMP]', '🔬': '[ANALYZE]'
            }
            if force_ascii:
                for emoji, replacement in emoji_mapping.items():
                    message = message.replace(emoji, replacement)
            return message

# Core mathematical libraries
try:
    from mathlib import MathLib as MathLibV1
    from mathlib.mathlib_v2 import MathLibV2
    from core.mathlib_v3 import MathLibV3, Dual
    MATHLIB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Mathematical libraries not fully available: {e}")
    MATHLIB_AVAILABLE = False
    MathLibV1 = None
    MathLibV2 = None
    MathLibV3 = None
    Dual = None

# Core system components
try:
    from core.constraints import ConstraintValidator
    from core.risk_monitor import RiskMonitor
    from core.risk_manager import RiskManager
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Risk management components not available: {e}")
    RISK_MANAGEMENT_AVAILABLE = False

# Trading system components
try:
    from core.strategy_logic import StrategyLogic
    from core.strategy_loader import StrategyLoader
    from core.strategy_execution_mapper import StrategyExecutionMapper
    from core.trade_tensor_router import TradeTensorRouter
    from core.simplified_btc_integration import SimplifiedBTCIntegration
    TRADING_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Trading system components not available: {e}")
    TRADING_SYSTEM_AVAILABLE = False

# Real-time processing components
try:
    from core.tick_processor import TickProcessor
    from core.system_monitor import SystemMonitor
    from core.unified_api_coordinator import UnifiedAPICoordinator
    REALTIME_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Real-time processing components not available: {e}")
    REALTIME_PROCESSING_AVAILABLE = False

# High-performance computing
try:
    from core.rittle_gemm import RittleGEMM
    from core.mathematical_optimization_bridge import MathematicalOptimizationBridge
    HPC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"High-performance computing components not available: {e}")
    HPC_AVAILABLE = False

# Advanced features (will be available after implementation)
try:
    from core.gan_filter import EntropyGAN, GanFilter
    GAN_FILTERING_AVAILABLE = True
except ImportError:
    GAN_FILTERING_AVAILABLE = False

try:
    from core.quantum_mathlib import QuantumMathLib
    QUANTUM_MATH_AVAILABLE = True
except ImportError:
    QUANTUM_MATH_AVAILABLE = False

try:
    from core.visualization import SchwaVisualization
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# System configuration
try:
    from core.config import SchwaConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Export integration orchestrator components
try:
    from core.integration_orchestrator import (
        IntegrationOrchestrator,
        get_integration_orchestrator,
        ComponentStatus,
        IntegrationMode
    )
    INTEGRATION_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    INTEGRATION_ORCHESTRATOR_AVAILABLE = False


class SchwaCore:
    """
    Core system orchestrator for Schwabot framework
    
    This class provides unified access to all core components and manages
    system initialization, configuration, and lifecycle.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize Schwabot core system
        
        Args:
            config: Optional configuration dictionary
        """
        self.version = __version__
        self.config = config or {}
        self.cli_handler = CLIHandler()
        
        # Component availability status
        self.component_status = {
            'mathlib': MATHLIB_AVAILABLE,
            'risk_management': RISK_MANAGEMENT_AVAILABLE,
            'trading_system': TRADING_SYSTEM_AVAILABLE,
            'realtime_processing': REALTIME_PROCESSING_AVAILABLE,
            'hpc': HPC_AVAILABLE,
            'gan_filtering': GAN_FILTERING_AVAILABLE,
            'quantum_math': QUANTUM_MATH_AVAILABLE,
            'visualization': VISUALIZATION_AVAILABLE,
            'config': CONFIG_AVAILABLE,
            'cli_compatibility': CLI_COMPATIBILITY_AVAILABLE,
            'integration_orchestrator': INTEGRATION_ORCHESTRATOR_AVAILABLE
        }
        
        # Initialize components
        self._initialize_components()
        
        # Log initialization
        self.safe_log('info', f"SchwaCore v{self.version} initialized")
    
    def safe_print(self, message: str, force_ascii: Optional[bool] = None) -> None:
        """
        Safe print function with CLI compatibility
        
        Args:
            message: Message to print
            force_ascii: Force ASCII conversion
        """
        if force_ascii is None:
            force_ascii = self.config.get('force_ascii_output', False)
        
        if CLI_COMPATIBILITY_AVAILABLE:
            safe_print(message, force_ascii=force_ascii)
        else:
            safe_message = self.cli_handler.safe_emoji_print(message, force_ascii=force_ascii)
            print(safe_message)
    
    def safe_log(self, level: str, message: str, context: str = "") -> bool:
        """
        Safe logging function with CLI compatibility
        
        Args:
            level: Log level
            message: Message to log
            context: Additional context
            
        Returns:
            True if logging was successful
        """
        if CLI_COMPATIBILITY_AVAILABLE:
            return safe_log(logger, level, message, context)
        else:
            try:
                log_func = getattr(logger, level.lower(), logger.info)
                log_func(message)
                return True
            except Exception:
                return False
    
    def _initialize_components(self) -> None:
        """Initialize available components"""
        try:
            # Initialize mathematical libraries
            if MATHLIB_AVAILABLE:
                self.mathlib_v1 = MathLibV1() if MathLibV1 else None
                self.mathlib_v2 = MathLibV2() if MathLibV2 else None
                self.mathlib_v3 = MathLibV3() if MathLibV3 else None
                self.safe_log('info', 'Mathematical libraries initialized')
            
            # Initialize risk management
            if RISK_MANAGEMENT_AVAILABLE:
                self.risk_monitor = RiskMonitor() if 'RiskMonitor' in globals() else None
                self.risk_manager = RiskManager() if 'RiskManager' in globals() else None
                self.safe_log('info', 'Risk management components initialized')
            
            # Initialize trading system
            if TRADING_SYSTEM_AVAILABLE:
                self.strategy_logic = StrategyLogic() if 'StrategyLogic' in globals() else None
                self.btc_integration = SimplifiedBTCIntegration() if 'SimplifiedBTCIntegration' in globals() else None
                self.safe_log('info', 'Trading system components initialized')
            
            # Initialize real-time processing
            if REALTIME_PROCESSING_AVAILABLE:
                self.tick_processor = TickProcessor() if 'TickProcessor' in globals() else None
                self.system_monitor = SystemMonitor() if 'SystemMonitor' in globals() else None
                self.safe_log('info', 'Real-time processing components initialized')
            
        except Exception as e:
            error_msg = f"Error initializing components: {e}"
            self.safe_log('error', error_msg)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Dictionary containing system status information
        """
        return {
            'version': self.version,
            'components': self.component_status,
            'python_version': sys.version,
            'platform': sys.platform,
            'initialized': True
        }
    
    def validate_system(self) -> Dict[str, Any]:
        """
        Validate system integrity and component availability
        
        Returns:
            Validation results
        """
        try:
            validation_results = {
                'status': 'success',
                'components_available': sum(self.component_status.values()),
                'total_components': len(self.component_status),
                'critical_components': {
                    'mathlib': self.component_status['mathlib'],
                    'trading_system': self.component_status['trading_system'],
                    'cli_compatibility': self.component_status['cli_compatibility']
                },
                'warnings': [],
                'errors': []
            }
            
            # Check critical components
            if not self.component_status['mathlib']:
                validation_results['warnings'].append('Mathematical libraries not fully available')
            
            if not self.component_status['trading_system']:
                validation_results['warnings'].append('Trading system components not fully available')
            
            if not self.component_status['cli_compatibility']:
                validation_results['warnings'].append('CLI compatibility handler not available')
            
            # Calculate availability percentage
            availability_percentage = (validation_results['components_available'] / 
                                     validation_results['total_components']) * 100
            validation_results['availability_percentage'] = availability_percentage
            
            if availability_percentage < 50:
                validation_results['status'] = 'critical'
                validation_results['errors'].append('Less than 50% of components available')
            elif availability_percentage < 80:
                validation_results['status'] = 'warning'
                validation_results['warnings'].append('Less than 80% of components available')
            
            return validation_results
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'components_available': 0,
                'total_components': len(self.component_status)
            }


# Global core instance (singleton pattern)
_core_instance: Optional[SchwaCore] = None


def get_core(config: Optional[Dict[str, Any]] = None) -> SchwaCore:
    """
    Get or create the global core instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        SchwaCore instance
    """
    global _core_instance
    if _core_instance is None:
        _core_instance = SchwaCore(config)
    return _core_instance


def initialize_schwabot(config: Optional[Dict[str, Any]] = None) -> SchwaCore:
    """
    Initialize Schwabot system with full integration orchestrator
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized SchwaCore instance
    """
    # Initialize core
    core = get_core(config)
    
    # Initialize integration orchestrator if available
    if INTEGRATION_ORCHESTRATOR_AVAILABLE:
        try:
            orchestrator = get_integration_orchestrator(core.config_manager if hasattr(core, 'config_manager') else None)
            
            # Start integration
            integration_success = orchestrator.start_integration()
            
            if integration_success:
                core.safe_print("🎛️ Integration Orchestrator started")
                
                # Get system status from orchestrator
                orchestrator_status = orchestrator.get_system_status()
                running_components = orchestrator_status['metrics']['running_components']
                total_components = orchestrator_status['metrics']['total_components']
                
                core.safe_print(f"🔗 Integrated components: {running_components}/{total_components}")
                
                # Store orchestrator reference in core
                core.orchestrator = orchestrator
                
            else:
                core.safe_print("⚠️ Integration Orchestrator failed to start")
                
        except Exception as e:
            core.safe_log('error', f'Error starting integration orchestrator: {e}')
            core.safe_print("⚠️ Integration Orchestrator not available")
    else:
        core.safe_print("⚠️ Integration Orchestrator not available")
    
    # Print initialization status
    status = core.get_system_status()
    core.safe_print(f"🚀 Schwabot v{status['version']} initialized")
    
    # Enhanced component reporting
    if INTEGRATION_ORCHESTRATOR_AVAILABLE and hasattr(core, 'orchestrator'):
        orchestrator_status = core.orchestrator.get_system_status()
        core.safe_print(f"📊 System Status:")
        core.safe_print(f"   Core components: {sum(status['components'].values())}/{len(status['components'])}")
        core.safe_print(f"   Integrated components: {orchestrator_status['metrics']['running_components']}/{orchestrator_status['metrics']['total_components']}")
        core.safe_print(f"   Integration mode: {orchestrator_status['orchestrator']['mode']}")
    else:
        core.safe_print(f"📊 Components available: {sum(status['components'].values())}/{len(status['components'])}")
    
    # Validate system
    validation = core.validate_system()
    if validation['status'] == 'success':
        core.safe_print("✅ System validation successful")
    elif validation['status'] == 'warning':
        core.safe_print("⚠️ System validation completed with warnings")
    else:
        core.safe_print("❌ System validation failed")
    
    return core


# Export key components for easy access
__all__ = [
    # Core classes
    'SchwaCore',
    'get_core',
    'initialize_schwabot',
    
    # Integration orchestrator (if available)
    'IntegrationOrchestrator',
    'get_integration_orchestrator',
    'ComponentStatus',
    'IntegrationMode',
    
    # Mathematical libraries (if available)
    'MathLibV1',
    'MathLibV2', 
    'MathLibV3',
    'Dual',
    
    # Component availability flags
    'MATHLIB_AVAILABLE',
    'RISK_MANAGEMENT_AVAILABLE',
    'TRADING_SYSTEM_AVAILABLE',
    'REALTIME_PROCESSING_AVAILABLE',
    'HPC_AVAILABLE',
    'GAN_FILTERING_AVAILABLE',
    'QUANTUM_MATH_AVAILABLE',
    'VISUALIZATION_AVAILABLE',
    'CONFIG_AVAILABLE',
    'CLI_COMPATIBILITY_AVAILABLE',
    'INTEGRATION_ORCHESTRATOR_AVAILABLE',
    
    # Version information
    '__version__',
    '__author__',
    '__description__'
]


# Initialize logging for the package
logger.info(f"Schwabot Core Package v{__version__} loaded")
