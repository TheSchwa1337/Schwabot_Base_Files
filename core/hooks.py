"""
Enhanced Hook System Integration
===============================

Provides backward compatibility while integrating with the new thermal-aware,
profit-synchronized dynamic hook routing system.
"""

import os
import logging
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Environment variables (preserved for compatibility)
DEBUG_CLUSTERS = os.getenv("DEBUG_CLUSTERS", "0") == "1"
DEBUG_DRIFTS = os.getenv("DEBUG_DRIFTS", "0") == "1"
SIMULATE_STRAT = os.getenv("SIMULATE_STRATEGY", "0") == "1"

# Import the enhanced hook system
try:
    from .enhanced_hooks import DynamicHookRouter, get_hook_router
    ENHANCED_HOOKS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced hooks not available: {e}")
    ENHANCED_HOOKS_AVAILABLE = False

# Import legacy hook components
try:
    from .ncco_manager import NCCOManager
    from .sfsss_router import SFSSSRouter
    from .cluster_mapper import ClusterMapper
    from .drift_shell_engine import DriftShellEngine
    from .ufs_echo_logger import UFSEchoLogger
    from .vault_router import VaultRouter
    LEGACY_HOOKS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Legacy hooks not available: {e}")
    LEGACY_HOOKS_AVAILABLE = False

class HookSystemManager:
    """
    Manages the hook system with enhanced routing capabilities while
    maintaining backward compatibility with legacy hook access patterns.
    """
    
    def __init__(self):
        self._enhanced_router: Optional[DynamicHookRouter] = None
        self._legacy_hooks: Dict[str, Any] = {}
        self._initialized = False
        
        # Initialize the hook system
        self._initialize()
        
    def _initialize(self):
        """Initialize hook system with enhanced routing if available"""
        try:
            if ENHANCED_HOOKS_AVAILABLE:
                # Use enhanced hook router
                self._enhanced_router = get_hook_router()
                
                # Register legacy hooks with enhanced router if available
                if LEGACY_HOOKS_AVAILABLE:
                    self._register_legacy_hooks()
                    
                logger.info("Enhanced hook system initialized successfully")
            elif LEGACY_HOOKS_AVAILABLE:
                # Fallback to legacy initialization
                self._initialize_legacy_hooks()
                logger.info("Legacy hook system initialized")
            else:
                logger.error("No hook system available - neither enhanced nor legacy")
                
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing hook system: {e}")
            
    def _register_legacy_hooks(self):
        """Register legacy hook instances with enhanced router"""
        try:
            legacy_hooks = {
                "ncco_manager": NCCOManager(),
                "sfsss_router": SFSSSRouter(),
                "cluster_mapper": ClusterMapper(),
                "drift_engine": DriftShellEngine(),
                "echo_logger": UFSEchoLogger(),
                "vault_router": VaultRouter()
            }
            
            for hook_id, hook_instance in legacy_hooks.items():
                self._enhanced_router.register_hook(hook_id, hook_instance)
                self._legacy_hooks[hook_id] = hook_instance
                
            logger.info(f"Registered {len(legacy_hooks)} legacy hooks with enhanced router")
            
        except Exception as e:
            logger.error(f"Error registering legacy hooks: {e}")
            
    def _initialize_legacy_hooks(self):
        """Initialize legacy hooks without enhanced routing"""
        try:
            self._legacy_hooks = {
                "ncco_manager": NCCOManager(),
                "sfsss_router": SFSSSRouter(),
                "cluster_mapper": ClusterMapper(),
                "drift_engine": DriftShellEngine(),
                "echo_logger": UFSEchoLogger(),
                "vault_router": VaultRouter()
            }
            
            logger.info(f"Initialized {len(self._legacy_hooks)} legacy hooks")
            
        except Exception as e:
            logger.error(f"Error initializing legacy hooks: {e}")
            
    def execute_hook(self, hook_id: str, method_name: str, *args, **kwargs) -> Any:
        """
        Execute a hook method through enhanced router if available,
        otherwise fallback to direct execution
        """
        if not self._initialized:
            logger.error("Hook system not initialized")
            return None
            
        try:
            if self._enhanced_router:
                # Use enhanced router with thermal/profit awareness
                return self._enhanced_router.execute_hook(hook_id, method_name, *args, **kwargs)
            elif hook_id in self._legacy_hooks:
                # Direct execution for legacy hooks
                hook_instance = self._legacy_hooks[hook_id]
                if hasattr(hook_instance, method_name):
                    return getattr(hook_instance, method_name)(*args, **kwargs)
                else:
                    logger.error(f"Method {method_name} not found on hook {hook_id}")
                    return None
            else:
                logger.error(f"Hook {hook_id} not available")
                return None
                
        except Exception as e:
            logger.error(f"Error executing hook {hook_id}.{method_name}: {e}")
            return None
            
    def get_hook(self, hook_id: str) -> Optional[Any]:
        """Get direct access to a hook instance for legacy compatibility"""
        if self._enhanced_router and hook_id in self._enhanced_router.hook_registry:
            return self._enhanced_router.hook_registry[hook_id]
        elif hook_id in self._legacy_hooks:
            return self._legacy_hooks[hook_id]
        else:
            logger.warning(f"Hook {hook_id} not found")
            return None
            
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hook system statistics"""
        if self._enhanced_router:
            return self._enhanced_router.get_hook_statistics()
        else:
            return {
                "system": "legacy",
                "available_hooks": list(self._legacy_hooks.keys()),
                "enhanced_routing": False
            }
            
    def get_current_context(self):
        """Get current hook execution context"""
        if self._enhanced_router:
            return self._enhanced_router.get_current_context()
        else:
            # Return dummy context for legacy compatibility
            from datetime import datetime, timezone
            return {
                "thermal_zone": "unknown",
                "profit_zone": "unknown",
                "thermal_temp": 65.0,
                "profit_vector_strength": 0.5,
                "memory_confidence": 0.5,
                "timestamp": datetime.now(timezone.utc),
                "legacy_mode": True
            }
            
    def enable_hook(self, hook_id: str) -> bool:
        """Enable a specific hook"""
        if self._enhanced_router:
            return self._enhanced_router.enable_hook(hook_id)
        else:
            logger.info(f"Hook enabling not supported in legacy mode for {hook_id}")
            return hook_id in self._legacy_hooks
            
    def disable_hook(self, hook_id: str) -> bool:
        """Disable a specific hook"""
        if self._enhanced_router:
            return self._enhanced_router.disable_hook(hook_id)
        else:
            logger.info(f"Hook disabling not supported in legacy mode for {hook_id}")
            return False
            
    def shutdown(self):
        """Gracefully shutdown the hook system"""
        if self._enhanced_router:
            self._enhanced_router.shutdown()
        logger.info("Hook system shutdown complete")

# Initialize global hook system manager
_hook_manager = HookSystemManager()

# Legacy compatibility - expose individual hook instances
ncco_manager = _hook_manager.get_hook("ncco_manager")
sfsss_router = _hook_manager.get_hook("sfsss_router")
cluster_mapper = _hook_manager.get_hook("cluster_mapper")
drift_engine = _hook_manager.get_hook("drift_engine")
echo_logger = _hook_manager.get_hook("echo_logger")
vault_router = _hook_manager.get_hook("vault_router")

# Enhanced hook execution functions
def execute_hook(hook_id: str, method_name: str, *args, **kwargs) -> Any:
    """Execute a hook through the hook manager"""
    return _hook_manager.execute_hook(hook_id, method_name, *args, **kwargs)

def get_hook_context():
    """Get current hook execution context"""
    return _hook_manager.get_current_context()

def get_hook_stats() -> Dict[str, Any]:
    """Get hook system statistics"""
    return _hook_manager.get_system_statistics()

def enable_hook(hook_id: str) -> bool:
    """Enable a specific hook"""
    return _hook_manager.enable_hook(hook_id)

def disable_hook(hook_id: str) -> bool:
    """Disable a specific hook"""
    return _hook_manager.disable_hook(hook_id)

def shutdown_hooks():
    """Shutdown the hook system"""
    _hook_manager.shutdown()

# Logging hook execution results for debugging
if DEBUG_CLUSTERS or DEBUG_DRIFTS:
    logger.info("Hook system debug mode enabled")
    logger.info(f"Available hooks: {list(_hook_manager._legacy_hooks.keys()) if _hook_manager._legacy_hooks else 'None'}")
    logger.info(f"Enhanced routing: {_hook_manager._enhanced_router is not None}")

# Example usage logging
if __name__ == "__main__":
    logger.info("Hook System Test")
    logger.info(f"Enhanced hooks available: {ENHANCED_HOOKS_AVAILABLE}")
    logger.info(f"Legacy hooks available: {LEGACY_HOOKS_AVAILABLE}")
    
    # Test hook access
    if ncco_manager:
        logger.info("NCCO Manager available")
    if sfsss_router:
        logger.info("SFSSS Router available")
        
    # Test enhanced features
    context = get_hook_context()
    logger.info(f"Current context: {context}")
    
    stats = get_hook_stats()
    logger.info(f"System stats: {stats}") 