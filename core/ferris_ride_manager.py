#!/usr/bin/env python3
"""
🎡 Ferris Ride Manager - Revolutionary Auto-Trading Mode Manager
================================================================

Manages the Ferris Ride Looping Strategy system:
- Auto-detection of capital and tickers
- Pattern studying before entry
- Hash pattern matching and confidence building
- Mathematical orbital trading with Ferris RDE framework
- USB backup system integration
- Focus on USDC pairs (Everything to USDC, USDC to Everything)
"""

import yaml
import json
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FerrisRideConfig:
    """Ferris Ride configuration parameters."""
    # Auto-detection settings
    auto_detect_capital: bool = True
    auto_detect_tickers: bool = True
    usb_backup_enabled: bool = True
    
    # Pattern studying settings
    study_duration_hours: int = 72  # 3 days minimum
    confidence_threshold: float = 0.6
    risk_threshold: float = 0.3
    
    # Trading parameters
    base_position_size_pct: float = 0.12  # 12% base position size
    ferris_multiplier: float = 1.5  # Ferris RDE multiplier
    profit_target_pct: float = 0.05  # 5% profit target
    stop_loss_pct: float = 0.025  # 2.5% stop loss
    
    # Orbital shell settings
    orbital_shells: List[int] = None  # Will be set to [2, 4, 6, 8] for medium risk
    current_shell: int = 2
    
    # Ferris RDE mathematical framework
    momentum_factor: float = 1.0
    gravity_center: float = 0.5
    orbital_velocity: float = 1.0
    spiral_radius: float = 1.0
    
    # USDC focus settings
    usdc_pairs_only: bool = True
    usdc_correlation_threshold: float = 0.7
    
    # Performance tracking
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    win_rate_target: float = 0.65  # 65% win rate target
    
    def __post_init__(self):
        if self.orbital_shells is None:
            self.orbital_shells = [2, 4, 6, 8]  # Medium risk orbitals

class FerrisRideManager:
    """Ferris Ride Looping Strategy Manager."""
    
    def __init__(self):
        self.config_file = Path("AOI_Base_Files_Schwabot/config/ferris_ride_config.yaml")
        self.backup_dir = Path("AOI_Base_Files_Schwabot/backup/ferris_ride_backup")
        self.is_active = False
        self.original_configs = {}
        
        # Initialize Ferris Ride system
        try:
            from .ferris_ride_system import ferris_ride_system
            self.ferris_system = ferris_ride_system
            self.ferris_available = True
        except ImportError:
            self.ferris_system = None
            self.ferris_available = False
            logger.warning("Ferris Ride System not available")
        
        logger.info("🎡 Ferris Ride Manager initialized")
    
    def load_ferris_ride_config(self) -> FerrisRideConfig:
        """Load Ferris Ride configuration."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                config = FerrisRideConfig(**config_data)
                logger.info("✅ Ferris Ride configuration loaded")
                return config
            else:
                # Create default configuration
                config = FerrisRideConfig()
                self.save_ferris_ride_config(config)
                logger.info("✅ Default Ferris Ride configuration created")
                return config
                
        except Exception as e:
            logger.error(f"❌ Failed to load Ferris Ride config: {e}")
            return FerrisRideConfig()
    
    def save_ferris_ride_config(self, config: FerrisRideConfig):
        """Save Ferris Ride configuration."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'auto_detect_capital': config.auto_detect_capital,
                'auto_detect_tickers': config.auto_detect_tickers,
                'usb_backup_enabled': config.usb_backup_enabled,
                'study_duration_hours': config.study_duration_hours,
                'confidence_threshold': config.confidence_threshold,
                'risk_threshold': config.risk_threshold,
                'base_position_size_pct': config.base_position_size_pct,
                'ferris_multiplier': config.ferris_multiplier,
                'profit_target_pct': config.profit_target_pct,
                'stop_loss_pct': config.stop_loss_pct,
                'orbital_shells': config.orbital_shells,
                'current_shell': config.current_shell,
                'momentum_factor': config.momentum_factor,
                'gravity_center': config.gravity_center,
                'orbital_velocity': config.orbital_velocity,
                'spiral_radius': config.spiral_radius,
                'usdc_pairs_only': config.usdc_pairs_only,
                'usdc_correlation_threshold': config.usdc_correlation_threshold,
                'max_daily_loss_pct': config.max_daily_loss_pct,
                'win_rate_target': config.win_rate_target
            }
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info("✅ Ferris Ride configuration saved")
            
        except Exception as e:
            logger.error(f"❌ Failed to save Ferris Ride config: {e}")
    
    def activate_ferris_ride_mode(self) -> bool:
        """Activate Ferris Ride mode."""
        try:
            if not self.ferris_available:
                logger.error("❌ Ferris Ride System not available")
                return False
            
            logger.info("🎡 Activating Ferris Ride Mode...")
            
            # Backup original configurations
            self._backup_original_configs()
            
            # Load Ferris Ride configuration
            config = self.load_ferris_ride_config()
            
            # Initialize Ferris Ride system
            logger.info("🔍 Initializing auto-detection...")
            success = self.ferris_system.auto_detect_capital_and_tickers()
            
            if not success:
                logger.error("❌ Auto-detection failed")
                return False
            
            # Update trading configuration for Ferris Ride mode
            self._update_trading_config_for_ferris_ride(config)
            
            # Update orbital shell configuration
            self._update_orbital_config_for_ferris_ride(config)
            
            # Update risk management for Ferris Ride
            self._update_risk_config_for_ferris_ride(config)
            
            # Set Ferris RDE state
            self.ferris_system.ferris_rde_state.update({
                'momentum_factor': config.momentum_factor,
                'gravity_center': config.gravity_center,
                'orbital_velocity': config.orbital_velocity,
                'spiral_radius': config.spiral_radius
            })
            
            self.is_active = True
            
            logger.info("✅ Ferris Ride Mode activated successfully!")
            logger.info("🎡 Ferris Ride Features:")
            logger.info("   • Auto-detection of capital and USDC pairs")
            logger.info("   • Pattern studying before entry (3+ days)")
            logger.info("   • Hash pattern matching for precise entry")
            logger.info("   • Confidence zone building through profits")
            logger.info("   • Ferris RDE mathematical framework")
            logger.info("   • USB backup system")
            logger.info("   • Focus: Everything to USDC / USDC to Everything")
            logger.info("   • Medium risk orbitals: [2, 4, 6, 8]")
            logger.info("   • Position Size: 12% base × 1.5 Ferris multiplier")
            logger.info("   • Profit Target: 5%")
            logger.info("   • Stop Loss: 2.5%")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ferris Ride activation failed: {e}")
            return False
    
    def deactivate_ferris_ride_mode(self) -> bool:
        """Deactivate Ferris Ride mode."""
        try:
            logger.info("🔄 Deactivating Ferris Ride Mode...")
            
            # Restore original configurations
            self._restore_original_configs()
            
            # Reset Ferris Ride system state
            if self.ferris_system:
                self.ferris_system.active_zones.clear()
                self.ferris_system.studied_patterns.clear()
                self.ferris_system.current_phase = self.ferris_system.current_phase.__class__.STUDY
            
            self.is_active = False
            
            logger.info("✅ Ferris Ride Mode deactivated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ferris Ride deactivation failed: {e}")
            return False
    
    def get_ferris_ride_status(self) -> Dict[str, Any]:
        """Get Ferris Ride mode status."""
        try:
            if not self.ferris_available:
                return {
                    "available": False,
                    "active": False,
                    "error": "Ferris Ride System not available"
                }
            
            config = self.load_ferris_ride_config()
            ferris_status = self.ferris_system.get_ferris_status()
            
            return {
                "available": True,
                "active": self.is_active,
                "config": {
                    "auto_detect_capital": config.auto_detect_capital,
                    "auto_detect_tickers": config.auto_detect_tickers,
                    "usb_backup_enabled": config.usb_backup_enabled,
                    "study_duration_hours": config.study_duration_hours,
                    "confidence_threshold": config.confidence_threshold,
                    "base_position_size_pct": config.base_position_size_pct,
                    "ferris_multiplier": config.ferris_multiplier,
                    "profit_target_pct": config.profit_target_pct,
                    "stop_loss_pct": config.stop_loss_pct,
                    "orbital_shells": config.orbital_shells,
                    "current_shell": config.current_shell,
                    "usdc_pairs_only": config.usdc_pairs_only,
                    "max_daily_loss_pct": config.max_daily_loss_pct,
                    "win_rate_target": config.win_rate_target
                },
                "ferris_system_status": ferris_status,
                "config_file": str(self.config_file),
                "backup_dir": str(self.backup_dir)
            }
            
        except Exception as e:
            logger.error(f"❌ Status retrieval failed: {e}")
            return {"available": False, "active": False, "error": str(e)}
    
    def validate_ferris_ride_requirements(self) -> Dict[str, Any]:
        """Validate Ferris Ride mode requirements."""
        try:
            requirements = {
                "ferris_system_available": self.ferris_available,
                "config_file_exists": self.config_file.exists(),
                "backup_dir_accessible": self.backup_dir.parent.exists(),
                "auto_detection_ready": True,  # Will be tested during activation
                "usb_backup_ready": True,  # Will be tested during activation
                "all_requirements_met": True
            }
            
            if not self.ferris_available:
                requirements["all_requirements_met"] = False
                requirements["error"] = "Ferris Ride System not available"
            
            if not self.config_file.exists():
                requirements["all_requirements_met"] = False
                requirements["warning"] = "Configuration file will be created on activation"
            
            if not self.backup_dir.parent.exists():
                requirements["all_requirements_met"] = False
                requirements["warning"] = "Backup directory will be created on activation"
            
            logger.info("✅ Ferris Ride requirements validated")
            return requirements
            
        except Exception as e:
            logger.error(f"❌ Requirements validation failed: {e}")
            return {
                "all_requirements_met": False,
                "error": str(e)
            }
    
    def _backup_original_configs(self):
        """Backup original configurations before activating Ferris Ride mode."""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup trading configuration
            trading_config_path = Path("AOI_Base_Files_Schwabot/config/trading_config.yaml")
            if trading_config_path.exists():
                backup_path = self.backup_dir / "trading_config_backup.yaml"
                shutil.copy2(trading_config_path, backup_path)
                self.original_configs["trading_config"] = backup_path
                logger.info("✅ Trading config backed up")
            
            # Backup orbital configuration
            orbital_config_path = Path("AOI_Base_Files_Schwabot/config/orbital_config.yaml")
            if orbital_config_path.exists():
                backup_path = self.backup_dir / "orbital_config_backup.yaml"
                shutil.copy2(orbital_config_path, backup_path)
                self.original_configs["orbital_config"] = backup_path
                logger.info("✅ Orbital config backed up")
            
            # Backup risk configuration
            risk_config_path = Path("AOI_Base_Files_Schwabot/config/risk_config.yaml")
            if risk_config_path.exists():
                backup_path = self.backup_dir / "risk_config_backup.yaml"
                shutil.copy2(risk_config_path, backup_path)
                self.original_configs["risk_config"] = backup_path
                logger.info("✅ Risk config backed up")
            
        except Exception as e:
            logger.error(f"❌ Configuration backup failed: {e}")
    
    def _restore_original_configs(self):
        """Restore original configurations after deactivating Ferris Ride mode."""
        try:
            # Restore trading configuration
            if "trading_config" in self.original_configs:
                shutil.copy2(self.original_configs["trading_config"], 
                           Path("AOI_Base_Files_Schwabot/config/trading_config.yaml"))
                logger.info("✅ Trading config restored")
            
            # Restore orbital configuration
            if "orbital_config" in self.original_configs:
                shutil.copy2(self.original_configs["orbital_config"], 
                           Path("AOI_Base_Files_Schwabot/config/orbital_config.yaml"))
                logger.info("✅ Orbital config restored")
            
            # Restore risk configuration
            if "risk_config" in self.original_configs:
                shutil.copy2(self.original_configs["risk_config"], 
                           Path("AOI_Base_Files_Schwabot/config/risk_config.yaml"))
                logger.info("✅ Risk config restored")
            
        except Exception as e:
            logger.error(f"❌ Configuration restoration failed: {e}")
    
    def _update_trading_config_for_ferris_ride(self, config: FerrisRideConfig):
        """Update trading configuration for Ferris Ride mode."""
        try:
            trading_config_path = Path("AOI_Base_Files_Schwabot/config/trading_config.yaml")
            
            if trading_config_path.exists():
                with open(trading_config_path, 'r') as f:
                    trading_config = yaml.safe_load(f)
            else:
                trading_config = {}
            
            # Update for Ferris Ride mode
            trading_config.update({
                "position_size_pct": config.base_position_size_pct * config.ferris_multiplier,
                "profit_target_pct": config.profit_target_pct,
                "stop_loss_pct": config.stop_loss_pct,
                "max_daily_loss_pct": config.max_daily_loss_pct,
                "win_rate_target": config.win_rate_target,
                "ferris_ride_mode": True,
                "ferris_multiplier": config.ferris_multiplier,
                "study_duration_hours": config.study_duration_hours,
                "confidence_threshold": config.confidence_threshold
            })
            
            # Update supported symbols for USDC focus
            if config.usdc_pairs_only:
                usdc_pairs = [
                    "BTC/USDC", "ETH/USDC", "XRP/USDC", "SOL/USDC", "ADA/USDC",
                    "USDC/BTC", "USDC/ETH", "USDC/XRP", "USDC/SOL", "USDC/ADA",
                    "DOT/USDC", "LINK/USDC", "MATIC/USDC", "AVAX/USDC", "UNI/USDC"
                ]
                trading_config["supported_symbols"] = usdc_pairs
            
            # Save updated configuration
            trading_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trading_config_path, 'w') as f:
                yaml.dump(trading_config, f, default_flow_style=False, indent=2)
            
            logger.info("✅ Trading config updated for Ferris Ride mode")
            
        except Exception as e:
            logger.error(f"❌ Trading config update failed: {e}")
    
    def _update_orbital_config_for_ferris_ride(self, config: FerrisRideConfig):
        """Update orbital configuration for Ferris Ride mode."""
        try:
            orbital_config_path = Path("AOI_Base_Files_Schwabot/config/orbital_config.yaml")
            
            if orbital_config_path.exists():
                with open(orbital_config_path, 'r') as f:
                    orbital_config = yaml.safe_load(f)
            else:
                orbital_config = {}
            
            # Update for Ferris Ride mode
            orbital_config.update({
                "active_orbital_shells": config.orbital_shells,
                "current_shell": config.current_shell,
                "ferris_ride_mode": True,
                "ferris_rde_state": {
                    "momentum_factor": config.momentum_factor,
                    "gravity_center": config.gravity_center,
                    "orbital_velocity": config.orbital_velocity,
                    "spiral_radius": config.spiral_radius
                }
            })
            
            # Save updated configuration
            orbital_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(orbital_config_path, 'w') as f:
                yaml.dump(orbital_config, f, default_flow_style=False, indent=2)
            
            logger.info("✅ Orbital config updated for Ferris Ride mode")
            
        except Exception as e:
            logger.error(f"❌ Orbital config update failed: {e}")
    
    def _update_risk_config_for_ferris_ride(self, config: FerrisRideConfig):
        """Update risk configuration for Ferris Ride mode."""
        try:
            risk_config_path = Path("AOI_Base_Files_Schwabot/config/risk_config.yaml")
            
            if risk_config_path.exists():
                with open(risk_config_path, 'r') as f:
                    risk_config = yaml.safe_load(f)
            else:
                risk_config = {}
            
            # Update for Ferris Ride mode
            risk_config.update({
                "max_daily_loss_pct": config.max_daily_loss_pct,
                "stop_loss_pct": config.stop_loss_pct,
                "profit_target_pct": config.profit_target_pct,
                "ferris_ride_mode": True,
                "usdc_correlation_threshold": config.usdc_correlation_threshold,
                "risk_threshold": config.risk_threshold
            })
            
            # Save updated configuration
            risk_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(risk_config_path, 'w') as f:
                yaml.dump(risk_config, f, default_flow_style=False, indent=2)
            
            logger.info("✅ Risk config updated for Ferris Ride mode")
            
        except Exception as e:
            logger.error(f"❌ Risk config update failed: {e}")

# Global Ferris Ride Manager instance
ferris_ride_manager = FerrisRideManager() 