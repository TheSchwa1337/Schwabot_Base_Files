#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware Optimization Script for Schwabot Trading System
=======================================================

Simple CLI script to automatically detect hardware and optimize system configuration.
This script will:

1. Detect your hardware (GPU, RAM, CPU)
2. Generate optimal memory configurations
3. Update all configuration files automatically
4. Create backups before making changes
5. Validate the configuration

Usage:
    python hardware_optimize.py [--force-redetect] [--validate-only]

Examples:
    python hardware_optimize.py                    # Normal optimization
    python hardware_optimize.py --force-redetect   # Force re-detection
    python hardware_optimize.py --validate-only    # Only validate existing config
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.hardware_auto_detector import HardwareAutoDetector
from core.hardware_config_integrator import HardwareConfigIntegrator

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/hardware_optimization.log')
        ]
    )

def main():
    """Main function for hardware optimization."""
    parser = argparse.ArgumentParser(
        description="Hardware Optimization for Schwabot Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hardware_optimize.py                    # Normal optimization
  python hardware_optimize.py --force-redetect   # Force re-detection
  python hardware_optimize.py --validate-only    # Only validate existing config
  python hardware_optimize.py --detect-only      # Only detect hardware, don't update configs
        """
    )
    
    parser.add_argument(
        '--force-redetect',
        action='store_true',
        help='Force hardware re-detection even if configuration exists'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing configuration, don\'t make changes'
    )
    
    parser.add_argument(
        '--detect-only',
        action='store_true',
        help='Only detect hardware and show summary, don\'t update configs'
    )
    
    parser.add_argument(
        '--backup-only',
        action='store_true',
        help='Only create backup of current configuration'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 Schwabot Hardware Optimization Tool")
    print("=" * 50)
    
    try:
        if args.validate_only:
            # Only validate existing configuration
            logger.info("🔍 Validating existing configuration...")
            integrator = HardwareConfigIntegrator()
            if integrator.validate_configuration():
                print("✅ Configuration validation passed!")
                return 0
            else:
                print("❌ Configuration validation failed!")
                return 1
        
        elif args.backup_only:
            # Only create backup
            logger.info("📋 Creating configuration backup...")
            integrator = HardwareConfigIntegrator()
            integrator._create_config_backup()
            print("✅ Configuration backup created!")
            return 0
        
        elif args.detect_only:
            # Only detect hardware
            logger.info("🔍 Detecting hardware...")
            detector = HardwareAutoDetector()
            
            if detector.load_configuration():
                print("✅ Loaded existing hardware configuration")
            else:
                print("🔍 Performing hardware detection...")
                detector.detect_hardware()
                detector.generate_memory_config()
                detector.save_configuration()
            
            detector.print_system_summary()
            return 0
        
        else:
            # Full optimization
            logger.info("🚀 Starting full hardware optimization...")
            
            # Create integrator
            integrator = HardwareConfigIntegrator()
            
            # Perform hardware configuration integration
            success = integrator.integrate_hardware_config(force_redetect=args.force_redetect)
            
            if success:
                # Validate the configuration
                if integrator.validate_configuration():
                    print("\n🎉 Hardware optimization completed successfully!")
                    print("The system is now optimized for your hardware configuration.")
                    print("\nKey optimizations applied:")
                    
                    # Show key optimizations
                    detector = integrator.detector
                    print(f"  • GPU: {detector.system_info.gpu.name} ({detector.system_info.gpu.tier.value})")
                    print(f"  • RAM: {detector.system_info.ram_gb:.1f} GB ({detector.system_info.ram_tier.value})")
                    print(f"  • Optimization Mode: {detector.system_info.optimization_mode.value}")
                    
                    # Show TIC map optimizations
                    print(f"  • TIC Map Sizes:")
                    for bit_depth, size in detector.memory_config.tic_map_sizes.items():
                        print(f"    - {bit_depth}: {size:,} operations")
                    
                    # Show cache optimizations
                    print(f"  • Cache Sizes:")
                    for cache_type, size in detector.memory_config.cache_sizes.items():
                        print(f"    - {cache_type}: {size:,} entries")
                    
                    print(f"\n💾 Configuration backup created at: {integrator.backup_dir}")
                    print(f"📁 Hardware config saved to: config/hardware_auto_config.json")
                    
                    return 0
                else:
                    print("\n❌ Configuration validation failed after optimization.")
                    print("Check the logs for more details.")
                    return 1
            else:
                print("\n❌ Hardware optimization failed.")
                print("Check the logs for more details.")
                return 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.")
        return 1
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        print("Check the logs for more details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 