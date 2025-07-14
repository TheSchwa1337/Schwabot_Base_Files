#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Hardware Detection Script
==============================

Simple test script to verify hardware detection works properly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.hardware_auto_detector import HardwareAutoDetector

def main():
    print("Testing Hardware Detection...")
    print("=" * 40)
    
    detector = HardwareAutoDetector()
    
    # Detect hardware
    system_info = detector.detect_hardware()
    
    print(f"Platform: {system_info.platform}")
    print(f"CPU: {system_info.cpu_model}")
    print(f"CPU Cores: {system_info.cpu_cores}")
    print(f"RAM: {system_info.ram_gb:.1f} GB")
    print(f"GPU: {system_info.gpu.name}")
    print(f"GPU Memory: {system_info.gpu.memory_gb:.1f} GB")
    print(f"GPU Tier: {system_info.gpu.tier.value}")
    print(f"CUDA Cores: {system_info.gpu.cuda_cores}")
    print(f"Optimization Mode: {system_info.optimization_mode.value}")
    
    # Generate memory config
    memory_config = detector.generate_memory_config()
    
    print("\nMemory Configuration:")
    print("-" * 20)
    for bit_depth, size in memory_config.tic_map_sizes.items():
        print(f"{bit_depth}: {size:,}")
    
    print("\nCache Sizes:")
    print("-" * 20)
    for cache_type, size in memory_config.cache_sizes.items():
        print(f"{cache_type}: {size:,}")
    
    print("\nMemory Pools:")
    print("-" * 20)
    for pool_name, pool_config in memory_config.memory_pools.items():
        print(f"{pool_name}: {pool_config['size_mb']} MB ({pool_config['bit_depth']})")
    
    # Save configuration
    detector.save_configuration()
    print(f"\nConfiguration saved to: config/hardware_auto_config.json")

if __name__ == "__main__":
    main() 