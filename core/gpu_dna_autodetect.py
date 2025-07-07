#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU DNA Auto-Detection for Schwabot Trading System
================================================

GPU-specific detection and shader configuration system that:
- Auto-detects GPU hardware and capabilities
- Maps GPUs to optimal shader configurations
- Provides hardware-specific matrix operation settings
- Integrates with Schwabot's trading pipeline for optimal performance

Key Features:
- OpenGL-based GPU fingerprinting
- Shader complexity auto-scaling (Pi 4 → RTX 5090)
- Matrix size optimization based on GPU tier
- Precision mode selection (half vs float)
- Fallback support for headless/integrated systems
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    from OpenGL.GL import *
    from OpenGL.GL.shaders import *
    import pygame
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

from .system_state_profiler import (
    SystemStateProfiler, 
    GPUTier, 
    GPUProfile,
    get_system_profile
)

logger = logging.getLogger(__name__)


@dataclass
class ShaderConfig:
    """GPU shader configuration optimized for specific hardware."""
    matrix_size: int
    batch_size: int
    use_half_precision: bool
    shader_morph_enabled: bool
    max_texture_size: int
    fragment_passes: int
    instanced_rendering: bool
    gpu_tier: str
    performance_multiplier: float


class GPUDNAAutoDetect:
    """GPU DNA detection and configuration system."""
    
    # Performance multipliers by GPU tier
    PERFORMANCE_MULTIPLIERS = {
        GPUTier.TIER_PI4: 1.0,      # Baseline (Pi 4)
        GPUTier.TIER_LOW: 2.0,      # 2x Pi 4 performance
        GPUTier.TIER_MID: 5.0,      # 5x Pi 4 performance  
        GPUTier.TIER_HIGH: 12.0,    # 12x Pi 4 performance
        GPUTier.TIER_ULTRA: 30.0,   # 30x Pi 4 performance
        GPUTier.TIER_UNKNOWN: 1.5   # Conservative estimate
    }
    
    # Shader configuration templates by tier
    SHADER_CONFIGS = {
        GPUTier.TIER_PI4: {
            "matrix_size": 8,
            "batch_size": 1,
            "use_half_precision": True,
            "shader_morph_enabled": False,
            "max_texture_size": 512,
            "fragment_passes": 1,
            "instanced_rendering": False
        },
        GPUTier.TIER_LOW: {
            "matrix_size": 16,
            "batch_size": 2,
            "use_half_precision": True,
            "shader_morph_enabled": False,
            "max_texture_size": 1024,
            "fragment_passes": 2,
            "instanced_rendering": False
        },
        GPUTier.TIER_MID: {
            "matrix_size": 32,
            "batch_size": 4,
            "use_half_precision": False,
            "shader_morph_enabled": True,
            "max_texture_size": 2048,
            "fragment_passes": 4,
            "instanced_rendering": True
        },
        GPUTier.TIER_HIGH: {
            "matrix_size": 64,
            "batch_size": 8,
            "use_half_precision": False,
            "shader_morph_enabled": True,
            "max_texture_size": 4096,
            "fragment_passes": 8,
            "instanced_rendering": True
        },
        GPUTier.TIER_ULTRA: {
            "matrix_size": 128,
            "batch_size": 16,
            "use_half_precision": False,
            "shader_morph_enabled": True,
            "max_texture_size": 8192,
            "fragment_passes": 16,
            "instanced_rendering": True
        },
        GPUTier.TIER_UNKNOWN: {
            "matrix_size": 16,
            "batch_size": 1,
            "use_half_precision": True,
            "shader_morph_enabled": False,
            "max_texture_size": 1024,
            "fragment_passes": 1,
            "instanced_rendering": False
        }
    }
    
    def __init__(self):
        self.system_profile = None
        self.shader_config = None
        self.gpu_capabilities = None
        
    def detect_gpu_dna(self) -> Dict[str, Any]:
        """
        Comprehensive GPU DNA detection and configuration.
        
        Returns:
            Dict containing GPU fingerprint and optimized shader config
        """
        logger.info("🧬 Detecting GPU DNA and shader capabilities...")
        
        # Get system profile
        self.system_profile = get_system_profile()
        gpu_profile = self.system_profile.gpu
        
        # Get GPU capabilities if OpenGL is available
        if OPENGL_AVAILABLE:
            self.gpu_capabilities = self._probe_gpu_capabilities()
        else:
            self.gpu_capabilities = self._create_fallback_capabilities()
        
        # Generate shader configuration
        self.shader_config = self._generate_shader_config(gpu_profile)
        
        # Create comprehensive DNA profile
        dna_profile = {
            "gpu_fingerprint": {
                "vendor": gpu_profile.vendor,
                "renderer": gpu_profile.renderer,
                "gl_version": gpu_profile.gl_version,
                "glsl_version": gpu_profile.glsl_version,
                "gpu_tier": gpu_profile.gpu_tier.value,
                "system_tier": self.system_profile.system_tier.value
            },
            "gpu_capabilities": self.gpu_capabilities,
            "shader_config": {
                "matrix_size": self.shader_config.matrix_size,
                "batch_size": self.shader_config.batch_size,
                "use_half_precision": self.shader_config.use_half_precision,
                "shader_morph_enabled": self.shader_config.shader_morph_enabled,
                "max_texture_size": self.shader_config.max_texture_size,
                "fragment_passes": self.shader_config.fragment_passes,
                "instanced_rendering": self.shader_config.instanced_rendering,
                "gpu_tier": self.shader_config.gpu_tier,
                "performance_multiplier": self.shader_config.performance_multiplier
            },
            "system_hash": self.system_profile.system_hash,
            "detection_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        # Save DNA profile
        self._save_dna_profile(dna_profile)
        
        logger.info(f"✅ GPU DNA Detection Complete")
        logger.info(f"🎮 GPU: {gpu_profile.renderer} ({gpu_profile.gpu_tier.value})")
        logger.info(f"📊 Matrix Size: {self.shader_config.matrix_size}x{self.shader_config.matrix_size}")
        logger.info(f"⚡ Performance Multiplier: {self.shader_config.performance_multiplier}x")
        logger.info(f"🔧 Shader Morph: {'Enabled' if self.shader_config.shader_morph_enabled else 'Disabled'}")
        
        return dna_profile
    
    def _probe_gpu_capabilities(self) -> Dict[str, Any]:
        """Probe GPU capabilities using OpenGL."""
        try:
            # Initialize minimal OpenGL context
            pygame.init()
            pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.HIDDEN)
            
            capabilities = {
                "max_texture_size": glGetIntegerv(GL_MAX_TEXTURE_SIZE),
                "max_vertex_attribs": glGetIntegerv(GL_MAX_VERTEX_ATTRIBS),
                "max_uniform_locations": glGetIntegerv(GL_MAX_UNIFORM_LOCATIONS),
                "max_texture_image_units": glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS),
                "max_combined_texture_image_units": glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS),
                "max_vertex_uniform_vectors": glGetIntegerv(GL_MAX_VERTEX_UNIFORM_VECTORS),
                "max_fragment_uniform_vectors": glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS),
                "max_varying_vectors": glGetIntegerv(GL_MAX_VARYING_VECTORS),
                "vendor": glGetString(GL_VENDOR).decode(),
                "renderer": glGetString(GL_RENDERER).decode(),
                "version": glGetString(GL_VERSION).decode(),
                "shading_language_version": glGetString(GL_SHADING_LANGUAGE_VERSION).decode()
            }
            
            # Get extensions
            num_extensions = glGetIntegerv(GL_NUM_EXTENSIONS)
            extensions = []
            for i in range(num_extensions):
                ext = glGetStringi(GL_EXTENSIONS, i)
                if ext:
                    extensions.append(ext.decode())
            
            capabilities["extensions"] = extensions[:50]  # Limit to first 50 extensions
            
            pygame.quit()
            
            logger.info(f"🔍 GPU Capabilities Probed: {len(capabilities)} properties detected")
            return capabilities
            
        except Exception as e:
            logger.warning(f"GPU capability probing failed: {e}")
            return self._create_fallback_capabilities()
    
    def _create_fallback_capabilities(self) -> Dict[str, Any]:
        """Create fallback GPU capabilities."""
        return {
            "max_texture_size": 1024,
            "max_vertex_attribs": 16,
            "max_uniform_locations": 256,
            "max_texture_image_units": 8,
            "max_combined_texture_image_units": 16,
            "max_vertex_uniform_vectors": 256,
            "max_fragment_uniform_vectors": 256,
            "max_varying_vectors": 16,
            "vendor": "Unknown",
            "renderer": "Fallback Detection",
            "version": "Unknown",
            "shading_language_version": "Unknown",
            "extensions": []
        }
    
    def _generate_shader_config(self, gpu_profile: GPUProfile) -> ShaderConfig:
        """Generate optimized shader configuration for detected GPU."""
        gpu_tier = gpu_profile.gpu_tier
        
        # Get base configuration for tier
        base_config = self.SHADER_CONFIGS.get(gpu_tier, self.SHADER_CONFIGS[GPUTier.TIER_UNKNOWN])
        
        # Get performance multiplier
        performance_multiplier = self.PERFORMANCE_MULTIPLIERS.get(gpu_tier, 1.0)
        
        # Apply GPU capability adjustments if available
        if self.gpu_capabilities:
            # Adjust texture size based on GPU limits
            max_texture = self.gpu_capabilities.get("max_texture_size", 1024)
            if base_config["max_texture_size"] > max_texture:
                base_config["max_texture_size"] = max_texture
            
            # Adjust matrix size if texture limits are restrictive
            if max_texture < 2048 and base_config["matrix_size"] > 32:
                base_config["matrix_size"] = 32
                logger.info(f"⚠️  Matrix size reduced to {base_config['matrix_size']} due to texture limits")
        
        return ShaderConfig(
            matrix_size=base_config["matrix_size"],
            batch_size=base_config["batch_size"],
            use_half_precision=base_config["use_half_precision"],
            shader_morph_enabled=base_config["shader_morph_enabled"],
            max_texture_size=base_config["max_texture_size"],
            fragment_passes=base_config["fragment_passes"],
            instanced_rendering=base_config["instanced_rendering"],
            gpu_tier=gpu_tier.value,
            performance_multiplier=performance_multiplier
        )
    
    def _save_dna_profile(self, dna_profile: Dict[str, Any]):
        """Save GPU DNA profile to disk."""
        try:
            # Ensure directory exists
            dna_dir = "init/gpu_dna_profiles"
            os.makedirs(dna_dir, exist_ok=True)
            
            # Save latest DNA profile
            latest_path = os.path.join(dna_dir, "latest_gpu_dna.json")
            with open(latest_path, "w") as f:
                json.dump(dna_profile, f, indent=2)
            
            # Save timestamped profile
            timestamp = dna_profile["detection_timestamp"].replace(":", "-")
            system_hash = dna_profile["system_hash"][:8]
            timestamped_path = os.path.join(
                dna_dir, 
                f"gpu_dna_{timestamp}_{system_hash}.json"
            )
            with open(timestamped_path, "w") as f:
                json.dump(dna_profile, f, indent=2)
            
            logger.info(f"💾 GPU DNA Profile saved: {latest_path}")
            
        except Exception as e:
            logger.error(f"Failed to save GPU DNA profile: {e}")
    
    def get_shader_config(self) -> ShaderConfig:
        """Get current shader configuration."""
        if not self.shader_config:
            self.detect_gpu_dna()
        return self.shader_config
    
    def get_cosine_similarity_config(self) -> Dict[str, Any]:
        """Get configuration specifically for cosine similarity shader operations."""
        if not self.shader_config:
            self.detect_gpu_dna()
        
        config = self.get_shader_config()
        
        return {
            "matrix_size": config.matrix_size,
            "precision": "mediump" if config.use_half_precision else "highp",
            "batch_strategies": config.batch_size,
            "enable_morphing": config.shader_morph_enabled,
            "texture_format": "GL_R16F" if config.use_half_precision else "GL_R32F",
            "fragment_shader_version": "#version 300 es" if config.gpu_tier in ["pi4", "low"] else "#version 330 core"
        }
    
    def run_gpu_fit_test(self) -> Dict[str, Any]:
        """
        Run GPU fit test to validate shader capabilities.
        
        Returns:
            Test results including maximum viable matrix size
        """
        logger.info("🧪 Running GPU Fit Test...")
        
        if not OPENGL_AVAILABLE:
            logger.warning("OpenGL not available - skipping fit test")
            return {
                "test_passed": False,
                "max_matrix_size": 16,
                "error": "OpenGL not available"
            }
        
        try:
            # Initialize test environment
            pygame.init()
            pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.HIDDEN)
            
            # Test shader compilation
            test_vertex_shader = """
            #version 300 es
            in vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """
            
            test_fragment_shader = """
            #version 300 es
            precision highp float;
            out vec4 fragColor;
            uniform sampler2D u_test_texture;
            void main() {
                vec2 texCoord = gl_FragCoord.xy / vec2(64.0, 64.0);
                fragColor = texture(u_test_texture, texCoord);
            }
            """
            
            # Try to compile test shaders
            vertex_shader = compileShader(test_vertex_shader, GL_VERTEX_SHADER)
            fragment_shader = compileShader(test_fragment_shader, GL_FRAGMENT_SHADER)
            test_program = compileProgram(vertex_shader, fragment_shader)
            
            # Test matrix sizes starting from configured size
            config = self.get_shader_config()
            test_sizes = [8, 16, 32, 64, 128, 256]
            max_working_size = 8
            
            for size in test_sizes:
                if size > config.matrix_size * 2:
                    break
                    
                try:
                    # Create test texture
                    texture = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, texture)
                    
                    # Try to allocate texture memory
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, size, size, 0, GL_RED, GL_FLOAT, None)
                    
                    # Check for GL errors
                    error = glGetError()
                    if error == GL_NO_ERROR:
                        max_working_size = size
                        logger.info(f"✅ Matrix size {size}x{size} - OK")
                    else:
                        logger.warning(f"❌ Matrix size {size}x{size} - Failed (GL Error: {error})")
                        break
                    
                    glDeleteTextures([texture])
                    
                except Exception as e:
                    logger.warning(f"❌ Matrix size {size}x{size} - Exception: {e}")
                    break
            
            pygame.quit()
            
            test_result = {
                "test_passed": True,
                "max_matrix_size": max_working_size,
                "configured_size": config.matrix_size,
                "recommended_size": min(max_working_size, config.matrix_size),
                "gpu_tier": config.gpu_tier,
                "performance_multiplier": config.performance_multiplier
            }
            
            logger.info(f"🧪 GPU Fit Test Complete - Max Size: {max_working_size}x{max_working_size}")
            return test_result
            
        except Exception as e:
            logger.error(f"GPU fit test failed: {e}")
            return {
                "test_passed": False,
                "max_matrix_size": 16,
                "error": str(e)
            }


# Factory functions
def create_gpu_dna_detector() -> GPUDNAAutoDetect:
    """Create a new GPU DNA auto-detector."""
    return GPUDNAAutoDetect()


def detect_gpu_dna() -> Dict[str, Any]:
    """Detect GPU DNA and return configuration."""
    detector = create_gpu_dna_detector()
    return detector.detect_gpu_dna()


def get_gpu_shader_config() -> ShaderConfig:
    """Get optimized shader configuration for current GPU."""
    detector = create_gpu_dna_detector()
    return detector.get_shader_config()


def get_cosine_similarity_config() -> Dict[str, Any]:
    """Get configuration for cosine similarity shader operations."""
    detector = create_gpu_dna_detector()
    return detector.get_cosine_similarity_config()


def run_gpu_fit_test() -> Dict[str, Any]:
    """Run GPU fit test to validate capabilities."""
    detector = create_gpu_dna_detector()
    return detector.run_gpu_fit_test()


# Export key components
__all__ = [
    "GPUDNAAutoDetect",
    "ShaderConfig", 
    "create_gpu_dna_detector",
    "detect_gpu_dna",
    "get_gpu_shader_config",
    "get_cosine_similarity_config",
    "run_gpu_fit_test"
] 