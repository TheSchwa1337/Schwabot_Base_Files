"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Shader Integration for Schwabot Trading System
================================================

    Comprehensive GPU shader integration system that:
    - Integrates system profiling with GPU DNA detection
    - Manages GLSL shader compilation and execution
    - Provides hardware-adaptive cosine similarity calculations
    - Interfaces with Schwabot's strategy matching pipeline'

        Key Features:
        - Automatic shader adaptation (Pi 4 → RTX 5090)
        - Real-time strategy vector matching
        - Performance monitoring and optimization
        - Fallback support for CPU-only systems
        """

        import logging
        import os
        import time
        from dataclasses import dataclass
        from typing import Any, Dict, Optional, Tuple

        import numpy as np
        import pygame

        # OpenGL imports (always import for linter/static analysis)
        from OpenGL.GL import (
            GL_CLAMP_TO_EDGE,
            GL_COLOR_ATTACHMENT0,
            GL_FLOAT,
            GL_FRAMEBUFFER,
            GL_FRAMEBUFFER_COMPLETE,
            GL_NEAREST,
            GL_QUADS,
            GL_R32F,
            GL_RED,
            GL_RENDERER,
            GL_TEXTURE0,
            GL_TEXTURE1,
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            GL_TEXTURE_WRAP_S,
            GL_TEXTURE_WRAP_T,
            GL_VENDOR,
            GL_VERSION,
            glActiveTexture,
            glBegin,
            glBindFramebuffer,
            glBindTexture,
            glCheckFramebufferStatus,
            glDeleteFramebuffers,
            glDeleteProgram,
            glDeleteShader,
            glDeleteTextures,
            glEnd,
            glFramebufferTexture2D,
            glGenFramebuffers,
            glGenTextures,
            glGetString,
            glReadPixels,
            glTexCoord2f,
            glTexImage2D,
            glTexParameteri,
            glUniform1f,
            glUniform1i,
            glUseProgram,
            glVertex2f,
            glViewport,
        )
        from OpenGL.GL.shaders import (
            GL_FRAGMENT_SHADER,
            GL_VERTEX_SHADER,
            compileProgram,
            compileShader,
            glGetUniformLocation,
        )

        from .gpu_dna_autodetect import ShaderConfig, detect_gpu_dna, get_gpu_shader_config
        from .system_state_profiler import SystemProfile, get_system_profile

        # Set OPENGL_AVAILABLE flag for runtime fallback
            try:
            import OpenGL.GL

            OPENGL_AVAILABLE = True
                except ImportError:
                OPENGL_AVAILABLE = False

                logger = logging.getLogger(__name__)


                @dataclass
                    class ShaderProgramConfig:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """Configuration for compiled shader program."""

                    program_id: int
                    vertex_shader_id: int
                    fragment_shader_id: int
                    uniform_locations: Dict[str, int]
                    matrix_size: int
                    precision_mode: str
                    morphing_enabled: bool


                        class GPUShaderIntegration:
    """Class for Schwabot trading functionality."""
                        """Class for Schwabot trading functionality."""
                        """Main GPU shader integration system for Schwabot."""

                            def __init__(self) -> None:
                            """Initialize GPU shader integration system."""
                            self.system_profile: Optional[SystemProfile] = None
                            self.gpu_dna_profile: Optional[Dict[str, Any]] = None
                            self.shader_config: Optional[ShaderConfig] = None
                            self.cosine_shader_program: Optional[ShaderProgramConfig] = None
                            self.opengl_initialized = False
                            self.performance_metrics = {
                            "shader_compile_time": 0.0,
                            "gpu_init_time": 0.0,
                            "average_execution_time": 0.0,
                            "operations_count": 0,
                            }

                                def initialize(self) -> bool:
                                """
                                Initialize the GPU shader integration system.

                                    Returns:
                                    bool: True if initialization successful, False otherwise
                                    """
                                    logger.info("🚀 Initializing GPU Shader Integration System...")

                                    start_time = time.time()

                                        try:
                                        # Step 1: System profiling
                                        logger.info("📊 Step 1: System Profiling...")
                                        self.system_profile = get_system_profile()

                                        # Step 2: GPU DNA detection
                                        logger.info("🧬 Step 2: GPU DNA Detection...")
                                        self.gpu_dna_profile = detect_gpu_dna()
                                        self.shader_config = get_gpu_shader_config()

                                        # Step 3: OpenGL initialization (if, available)
                                            if OPENGL_AVAILABLE:
                                            logger.info("🎮 Step 3: OpenGL Initialization...")
                                            self._initialize_opengl()
                                            self._compile_shaders()
                                                else:
                                                logger.warning("⚠️  OpenGL not available - using CPU fallback mode")

                                                init_time = time.time() - start_time
                                                self.performance_metrics["gpu_init_time"] = init_time

                                                logger.info("✅ GPU Shader Integration Initialized in {0:.2f}s".format(init_time))
                                                logger.info("🔧 System: {0}".format(self.system_profile.device_type))
                                                logger.info("🎮 GPU: {0}".format(self.system_profile.gpu.renderer))
                                                logger.info("📊 Matrix Size: {0}x{0}".format(self.shader_config.matrix_size))
                                                logger.info("⚡ Performance Multiplier: {0}x".format(self.shader_config.performance_multiplier))

                                            return True

                                                except Exception as e:
                                                logger.error("❌ GPU Shader Integration initialization failed: {0}".format(e))
                                            return False

                                                def _initialize_opengl(self) -> None:
                                                """Initialize OpenGL context for shader operations."""
                                                    try:
                                                    pygame.init()
                                                    pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.HIDDEN)

                                                    # Verify OpenGL context
                                                    vendor = glGetString(GL_VENDOR).decode()
                                                    renderer = glGetString(GL_RENDERER).decode()
                                                    version = glGetString(GL_VERSION).decode()

                                                    logger.info("🎮 OpenGL Context: {0} {1} {2}".format(vendor, renderer, version))

                                                    self.opengl_initialized = True

                                                        except Exception as e:
                                                        logger.error("OpenGL initialization failed: {0}".format(e))
                                                    raise

                                                        def _compile_shaders(self) -> None:
                                                        """Compile and configure GLSL shaders based on GPU capabilities."""
                                                        logger.info("🔧 Compiling adaptive GLSL shaders...")

                                                        start_time = time.time()

                                                            try:
                                                            # Load and adapt shader source
                                                            shader_source = self._load_adaptive_shader_source()

                                                            # Compile vertex shader
                                                            vertex_source = """
                                                            #version 300 es
                                                            in vec2 a_position;
                                                            in vec2 a_texCoord;
                                                            out vec2 v_texCoord;
                                                            void main() {
                                                            v_texCoord = a_texCoord;
                                                            gl_Position = vec4(a_position, 0.0, 1.0);
                                                            }
                                                            """

                                                            vertex_shader = compileShader(vertex_source, GL_VERTEX_SHADER)
                                                            fragment_shader = compileShader(shader_source, GL_FRAGMENT_SHADER)

                                                            # Link shader program
                                                            program = compileProgram(vertex_shader, fragment_shader)

                                                            # Get uniform locations
                                                            uniform_locations = {
                                                            "u_tick_vector": glGetUniformLocation(program, "u_tick_vector"),
                                                            "u_strategy_matrix": glGetUniformLocation(program, "u_strategy_matrix"),
                                                            "u_vector_length": glGetUniformLocation(program, "u_vector_length"),
                                                            "u_strategy_index": glGetUniformLocation(program, "u_strategy_index"),
                                                            "u_matrix_size": glGetUniformLocation(program, "u_matrix_size"),
                                                            "u_epsilon": glGetUniformLocation(program, "u_epsilon"),
                                                            "u_enable_morphing": glGetUniformLocation(program, "u_enable_morphing"),
                                                            "u_batch_size": glGetUniformLocation(program, "u_batch_size"),
                                                            }

                                                            # Store compiled shader configuration
                                                            self.cosine_shader_program = ShaderProgramConfig(
                                                            program_id=program,
                                                            vertex_shader_id=vertex_shader,
                                                            fragment_shader_id=fragment_shader,
                                                            uniform_locations=uniform_locations,
                                                            matrix_size=self.shader_config.matrix_size,
                                                            precision_mode="mediump" if self.shader_config.use_half_precision else "highp",
                                                            morphing_enabled=self.shader_config.shader_morph_enabled,
                                                            )

                                                            compile_time = time.time() - start_time
                                                            self.performance_metrics["shader_compile_time"] = compile_time

                                                            logger.info("✅ Shaders compiled successfully in {0:.2f}s".format(compile_time))
                                                            logger.info("🔧 Precision: {0}".format(self.cosine_shader_program.precision_mode))
                                                            logger.info(
                                                            "🌊 Morphing: {0}".format("Enabled" if self.cosine_shader_program.morphing_enabled else "Disabled")
                                                            )

                                                                except Exception as e:
                                                                logger.error("Shader compilation failed: {0}".format(e))
                                                            raise

                                                                def _load_adaptive_shader_source(self) -> str:
                                                                """Load and adapt shader source based on GPU capabilities."""
                                                                # Load base shader source
                                                                shader_path = os.path.join(os.path.dirname(__file__), "cosine_similarity_shader.glsl")

                                                                    try:
                                                                        with open(shader_path, "r") as f:
                                                                        shader_source = f.read()
                                                                            except FileNotFoundError:
                                                                            # Fallback to embedded shader source
                                                                            shader_source = self._get_embedded_shader_source()

                                                                            # Adapt shader based on GPU capabilities
                                                                            precision = "mediump" if self.shader_config.use_half_precision else "highp"
                                                                            shader_source = shader_source.replace("PRECISION_PLACEHOLDER", precision)

                                                                            # Add precision preprocessor directive for Pi 4 optimization
                                                                                if self.shader_config.use_half_precision:
                                                                                shader_source = "  #define MEDIUMP_PRECISION\n" + shader_source

                                                                            return shader_source

                                                                                def _get_embedded_shader_source(self) -> str:
                                                                                """Embedded fallback shader source."""
                                                                            return """
                                                                            #version 300 es
                                                                            precision PRECISION_PLACEHOLDER float;

                                                                            in vec2 v_texCoord;
                                                                            out vec4 fragColor;

                                                                            uniform sampler2D u_tick_vector;
                                                                            uniform sampler2D u_strategy_matrix;
                                                                            uniform int u_vector_length;
                                                                            uniform int u_strategy_index;
                                                                            uniform float u_epsilon;

                                                                            float computeCosineSimilarity(int vectorLength, int, strategyIndex) {
                                                                            float dotProduct = 0.0;
                                                                            float normA = 0.0;
                                                                            float normB = 0.0;

                                                                            for (int i = 0; i < vectorLength; i++) {
                                                                            if (i >= u_vector_length) break;

                                                                            float tickValue = texelFetch(u_tick_vector, ivec2(i, 0), 0).r;
                                                                            float strategyValue = texelFetch(u_strategy_matrix, ivec2(i, strategyIndex), 0).r;

                                                                            dotProduct += tickValue * strategyValue;
                                                                            normA += tickValue * tickValue;
                                                                            normB += strategyValue * strategyValue;
                                                                            }

                                                                            float denominator = sqrt(normA) * sqrt(normB) + u_epsilon;
                                                                        return dotProduct / denominator;
                                                                        }

                                                                        void main() {
                                                                        float similarity = computeCosineSimilarity(u_vector_length, u_strategy_index);
                                                                        fragColor = vec4(similarity, 0.0, 0.0, 1.0);
                                                                        }
                                                                        """

                                                                            def compute_strategy_similarity(self, tick_vector: np.ndarray, strategy_vectors: np.ndarray) -> np.ndarray:
                                                                            """
                                                                            Compute cosine similarity between tick vector and strategy vectors using GPU.

                                                                                Args:
                                                                                tick_vector: Current market tick vector (1D, array)
                                                                                strategy_vectors: Matrix of strategy vectors (2D, array)

                                                                                    Returns:
                                                                                    Array of cosine similarities for each strategy
                                                                                    """
                                                                                        if not self.opengl_initialized or not self.cosine_shader_program:
                                                                                        # Fallback to CPU computation
                                                                                    return self._compute_cpu_fallback(tick_vector, strategy_vectors)

                                                                                    start_time = time.time()

                                                                                        try:
                                                                                        # Prepare data for GPU
                                                                                        tick_texture = self._create_texture_from_vector(tick_vector)
                                                                                        strategy_texture = self._create_texture_from_matrix(strategy_vectors)

                                                                                        # Configure framebuffer for results
                                                                                        framebuffer, result_texture = self._create_result_framebuffer(strategy_vectors.shape[0])

                                                                                        # Execute shader
                                                                                        similarities = self._execute_cosine_shader(
                                                                                        tick_texture,
                                                                                        strategy_texture,
                                                                                        framebuffer,
                                                                                        tick_vector.shape[0],
                                                                                        strategy_vectors.shape[0],
                                                                                        )

                                                                                        # Cleanup GPU resources
                                                                                        glDeleteTextures([tick_texture, strategy_texture, result_texture])
                                                                                        glDeleteFramebuffers([framebuffer])

                                                                                        # Update performance metrics
                                                                                        execution_time = time.time() - start_time
                                                                                        self.performance_metrics["operations_count"] += 1
                                                                                        self.performance_metrics["average_execution_time"] = (
                                                                                        self.performance_metrics["average_execution_time"] * (self.performance_metrics["operations_count"] - 1)
                                                                                        + execution_time
                                                                                        ) / self.performance_metrics["operations_count"]

                                                                                        logger.debug("🔥 GPU cosine similarity computed in {0:.3f}s".format(execution_time))

                                                                                    return similarities

                                                                                        except Exception as e:
                                                                                        logger.warning("GPU computation failed, falling back to CPU: {0}".format(e))
                                                                                    return self._compute_cpu_fallback(tick_vector, strategy_vectors)

                                                                                        def _create_texture_from_vector(self, vector: np.ndarray) -> int:
                                                                                        """Create OpenGL texture from 1D vector."""
                                                                                        texture = glGenTextures(1)  # noqa: F405
                                                                                        glBindTexture(GL_TEXTURE_2D, texture)  # noqa: F405

                                                                                        # Ensure vector is float32
                                                                                        vector_f32 = vector.astype(np.float32)

                                                                                        # Create horizontal 1D texture (width = vector length, height = 1)
                                                                                        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, len(vector), 1, 0, GL_RED, GL_FLOAT, vector_f32)  # noqa: F405

                                                                                        # Set texture parameters
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)  # noqa: F405
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)  # noqa: F405
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)  # noqa: F405
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)  # noqa: F405

                                                                                    return texture

                                                                                        def _create_texture_from_matrix(self, matrix: np.ndarray) -> int:
                                                                                        """Create OpenGL texture from 2D matrix."""
                                                                                        texture = glGenTextures(1)  # noqa: F405
                                                                                        glBindTexture(GL_TEXTURE_2D, texture)  # noqa: F405

                                                                                        # Ensure matrix is float32 and transposed correctly for OpenGL
                                                                                        matrix_f32 = matrix.astype(np.float32)
                                                                                        height, width = matrix_f32.shape

                                                                                        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, matrix_f32)  # noqa: F405

                                                                                        # Set texture parameters
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)  # noqa: F405
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)  # noqa: F405
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)  # noqa: F405
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)  # noqa: F405

                                                                                    return texture

                                                                                        def _create_result_framebuffer(self, num_strategies: int) -> Tuple[int, int]:
                                                                                        """Create framebuffer for shader results."""
                                                                                        # Create result texture
                                                                                        result_texture = glGenTextures(1)  # noqa: F405
                                                                                        glBindTexture(GL_TEXTURE_2D, result_texture)  # noqa: F405
                                                                                        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, num_strategies, 1, 0, GL_RED, GL_FLOAT, None)  # noqa: F405
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)  # noqa: F405
                                                                                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)  # noqa: F405

                                                                                        # Create framebuffer
                                                                                        framebuffer = glGenFramebuffers(1)  # noqa: F405
                                                                                        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)  # noqa: F405
                                                                                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, result_texture, 0)  # noqa: F405

                                                                                        # Check framebuffer completeness
                                                                                        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:  # noqa: F405
                                                                                    raise RuntimeError("Framebuffer not complete")

                                                                                return framebuffer, result_texture

                                                                                def _execute_cosine_shader(
                                                                                self,
                                                                                tick_texture: int,
                                                                                strategy_texture: int,
                                                                                framebuffer: int,
                                                                                vector_length: int,
                                                                                num_strategies: int,
                                                                                    ) -> np.ndarray:
                                                                                    """Execute cosine similarity shader and return results."""

                                                                                    # Bind shader program
                                                                                    glUseProgram(self.cosine_shader_program.program_id)  # noqa: F405

                                                                                    # Bind textures
                                                                                    glActiveTexture(GL_TEXTURE0)  # noqa: F405
                                                                                    glBindTexture(GL_TEXTURE_2D, tick_texture)  # noqa: F405
                                                                                    glUniform1i(self.cosine_shader_program.uniform_locations["u_tick_vector"], 0)

                                                                                    glActiveTexture(GL_TEXTURE1)  # noqa: F405
                                                                                    glBindTexture(GL_TEXTURE_2D, strategy_texture)  # noqa: F405
                                                                                    glUniform1i(self.cosine_shader_program.uniform_locations["u_strategy_matrix"], 1)

                                                                                    # Set uniforms
                                                                                    glUniform1i(self.cosine_shader_program.uniform_locations["u_vector_length"], vector_length)
                                                                                    glUniform1i(
                                                                                    self.cosine_shader_program.uniform_locations["u_matrix_size"],
                                                                                    self.shader_config.matrix_size,
                                                                                    )
                                                                                    glUniform1f(self.cosine_shader_program.uniform_locations["u_epsilon"], 1e-8)
                                                                                    glUniform1i(
                                                                                    self.cosine_shader_program.uniform_locations["u_enable_morphing"],
                                                                                    1 if self.shader_config.shader_morph_enabled else 0,
                                                                                    )
                                                                                    glUniform1i(
                                                                                    self.cosine_shader_program.uniform_locations["u_batch_size"],
                                                                                    self.shader_config.batch_size,
                                                                                    )

                                                                                    # Set viewport and render to framebuffer
                                                                                    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
                                                                                    glViewport(0, 0, num_strategies, 1)

                                                                                    results = np.zeros(num_strategies, dtype=np.float32)

                                                                                    # Render for each strategy (or in batches if, supported)
                                                                                        for strategy_idx in range(0, num_strategies, self.shader_config.batch_size):
                                                                                        glUniform1i(self.cosine_shader_program.uniform_locations["u_strategy_index"], strategy_idx)

                                                                                        # Render quad
                                                                                        self._render_fullscreen_quad()

                                                                                        # Read results
                                                                                        batch_end = min(strategy_idx + self.shader_config.batch_size, num_strategies)
                                                                                        batch_size = batch_end - strategy_idx

                                                                                        glReadPixels(
                                                                                        strategy_idx,
                                                                                        0,
                                                                                        batch_size,
                                                                                        1,
                                                                                        GL_RED,  # noqa: F405
                                                                                        GL_FLOAT,  # noqa: F405
                                                                                        results[strategy_idx : strategy_idx + batch_size],
                                                                                        )

                                                                                        # Restore default framebuffer
                                                                                        glBindFramebuffer(GL_FRAMEBUFFER, 0)

                                                                                    return results

                                                                                        def _render_fullscreen_quad(self) -> None:
                                                                                        """Render a fullscreen quad for shader execution."""
                                                                                        # Simplified quad rendering - in production would use VAO/VBO
                                                                                        glBegin(GL_QUADS)  # noqa: F405
                                                                                        glTexCoord2f(0.0, 0.0)  # noqa: F405
                                                                                        glVertex2f(-1.0, -1.0)  # noqa: F405
                                                                                        glTexCoord2f(1.0, 0.0)  # noqa: F405
                                                                                        glVertex2f(1.0, -1.0)  # noqa: F405
                                                                                        glTexCoord2f(1.0, 1.0)  # noqa: F405
                                                                                        glVertex2f(1.0, 1.0)  # noqa: F405
                                                                                        glTexCoord2f(0.0, 1.0)  # noqa: F405
                                                                                        glVertex2f(-1.0, 1.0)  # noqa: F405
                                                                                        glEnd()  # noqa: F405

                                                                                            def _compute_cpu_fallback(self, tick_vector: np.ndarray, strategy_vectors: np.ndarray) -> np.ndarray:
                                                                                            """CPU fallback for cosine similarity computation."""
                                                                                            logger.debug(" Using CPU fallback for cosine similarity")

                                                                                            # Normalize vectors
                                                                                            tick_norm = np.linalg.norm(tick_vector)
                                                                                            strategy_norms = np.linalg.norm(strategy_vectors, axis=1)

                                                                                            # Compute dot products
                                                                                            dot_products = np.dot(strategy_vectors, tick_vector)

                                                                                            # Compute cosine similarities with epsilon protection
                                                                                            epsilon = 1e-8
                                                                                            similarities = dot_products / (tick_norm * strategy_norms + epsilon)

                                                                                        return similarities

                                                                                            def get_performance_metrics(self) -> Dict[str, Any]:
                                                                                            """Get performance metrics for GPU operations."""
                                                                                        return {
                                                                                        **self.performance_metrics,
                                                                                        "gpu_tier": self.shader_config.gpu_tier if self.shader_config else "unknown",
                                                                                        "matrix_size": self.shader_config.matrix_size if self.shader_config else 0,
                                                                                        "morphing_enabled": (self.shader_config.shader_morph_enabled if self.shader_config else False),
                                                                                        "opengl_available": OPENGL_AVAILABLE,
                                                                                        "opengl_initialized": self.opengl_initialized,
                                                                                        }

                                                                                            def cleanup(self) -> None:
                                                                                            """Cleanup GPU resources."""
                                                                                                if self.cosine_shader_program:
                                                                                                    try:
                                                                                                    glDeleteProgram(self.cosine_shader_program.program_id)  # noqa: F405
                                                                                                    glDeleteShader(self.cosine_shader_program.vertex_shader_id)  # noqa: F405
                                                                                                    glDeleteShader(self.cosine_shader_program.fragment_shader_id)  # noqa: F405
                                                                                                        except Exception:
                                                                                                    pass

                                                                                                        if self.opengl_initialized:
                                                                                                            try:
                                                                                                            pygame.quit()
                                                                                                                except Exception:
                                                                                                            pass


                                                                                                            # Factory functions
                                                                                                                def create_gpu_shader_integration() -> GPUShaderIntegration:
                                                                                                                """Create and initialize GPU shader integration system."""
                                                                                                                integration = GPUShaderIntegration()
                                                                                                                integration.initialize()
                                                                                                            return integration


                                                                                                                def compute_strategy_similarities_gpu(tick_vector: np.ndarray, strategy_vectors: np.ndarray) -> np.ndarray:
                                                                                                                """Compute strategy similarities using GPU acceleration."""
                                                                                                                integration = create_gpu_shader_integration()
                                                                                                                    try:
                                                                                                                return integration.compute_strategy_similarity(tick_vector, strategy_vectors)
                                                                                                                    finally:
                                                                                                                    integration.cleanup()


                                                                                                                    # Export key components
                                                                                                                    __all__ = [
                                                                                                                    "GPUShaderIntegration",
                                                                                                                    "ShaderProgramConfig",
                                                                                                                    "create_gpu_shader_integration",
                                                                                                                    "compute_strategy_similarities_gpu",
                                                                                                                    ]
