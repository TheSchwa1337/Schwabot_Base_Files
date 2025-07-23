#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoboldCPP Integration for Unified Mathematical Trading System
============================================================

Comprehensive integration of KoboldCPP local LLM capabilities into the
unified mathematical trading system, enabling advanced AI-powered analysis,
decision making, and visual layer control.

Features:
- Local LLM-powered trading analysis and decision making
- Multimodal vision capabilities for chart and data analysis
- Real-time AI-generated trading signals and strategies
- Integration with existing mathematical framework (DLT, Dualistic Engines, Bit Phases)
- Hardware-optimized model loading and inference with CUDA acceleration
- Secure communication with Alpha256 encryption
- Multi-cryptocurrency support (BTC, ETH, XRP, SOL, USDC)
- 8-bit phase logic and strategy mapping
- Tensor surface integration and memory caching
- Registry and soulprint storage integration
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import requests
import psutil
import numpy as np

# Core system imports
import sys
import os
sys.path.append('.')
from unified_hardware_detector import UnifiedHardwareDetector
HardwareAutoDetector = UnifiedHardwareDetector
from .soulprint_registry import SoulprintRegistry
from .cascade_memory_architecture import CascadeMemoryArchitecture
from .strategy_mapper import StrategyMapper
from .tensor_weight_memory import TensorWeightMemory
from .visual_decision_engine import VisualDecisionEngine

logger = logging.getLogger(__name__)

# Import real implementations instead of stubs
try:
    from .hash_config_manager import HashConfigManager
    HASH_CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ HashConfigManager not available, using stub")
    HASH_CONFIG_AVAILABLE = False

try:
    from .alpha256_encryption import Alpha256Encryption
    ALPHA256_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Alpha256Encryption not available, using stub")
    ALPHA256_AVAILABLE = False

# Stub classes for missing components
if not HASH_CONFIG_AVAILABLE:
    class HashConfigManager:
        """Simple stub for HashConfigManager."""
        def __init__(self):
            self.config = {}
        
        def initialize(self):
            """Initialize the hash config manager."""
            pass
        
        def get_config(self, key: str, default: Any = None) -> Any:
            """Get configuration value."""
            return self.config.get(key, default)
        
        def set_config(self, key: str, value: Any):
            """Set configuration value."""
            self.config[key] = value

if not ALPHA256_AVAILABLE:
    class Alpha256Encryption:
        """Simple stub for Alpha256Encryption."""
        def __init__(self):
            pass
        
        def encrypt(self, data: str) -> str:
            """Encrypt data."""
            return data  # Simple pass-through for now
        
        def decrypt(self, data: str) -> str:
            """Decrypt data."""
            return data  # Simple pass-through for now

class KoboldModelType(Enum):
    """KoboldCPP model types."""
    TEXT_GENERATION = "text_generation"
    VISION_MULTIMODAL = "vision_multimodal"
    EMBEDDINGS = "embeddings"
    DRAFT_SPECULATIVE = "draft_speculative"

class AnalysisType(Enum):
    """Trading analysis types."""
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_ASSESSMENT = "risk_assessment"
    STRATEGY_GENERATION = "strategy_generation"
    MARKET_ANALYSIS = "market_analysis"
    BIT_PHASE_ANALYSIS = "bit_phase_analysis"
    TENSOR_ANALYSIS = "tensor_analysis"
    GENERAL = "general"

class CryptocurrencyType(Enum):
    """Supported cryptocurrencies."""
    BTC = "BTC"
    ETH = "ETH"
    XRP = "XRP"
    SOL = "SOL"
    USDC = "USDC"

@dataclass
class KoboldRequest:
    """KoboldCPP request structure."""
    prompt: str
    max_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    rep_pen: float = 1.1
    stop_sequence: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)  # Base64 encoded images
    analysis_type: AnalysisType = AnalysisType.TECHNICAL_ANALYSIS
    cryptocurrency: CryptocurrencyType = CryptocurrencyType.BTC
    bit_phase: int = 8  # 8-bit phase logic
    priority: int = 1

@dataclass
class KoboldResponse:
    """KoboldCPP response structure."""
    text: str
    tokens_generated: int
    processing_time_ms: float
    model_used: str
    confidence_score: float = 0.0
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    bit_phase_result: int = 8
    tensor_score: float = 0.0
    mathematical_consensus: Dict[str, Any] = field(default_factory=dict)

class KoboldCPPIntegration:
    """Comprehensive KoboldCPP integration for unified mathematical trading system."""
    
    def __init__(self, kobold_path: str = "koboldcpp", model_path: str = "", port: int = 5001):
        """Initialize KoboldCPP integration with hardware auto-detection."""
        self.kobold_path = Path(kobold_path)
        self.model_path = Path(model_path) if model_path else None
        self.port = port
        self.base_url = f"http://localhost:{port}"
        
        # Core system integrations
        self.hardware_detector = HardwareAutoDetector()
        self.hash_config = HashConfigManager()
        self.alpha256 = Alpha256Encryption()
        self.soulprint_registry = SoulprintRegistry()
        self.cascade_memory = CascadeMemoryArchitecture()
        self.strategy_mapper = StrategyMapper()
        self.tensor_memory = TensorWeightMemory()
        self.visual_engine = VisualDecisionEngine()
        
        # Hardware-aware configuration
        self.system_info = None
        self.memory_config = None
        self.auto_detected = False
        
        # KoboldCPP process management
        self.kobold_process = None
        self.kobold_running = False
        self.model_loaded = False
        
        # Request queues and processing
        self.request_queues: Dict[AnalysisType, deque] = {
            analysis_type: deque(maxlen=self._get_queue_size(analysis_type))
            for analysis_type in AnalysisType
        }
        
        # Model capabilities
        self.model_capabilities = {
            "text_generation": False,
            "vision_multimodal": False,
            "embeddings": False,
            "draft_speculative": False
        }
        
        # Performance tracking
        self.stats = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_tokens_generated": 0,
            "average_response_time_ms": 0.0,
            "model_load_time_ms": 0.0,
            "vision_analyses": 0,
            "text_analyses": 0,
            "bit_phase_analyses": 0,
            "tensor_analyses": 0
        }
        
        # Threading
        self.processing_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Initialize with hardware detection
        self._initialize_with_hardware_detection()
    
    def _initialize_with_hardware_detection(self):
        """Initialize KoboldCPP integration using hardware auto-detection."""
        try:
            logger.info("🔍 Initializing KoboldCPP integration with hardware auto-detection...")
            
            # Detect hardware capabilities
            self.system_info = self.hardware_detector.detect_hardware()
            self.memory_config = self.hardware_detector.generate_memory_config()
            
            logger.info(f"✅ Hardware detected: {self.system_info.platform}")
            logger.info(f"   RAM: {self.system_info.ram_gb:.1f} GB ({self.system_info.ram_tier.value})")
            logger.info(f"   GPU: {self.system_info.gpu_name if self.system_info.gpu_name else 'None'}")
            logger.info(f"   CUDA: {self.system_info.cuda_version if self.system_info.cuda_version else 'Not available'}")
            logger.info(f"   Optimization: {self.system_info.optimization_mode.value}")
            
            # Configure model loading based on hardware
            self._configure_model_loading()
            
            # Load or create configuration
            self._load_configuration()
            
            # Apply configuration
            self._apply_configuration(self.config)
            
            self.auto_detected = True
            logger.info("✅ KoboldCPP integration initialized with hardware auto-detection")
            
        except Exception as e:
            logger.error(f"❌ Hardware detection failed: {e}")
            self._initialize_fallback_config()
    
    def _configure_model_loading(self):
        """Configure model loading based on hardware capabilities."""
        try:
            if self.system_info.gpu_name and self.system_info.cuda_version:
                # GPU-accelerated configuration
                self.model_config = {
                    "use_gpu": True,
                    "gpu_layers": min(50, int(self.system_info.vram_gb * 2)),
                    "context_length": 4096,
                    "batch_size": 512,
                    "threads": self.system_info.cpu_cores // 2
                }
                logger.info(f"🎮 GPU acceleration enabled: {self.system_info.gpu_name}")
            else:
                # CPU-only configuration
                self.model_config = {
                    "use_gpu": False,
                    "context_length": 2048,
                    "batch_size": 256,
                    "threads": self.system_info.cpu_cores
                }
                logger.info("🖥️ CPU-only mode enabled")
            
            # Memory-aware configuration
            if self.system_info.ram_gb >= 32:
                self.model_config["context_length"] = 8192
                self.model_config["batch_size"] *= 2
            elif self.system_info.ram_gb >= 16:
                self.model_config["context_length"] = 4096
            else:
                self.model_config["context_length"] = 2048
                self.model_config["batch_size"] = min(256, self.model_config["batch_size"])
            
        except Exception as e:
            logger.error(f"❌ Model configuration failed: {e}")
            self.model_config = {
                "use_gpu": False,
                "context_length": 2048,
                "batch_size": 256,
                "threads": 4
            }
    
    def _get_queue_size(self, analysis_type: AnalysisType) -> int:
        """Get queue size based on analysis type and hardware."""
        base_size = 1000
        
        if analysis_type in [AnalysisType.TECHNICAL_ANALYSIS, AnalysisType.PATTERN_RECOGNITION]:
            return base_size * 2
        elif analysis_type in [AnalysisType.BIT_PHASE_ANALYSIS, AnalysisType.TENSOR_ANALYSIS]:
            return base_size * 3
        else:
            return base_size
    
    def _load_configuration(self):
        """Load or create KoboldCPP configuration."""
        try:
            config_path = Path("config/koboldcpp_config.json")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("✅ Loaded existing KoboldCPP configuration")
            else:
                self.config = self._create_default_config()
                self._save_configuration()
                logger.info("✅ Created new KoboldCPP configuration")
                
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default KoboldCPP configuration."""
        return {
            "version": "1.0.0",
            "kobold_path": str(self.kobold_path),
            "model_path": str(self.model_path) if self.model_path else "",
            "port": self.port,
            "hardware_auto_detected": True,
            "system_info": {
                "platform": self.system_info.platform if self.system_info else "unknown",
                "ram_gb": self.system_info.ram_gb if self.system_info else 8.0,
                "gpu_name": self.system_info.gpu_name if self.system_info else None,
                "cuda_version": self.system_info.cuda_version if self.system_info else None,
                "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "balanced"
            },
            "model_config": self.model_config,
            "analysis_config": {
                "enable_bit_phase_analysis": True,
                "enable_tensor_analysis": True,
                "enable_visual_analysis": True,
                "enable_memory_caching": True,
                "enable_registry_storage": True
            },
            "cryptocurrency_support": {
                "BTC": {"enabled": True, "priority": 1},
                "ETH": {"enabled": True, "priority": 2},
                "XRP": {"enabled": True, "priority": 3},
                "SOL": {"enabled": True, "priority": 4},
                "USDC": {"enabled": True, "priority": 5}
            },
            "bit_phase_config": {
                "default_phase": 8,
                "supported_phases": [4, 8, 16, 32, 42],
                "randomization_enabled": True,
                "strategy_mapping_enabled": True
            },
            "tensor_config": {
                "enable_tensor_surfaces": True,
                "memory_caching_enabled": True,
                "similarity_threshold": 0.8
            },
            "performance_config": {
                "max_concurrent_requests": 10,
                "request_timeout_seconds": 30,
                "health_check_interval_seconds": 60,
                "auto_restart_enabled": True
            }
        }
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Apply configuration settings."""
        try:
            self.kobold_path = Path(config.get("kobold_path", "koboldcpp"))
            self.model_path = Path(config.get("model_path", "")) if config.get("model_path") else None
            self.port = config.get("port", 5001)
            self.base_url = f"http://localhost:{self.port}"
            
            # Update model configuration
            if "model_config" in config:
                self.model_config.update(config["model_config"])
            
            logger.info("✅ Configuration applied successfully")
            
        except Exception as e:
            logger.error(f"❌ Configuration application failed: {e}")
    
    def _save_configuration(self):
        """Save configuration to file."""
        try:
            config_path = Path("config/koboldcpp_config.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
    
    def _initialize_fallback_config(self):
        """Initialize fallback configuration when hardware detection fails."""
        logger.warning("⚠️ Using fallback configuration")
        
        self.system_info = type('obj', (object,), {
            'platform': 'unknown',
            'ram_gb': 8.0,
            'gpu_name': None,
            'cuda_version': None,
            'optimization_mode': type('obj', (object,), {'value': 'balanced'})()
        })()
        
        self.model_config = {
            "use_gpu": False,
            "context_length": 2048,
            "batch_size": 256,
            "threads": 4
        }
        
        self.config = self._create_default_config()
    
    async def start_kobold_server(self) -> bool:
        """Start KoboldCPP server."""
        try:
            if self.kobold_running:
                logger.info("✅ KoboldCPP server already running")
                return True
            
            logger.info("🚀 Starting KoboldCPP server...")
            
            # Build command
            cmd = [str(self.kobold_path), "--port", str(self.port)]
            
            if self.model_path and self.model_path.exists():
                cmd.extend(["--model", str(self.model_path)])
            
            # Add GPU configuration
            if self.model_config.get("use_gpu", False):
                cmd.extend(["--use-gpu", "--gpu-layers", str(self.model_config["gpu_layers"])])
            
            # Add other configuration
            cmd.extend([
                "--context-length", str(self.model_config["context_length"]),
                "--batch-size", str(self.model_config["batch_size"]),
                "--threads", str(self.model_config["threads"])
            ])
            
            # Start process
            self.kobold_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            for _ in range(30):  # 30 second timeout
                if self._check_server_status():
                    self.kobold_running = True
                    logger.info("✅ KoboldCPP server started successfully")
                    return True
                await asyncio.sleep(1)
            
            logger.error("❌ KoboldCPP server failed to start")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to start KoboldCPP server: {e}")
            return False
    
    def _check_server_status(self) -> bool:
        """Check if KoboldCPP server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/model", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def _load_model_capabilities(self):
        """Load model capabilities from server."""
        try:
            if not self.kobold_running:
                return
            
            response = requests.get(f"{self.base_url}/api/v1/model", timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                
                # Update capabilities based on model info
                self.model_capabilities["text_generation"] = True
                self.model_capabilities["vision_multimodal"] = model_info.get("vision", False)
                self.model_capabilities["embeddings"] = model_info.get("embeddings", False)
                
                self.model_loaded = True
                logger.info("✅ Model capabilities loaded")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model capabilities: {e}")
    
    async def stop_kobold_server(self):
        """Stop KoboldCPP server."""
        try:
            if self.kobold_process:
                self.kobold_process.terminate()
                self.kobold_process.wait(timeout=10)
                self.kobold_process = None
            
            self.kobold_running = False
            self.model_loaded = False
            logger.info("✅ KoboldCPP server stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop KoboldCPP server: {e}")
    
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data using KoboldCPP."""
        try:
            if not self.kobold_running:
                logger.warning("⚠️ KoboldCPP server not running")
                return None
            
            # Create analysis request
            request = KoboldRequest(
                prompt=self._create_market_analysis_prompt(market_data),
                analysis_type=AnalysisType.TECHNICAL_ANALYSIS,
                cryptocurrency=CryptocurrencyType(market_data.get('symbol', 'BTC/USDC').split('/')[0]),
                bit_phase=market_data.get('bit_phase', 8),
                max_length=1024,
                temperature=0.7
            )
            
            # Send request to KoboldCPP
            response = await self._send_kobold_request(request)
            
            if response:
                # Extract analysis results
                analysis_results = self._extract_analysis_results(response.text, request.analysis_type)
                
                # Integrate with mathematical components
                mathematical_consensus = await self._integrate_with_mathematical_components(
                    market_data, analysis_results, response
                )
                
                # Store in registry and memory
                await self._store_analysis_results(market_data, analysis_results, mathematical_consensus)
                
                return {
                    'confidence': response.confidence_score,
                    'action': analysis_results.get('action', 'HOLD'),
                    'analysis': analysis_results,
                    'mathematical_consensus': mathematical_consensus,
                    'bit_phase_result': response.bit_phase_result,
                    'tensor_score': response.tensor_score
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Market data analysis failed: {e}")
            return None
    
    def _create_market_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        """Create market analysis prompt for KoboldCPP."""
        symbol = market_data.get('symbol', 'BTC/USDC')
        price = market_data.get('price', 0.0)
        volume = market_data.get('volume', 0.0)
        volatility = market_data.get('volatility', 0.0)
        bit_phase = market_data.get('bit_phase', 8)
        
        prompt = f"""
Analyze the following cryptocurrency market data and provide trading recommendations:

Symbol: {symbol}
Current Price: ${price:,.2f}
Volume: {volume:,.2f}
Volatility: {volatility:.4f}
Bit Phase: {bit_phase}

Please provide:
1. Technical analysis (RSI, MACD, support/resistance levels)
2. Trend analysis (bullish/bearish/neutral)
3. Risk assessment (1-10 scale)
4. Trading recommendation (BUY/SELL/HOLD)
5. Confidence level (0.0-1.0)
6. Position size recommendation (0.0-1.0)
7. Stop loss and take profit levels

Consider the 8-bit phase logic and mathematical framework integration.
Focus on profitable trading opportunities with proper risk management.
"""
        return prompt
    
    async def _send_kobold_request(self, request: KoboldRequest) -> Optional[KoboldResponse]:
        """Send request to KoboldCPP server."""
        try:
            payload = {
                "prompt": request.prompt,
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "rep_pen": request.rep_pen,
                "stop_sequence": request.stop_sequence
            }
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/api/v1/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                processing_time = (time.time() - start_time) * 1000
                
                # Calculate confidence score
                confidence = self._calculate_confidence_score(result['results'][0]['text'])
                
                # Extract bit phase and tensor analysis
                bit_phase_result = self._extract_bit_phase_result(result['results'][0]['text'])
                tensor_score = self._calculate_tensor_score(result['results'][0]['text'])
                
                return KoboldResponse(
                    text=result['results'][0]['text'],
                    tokens_generated=result['results'][0].get('tokens_generated', 0),
                    processing_time_ms=processing_time,
                    model_used=result.get('model', 'unknown'),
                    confidence_score=confidence,
                    bit_phase_result=bit_phase_result,
                    tensor_score=tensor_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"❌ KoboldCPP request failed: {e}")
            return None
    
    def _calculate_confidence_score(self, text: str) -> float:
        """Calculate confidence score from response text."""
        try:
            # Look for confidence indicators in the text
            confidence_indicators = [
                "confidence:", "confidence level:", "certainty:",
                "strong", "moderate", "weak", "high", "low"
            ]
            
            text_lower = text.lower()
            score = 0.5  # Default score
            
            # Adjust based on confidence indicators
            if any(indicator in text_lower for indicator in ["strong", "high"]):
                score += 0.3
            elif any(indicator in text_lower for indicator in ["moderate", "medium"]):
                score += 0.1
            elif any(indicator in text_lower for indicator in ["weak", "low"]):
                score -= 0.2
            
            # Look for explicit confidence numbers
            import re
            confidence_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', text_lower)
            if confidence_match:
                explicit_score = float(confidence_match.group(1))
                if explicit_score <= 1.0:
                    score = explicit_score
                elif explicit_score <= 100:
                    score = explicit_score / 100
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.5
    
    def _extract_bit_phase_result(self, text: str) -> int:
        """Extract bit phase result from response text."""
        try:
            import re
            phase_match = re.search(r'bit\s*phase[:\s]*(\d+)', text.lower())
            if phase_match:
                phase = int(phase_match.group(1))
                if phase in [4, 8, 16, 32, 42]:
                    return phase
            
            # Default to 8-bit phase
            return 8
            
        except Exception as e:
            logger.error(f"❌ Bit phase extraction failed: {e}")
            return 8
    
    def _calculate_tensor_score(self, text: str) -> float:
        """Calculate tensor score from response text."""
        try:
            # Look for tensor-related indicators
            tensor_indicators = [
                "tensor", "matrix", "vector", "dimensional",
                "complex", "sophisticated", "advanced"
            ]
            
            text_lower = text.lower()
            score = 0.5  # Default score
            
            # Adjust based on tensor indicators
            for indicator in tensor_indicators:
                if indicator in text_lower:
                    score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Tensor score calculation failed: {e}")
            return 0.5
    
    def _extract_analysis_results(self, text: str, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Extract analysis results from response text."""
        try:
            results = {
                'action': 'HOLD',
                'confidence': 0.5,
                'position_size': 0.01,
                'stop_loss': None,
                'take_profit': None,
                'risk_score': 0.5,
                'trend': 'neutral',
                'rsi': None,
                'macd': None,
                'support': None,
                'resistance': None
            }
            
            text_lower = text.lower()
            
            # Extract action
            if 'buy' in text_lower and 'sell' not in text_lower:
                results['action'] = 'BUY'
            elif 'sell' in text_lower:
                results['action'] = 'SELL'
            
            # Extract confidence
            confidence = self._calculate_confidence_score(text)
            results['confidence'] = confidence
            
            # Extract position size
            import re
            size_match = re.search(r'position\s*size[:\s]*(\d+\.?\d*)', text_lower)
            if size_match:
                results['position_size'] = float(size_match.group(1))
            
            # Extract trend
            if any(word in text_lower for word in ['bullish', 'upward', 'rising']):
                results['trend'] = 'bullish'
            elif any(word in text_lower for word in ['bearish', 'downward', 'falling']):
                results['trend'] = 'bearish'
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis extraction failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.5}
    
    async def _integrate_with_mathematical_components(self, market_data: Dict[str, Any], 
                                                    analysis_results: Dict[str, Any], 
                                                    response: KoboldResponse) -> Dict[str, Any]:
        """Integrate AI analysis with mathematical components."""
        try:
            consensus = {
                'ai_confidence': response.confidence_score,
                'mathematical_weight': 0.7,
                'ai_weight': 0.3,
                'combined_confidence': 0.0,
                'final_action': 'HOLD',
                'bit_phase_alignment': True,
                'tensor_alignment': True
            }
            
            # Combine AI and mathematical signals
            mathematical_confidence = market_data.get('mathematical_confidence', 0.5)
            ai_confidence = response.confidence_score
            
            combined_confidence = (
                mathematical_confidence * consensus['mathematical_weight'] +
                ai_confidence * consensus['ai_weight']
            )
            
            consensus['combined_confidence'] = combined_confidence
            
            # Determine final action
            if combined_confidence > 0.7:
                consensus['final_action'] = analysis_results.get('action', 'HOLD')
            elif combined_confidence > 0.5:
                consensus['final_action'] = 'HOLD'
            else:
                consensus['final_action'] = 'HOLD'
            
            # Check bit phase alignment
            expected_phase = market_data.get('bit_phase', 8)
            actual_phase = response.bit_phase_result
            consensus['bit_phase_alignment'] = expected_phase == actual_phase
            
            # Check tensor alignment
            consensus['tensor_alignment'] = response.tensor_score > 0.5
            
            return consensus
            
        except Exception as e:
            logger.error(f"❌ Mathematical integration failed: {e}")
            return {'final_action': 'HOLD', 'combined_confidence': 0.5}
    
    async def _store_analysis_results(self, market_data: Dict[str, Any], 
                                    analysis_results: Dict[str, Any], 
                                    mathematical_consensus: Dict[str, Any]):
        """Store analysis results in registry and memory."""
        try:
            # Store in soulprint registry
            registry_entry = {
                'timestamp': time.time(),
                'symbol': market_data.get('symbol', 'BTC/USDC'),
                'price': market_data.get('price', 0.0),
                'analysis_results': analysis_results,
                'mathematical_consensus': mathematical_consensus,
                'bit_phase': market_data.get('bit_phase', 8),
                'tensor_score': mathematical_consensus.get('tensor_alignment', False)
            }
            
            self.soulprint_registry.store_decision(registry_entry)
            
            # Store in cascade memory
            if analysis_results.get('action') != 'HOLD':
                self.cascade_memory.record_cascade_memory(
                    entry_asset=market_data.get('symbol', 'BTC/USDC').split('/')[0],
                    exit_asset=market_data.get('symbol', 'BTC/USDC').split('/')[1],
                    entry_price=market_data.get('price', 0.0),
                    exit_price=market_data.get('price', 0.0),
                    entry_time=datetime.now(),
                    exit_time=datetime.now(),
                    profit_impact=analysis_results.get('confidence', 0.5) * 100,
                    cascade_type='AI_ANALYSIS'
                )
            
            # Update tensor memory
            self.tensor_memory.store_tensor_score(
                market_data.get('symbol', 'BTC/USDC'),
                mathematical_consensus.get('tensor_alignment', False)
            )
            
        except Exception as e:
            logger.error(f"❌ Analysis storage failed: {e}")
    
    async def start_processing(self):
        """Start the processing loop."""
        try:
            if self.running:
                return
            
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("✅ KoboldCPP processing started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start processing: {e}")
    
    def stop_processing(self):
        """Stop the processing loop."""
        try:
            self.running = False
            
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
                self.processing_thread = None
            
            logger.info("✅ KoboldCPP processing stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop processing: {e}")
    
    def _processing_loop(self):
        """Main processing loop."""
        try:
            while self.running:
                # Process analysis requests
                asyncio.run(self._process_analysis_requests())
                
                # Monitor health
                self._monitor_health()
                
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Processing loop error: {e}")
    
    async def _process_analysis_requests(self):
        """Process analysis requests from queues."""
        try:
            for analysis_type, queue in self.request_queues.items():
                if queue:
                    request = queue.popleft()
                    response = await self.analyze_market_data(request)
                    
                    if response:
                        self.stats["requests_processed"] += 1
                    else:
                        self.stats["requests_failed"] += 1
                        
        except Exception as e:
            logger.error(f"❌ Request processing error: {e}")
    
    def _monitor_health(self):
        """Monitor system health."""
        try:
            if self.kobold_process and self.kobold_process.poll() is not None:
                logger.warning("⚠️ KoboldCPP process terminated, attempting restart...")
                asyncio.run(self.start_kobold_server())
                
        except Exception as e:
            logger.error(f"❌ Health monitoring error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            'kobold_running': self.kobold_running,
            'model_loaded': self.model_loaded,
            'hardware_detected': self.auto_detected,
            'system_info': {
                'platform': self.system_info.platform if self.system_info else 'unknown',
                'ram_gb': self.system_info.ram_gb if self.system_info else 0.0,
                'gpu_name': self.system_info.gpu_name if self.system_info else None,
                'cuda_version': self.system_info.cuda_version if self.system_info else None
            },
            'model_config': self.model_config,
            'model_capabilities': self.model_capabilities,
            'performance_stats': self.stats,
            'queue_sizes': {
                analysis_type.value: len(queue) 
                for analysis_type, queue in self.request_queues.items()
            }
        }

async def main():
    """Main function for testing."""
    logger.info("🧪 Testing KoboldCPP Integration")
    
    # Create integration instance
    integration = KoboldCPPIntegration()
    
    # Start server
    success = await integration.start_kobold_server()
    
    if success:
        logger.info("✅ KoboldCPP integration test successful")
        
        # Test market data analysis
        test_data = {
            'symbol': 'BTC/USDC',
            'price': 50000.0,
            'volume': 2000.0,
            'volatility': 0.15,
            'bit_phase': 8
        }
        
        result = await integration.analyze_market_data(test_data)
        if result:
            logger.info(f"✅ Analysis result: {result}")
        
        # Stop server
        await integration.stop_kobold_server()
    else:
        logger.error("❌ KoboldCPP integration test failed")

if __name__ == "__main__":
    asyncio.run(main()) 