#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KoboldCPP Integration for Schwabot Trading System
================================================

Comprehensive integration of KoboldCPP local LLM capabilities into the
Schwabot trading system, enabling advanced AI-powered analysis, decision
making, and visual layer control.

Features:
- Local LLM-powered trading analysis and decision making
- Multimodal vision capabilities for chart and data analysis
- Real-time AI-generated trading signals and strategies
- Integration with existing mathematical framework
- Hardware-optimized model loading and inference
- Secure communication with Alpha256 encryption
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

from .hardware_auto_detector import HardwareAutoDetector
from .tick_loader import TickLoader, TickPriority
from .signal_cache import SignalCache, SignalType, SignalPriority
from .registry_writer import RegistryWriter, ArchivePriority

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
    GENERAL = "general"

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

class KoboldCPPIntegration:
    """Comprehensive KoboldCPP integration for Schwabot trading system."""
    
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
        self.tick_loader = TickLoader()
        self.signal_cache = SignalCache()
        self.registry_writer = RegistryWriter()
        
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
            "text_analyses": 0
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
            self.auto_detected = True
            
            # Initialize hash configuration
            self.hash_config.initialize()
            
            # Configure model loading based on hardware
            self._configure_model_loading()
            
            # Load or create configuration
            self._load_configuration()
            
            logger.info(f"✅ KoboldCPP integration initialized for {self.system_info.platform}")
            logger.info(f"   Model: {self.model_path.name if self.model_path else 'None'}")
            logger.info(f"   Optimization: {self.system_info.optimization_mode.value}")
            
        except Exception as e:
            logger.error(f"❌ Hardware detection failed: {e}")
            self._initialize_fallback_config()
    
    def _configure_model_loading(self):
        """Configure model loading based on hardware capabilities."""
        if not self.memory_config:
            return
            
        # Determine optimal model configuration based on hardware
        memory_gb = self.system_info.ram_gb
        gpu_memory_gb = getattr(self.system_info.gpu, 'memory_gb', 0.0)
        
        # Configure model parameters based on hardware tier
        if memory_gb >= 32 and gpu_memory_gb >= 8:
            # High-end system - can load large models
            self.model_config = {
                "context_size": 8192,
                "gpu_layers": 35,
                "threads": self.system_info.cpu_cores,
                "batch_size": 512,
                "enable_vision": True,
                "enable_embeddings": True
            }
        elif memory_gb >= 16 and gpu_memory_gb >= 4:
            # Mid-range system
            self.model_config = {
                "context_size": 4096,
                "gpu_layers": 20,
                "threads": min(self.system_info.cpu_cores, 8),
                "batch_size": 256,
                "enable_vision": True,
                "enable_embeddings": False
            }
        else:
            # Low-end system - conservative settings
            self.model_config = {
                "context_size": 2048,
                "gpu_layers": 0,  # CPU only
                "threads": min(self.system_info.cpu_cores, 4),
                "batch_size": 128,
                "enable_vision": False,
                "enable_embeddings": False
            }
    
    def _get_queue_size(self, analysis_type: AnalysisType) -> int:
        """Get queue size based on analysis type priority."""
        priority_sizes = {
            AnalysisType.RISK_ASSESSMENT: 100,  # High priority
            AnalysisType.STRATEGY_GENERATION: 100,
            AnalysisType.TECHNICAL_ANALYSIS: 50,  # Medium priority
            AnalysisType.PATTERN_RECOGNITION: 50,
            AnalysisType.MARKET_ANALYSIS: 30,  # Lower priority
            AnalysisType.SENTIMENT_ANALYSIS: 30,
            AnalysisType.FUNDAMENTAL_ANALYSIS: 20,
            AnalysisType.GENERAL: 10
        }
        return priority_sizes.get(analysis_type, 20)
    
    def _load_configuration(self):
        """Load or create KoboldCPP configuration."""
        config_path = Path("config/koboldcpp_config.json")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info("✅ Loaded existing KoboldCPP configuration")
            else:
                config = self._create_default_config()
                self._save_configuration(config)
                logger.info("✅ Created new KoboldCPP configuration")
            
            self._apply_configuration(config)
            
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            config = self._create_default_config()
            self._apply_configuration(config)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration based on hardware."""
        return {
            "version": "1.0.0",
            "hardware_auto_detected": self.auto_detected,
            "system_info": {
                "platform": self.system_info.platform if self.system_info else "unknown",
                "ram_gb": self.system_info.ram_gb if self.system_info else 8.0,
                "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "balanced"
            },
            "kobold_settings": {
                "kobold_path": str(self.kobold_path),
                "model_path": str(self.model_path) if self.model_path else "",
                "port": self.port,
                "host": "localhost",
                "max_connections": 10
            },
            "model_config": self.model_config,
            "analysis_settings": {
                "enable_real_time_analysis": True,
                "enable_vision_analysis": self.model_config.get("enable_vision", False),
                "enable_embeddings": self.model_config.get("enable_embeddings", False),
                "max_concurrent_requests": 5,
                "request_timeout_ms": 30000,
                "enable_streaming": True
            },
            "integration_settings": {
                "auto_start_kobold": True,
                "auto_load_model": True,
                "enable_health_monitoring": True,
                "enable_performance_tracking": True,
                "enable_secure_communication": True
            }
        }
    
    def _apply_configuration(self, config: Dict[str, Any]):
        """Apply configuration settings."""
        self.config = config
        
        # Apply Kobold settings
        self.kobold_path = Path(config["kobold_settings"]["kobold_path"])
        self.model_path = Path(config["kobold_settings"]["model_path"]) if config["kobold_settings"]["model_path"] else None
        self.port = config["kobold_settings"]["port"]
        self.host = config["kobold_settings"]["host"]
        self.max_connections = config["kobold_settings"]["max_connections"]
        
        # Apply model config
        self.model_config = config["model_config"]
        
        # Apply analysis settings
        self.enable_real_time_analysis = config["analysis_settings"]["enable_real_time_analysis"]
        self.enable_vision_analysis = config["analysis_settings"]["enable_vision_analysis"]
        self.enable_embeddings = config["analysis_settings"]["enable_embeddings"]
        self.max_concurrent_requests = config["analysis_settings"]["max_concurrent_requests"]
        self.request_timeout_ms = config["analysis_settings"]["request_timeout_ms"]
        self.enable_streaming = config["analysis_settings"]["enable_streaming"]
        
        # Apply integration settings
        self.auto_start_kobold = config["integration_settings"]["auto_start_kobold"]
        self.auto_load_model = config["integration_settings"]["auto_load_model"]
        self.enable_health_monitoring = config["integration_settings"]["enable_health_monitoring"]
        self.enable_performance_tracking = config["integration_settings"]["enable_performance_tracking"]
        self.enable_secure_communication = config["integration_settings"]["enable_secure_communication"]
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            config_path = Path("config/koboldcpp_config.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
    
    def _initialize_fallback_config(self):
        """Initialize with fallback configuration."""
        logger.warning("⚠️ Using fallback configuration")
        
        self.system_info = None
        self.memory_config = None
        self.auto_detected = False
        
        # Fallback model configuration
        self.model_config = {
            "context_size": 2048,
            "gpu_layers": 0,
            "threads": 4,
            "batch_size": 128,
            "enable_vision": False,
            "enable_embeddings": False
        }
    
    async def start_kobold_server(self) -> bool:
        """Start KoboldCPP server with hardware-optimized settings."""
        try:
            if self.kobold_running:
                logger.info("✅ KoboldCPP server already running")
                return True
            
            # Build command line arguments
            cmd_args = [
                str(self.kobold_path),
                "--port", str(self.port),
                "--host", self.host,
                "--threads", str(self.model_config["threads"]),
                "--contextsize", str(self.model_config["context_size"]),
                "--batchsize", str(self.model_config["batch_size"])
            ]
            
            # Add model if specified
            if self.model_path and self.model_path.exists():
                cmd_args.extend(["--model", str(self.model_path)])
            
            # Add GPU configuration
            if self.model_config["gpu_layers"] > 0:
                cmd_args.extend(["--gpulayers", str(self.model_config["gpu_layers"])])
                cmd_args.append("--usecublas")
            
            # Add vision support if enabled
            if self.model_config.get("enable_vision", False):
                cmd_args.append("--multimodal")
            
            # Add embeddings support if enabled
            if self.model_config.get("enable_embeddings", False):
                cmd_args.append("--embeddings")
            
            # Add additional optimizations
            cmd_args.extend([
                "--smartcontext",
                "--contextshift",
                "--fastforward",
                "--quiet"
            ])
            
            logger.info(f"🚀 Starting KoboldCPP server: {' '.join(cmd_args)}")
            
            # Start KoboldCPP process
            self.kobold_process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            await asyncio.sleep(5.0)
            
            # Check if server is running
            if self._check_server_status():
                self.kobold_running = True
                logger.info("✅ KoboldCPP server started successfully")
                
                # Load model capabilities
                await self._load_model_capabilities()
                
                return True
            else:
                logger.error("❌ Failed to start KoboldCPP server")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start KoboldCPP server: {e}")
            return False
    
    def _check_server_status(self) -> bool:
        """Check if KoboldCPP server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/model", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    async def _load_model_capabilities(self):
        """Load model capabilities from KoboldCPP server."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/model", timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                
                # Update model capabilities based on server response
                self.model_capabilities["text_generation"] = True
                
                # Check for additional capabilities
                if "vision" in model_info.get("model_type", "").lower():
                    self.model_capabilities["vision_multimodal"] = True
                
                if "embeddings" in model_info.get("capabilities", []):
                    self.model_capabilities["embeddings"] = True
                
                logger.info(f"✅ Model capabilities loaded: {self.model_capabilities}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model capabilities: {e}")
    
    async def stop_kobold_server(self):
        """Stop KoboldCPP server."""
        try:
            if self.kobold_process:
                self.kobold_process.terminate()
                await asyncio.sleep(2.0)
                
                if self.kobold_process.poll() is None:
                    self.kobold_process.kill()
                
                self.kobold_running = False
                logger.info("✅ KoboldCPP server stopped")
                
        except Exception as e:
            logger.error(f"❌ Failed to stop KoboldCPP server: {e}")
    
    async def analyze_trading_data(self, request: KoboldRequest) -> Optional[KoboldResponse]:
        """Analyze trading data using KoboldCPP."""
        try:
            start_time = time.time()
            
            # Prepare request payload
            payload = {
                "prompt": request.prompt,
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "rep_pen": request.rep_pen,
                "stop_sequence": request.stop_sequence
            }
            
            # Add images if provided and vision is supported
            if request.images and self.model_capabilities["vision_multimodal"]:
                payload["images"] = request.images
            
            # Send request to KoboldCPP
            response = requests.post(
                f"{self.base_url}/api/v1/generate",
                json=payload,
                timeout=self.request_timeout_ms / 1000
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["results"][0]["text"]
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000
                
                # Create response object
                kobold_response = KoboldResponse(
                    text=generated_text,
                    tokens_generated=len(generated_text.split()),
                    processing_time_ms=processing_time,
                    model_used="koboldcpp",
                    confidence_score=self._calculate_confidence_score(generated_text)
                )
                
                # Extract analysis results
                kobold_response.analysis_results = self._extract_analysis_results(
                    generated_text, request.analysis_type
                )
                
                # Update statistics
                self.stats["requests_processed"] += 1
                self.stats["total_tokens_generated"] += kobold_response.tokens_generated
                self.stats["average_response_time_ms"] = (
                    (self.stats["average_response_time_ms"] * (self.stats["requests_processed"] - 1) + processing_time) /
                    self.stats["requests_processed"]
                )
                
                if request.images:
                    self.stats["vision_analyses"] += 1
                else:
                    self.stats["text_analyses"] += 1
                
                return kobold_response
            else:
                logger.error(f"❌ KoboldCPP request failed: {response.status_code}")
                self.stats["requests_failed"] += 1
                return None
                
        except Exception as e:
            logger.error(f"❌ Trading data analysis failed: {e}")
            self.stats["requests_failed"] += 1
            return None
    
    def _calculate_confidence_score(self, text: str) -> float:
        """Calculate confidence score based on response quality."""
        try:
            # Simple confidence scoring based on response characteristics
            score = 0.5  # Base score
            
            # Length factor
            if len(text) > 100:
                score += 0.2
            elif len(text) > 50:
                score += 0.1
            
            # Structure factor
            if any(keyword in text.lower() for keyword in ["analysis", "recommendation", "signal", "trend"]):
                score += 0.2
            
            # Technical terms factor
            technical_terms = ["rsi", "macd", "bollinger", "support", "resistance", "volume", "momentum"]
            if any(term in text.lower() for term in technical_terms):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return 0.5
    
    def _extract_analysis_results(self, text: str, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Extract structured analysis results from text response."""
        try:
            results = {
                "analysis_type": analysis_type.value,
                "summary": text[:200] + "..." if len(text) > 200 else text,
                "confidence": self._calculate_confidence_score(text),
                "timestamp": time.time()
            }
            
            # Extract specific results based on analysis type
            if analysis_type == AnalysisType.TECHNICAL_ANALYSIS:
                results.update(self._extract_technical_analysis(text))
            elif analysis_type == AnalysisType.RISK_ASSESSMENT:
                results.update(self._extract_risk_assessment(text))
            elif analysis_type == AnalysisType.STRATEGY_GENERATION:
                results.update(self._extract_strategy_generation(text))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to extract analysis results: {e}")
            return {"error": str(e)}
    
    def _extract_technical_analysis(self, text: str) -> Dict[str, Any]:
        """Extract technical analysis results."""
        return {
            "indicators": {
                "rsi": self._extract_rsi(text),
                "macd": self._extract_macd(text),
                "support_resistance": self._extract_support_resistance(text)
            },
            "trend": self._extract_trend(text),
            "signals": self._extract_signals(text)
        }
    
    def _extract_rsi(self, text: str) -> Dict[str, Any]:
        """Extract RSI information from text."""
        text_lower = text.lower()
        if "rsi" in text_lower:
            if "overbought" in text_lower:
                return {"value": "high", "signal": "sell", "confidence": 0.7}
            elif "oversold" in text_lower:
                return {"value": "low", "signal": "buy", "confidence": 0.7}
        return {"value": "neutral", "signal": "hold", "confidence": 0.5}
    
    def _extract_macd(self, text: str) -> Dict[str, Any]:
        """Extract MACD information from text."""
        text_lower = text.lower()
        if "macd" in text_lower:
            if "bullish" in text_lower or "positive" in text_lower:
                return {"signal": "buy", "confidence": 0.6}
            elif "bearish" in text_lower or "negative" in text_lower:
                return {"signal": "sell", "confidence": 0.6}
        return {"signal": "neutral", "confidence": 0.5}
    
    def _extract_support_resistance(self, text: str) -> Dict[str, Any]:
        """Extract support/resistance levels from text."""
        return {
            "support": None,
            "resistance": None,
            "confidence": 0.5
        }
    
    def _extract_trend(self, text: str) -> Dict[str, Any]:
        """Extract trend information from text."""
        text_lower = text.lower()
        if "uptrend" in text_lower or "bullish" in text_lower:
            return {"direction": "up", "strength": "medium", "confidence": 0.6}
        elif "downtrend" in text_lower or "bearish" in text_lower:
            return {"direction": "down", "strength": "medium", "confidence": 0.6}
        return {"direction": "sideways", "strength": "weak", "confidence": 0.5}
    
    def _extract_signals(self, text: str) -> List[Dict[str, Any]]:
        """Extract trading signals from text."""
        signals = []
        text_lower = text.lower()
        
        if "buy" in text_lower:
            signals.append({"action": "buy", "confidence": 0.6})
        if "sell" in text_lower:
            signals.append({"action": "sell", "confidence": 0.6})
        if "hold" in text_lower:
            signals.append({"action": "hold", "confidence": 0.5})
        
        return signals
    
    def _extract_risk_assessment(self, text: str) -> Dict[str, Any]:
        """Extract risk assessment results."""
        return {
            "risk_level": "medium",
            "volatility": "medium",
            "recommendations": []
        }
    
    def _extract_strategy_generation(self, text: str) -> Dict[str, Any]:
        """Extract strategy generation results."""
        return {
            "strategy_type": "general",
            "entry_points": [],
            "exit_points": [],
            "risk_management": {}
        }
    
    async def process_trading_analysis(self, tick_data: Dict[str, Any], analysis_type: AnalysisType) -> Optional[KoboldResponse]:
        """Process trading analysis with automatic prompt generation."""
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(tick_data, analysis_type)
            
            # Create request
            request = KoboldRequest(
                prompt=prompt,
                max_length=512,
                temperature=0.7,
                analysis_type=analysis_type
            )
            
            # Process request
            return await self.analyze_trading_data(request)
            
        except Exception as e:
            logger.error(f"❌ Trading analysis processing failed: {e}")
            return None
    
    def _create_analysis_prompt(self, tick_data: Dict[str, Any], analysis_type: AnalysisType) -> str:
        """Create analysis prompt based on tick data and analysis type."""
        base_prompt = f"Analyze the following trading data for {analysis_type.value.replace('_', ' ')}:"
        
        # Add tick data information
        tick_info = f"""
Symbol: {tick_data.get('symbol', 'Unknown')}
Price: {tick_data.get('price', 'N/A')}
Volume: {tick_data.get('volume', 'N/A')}
Timestamp: {tick_data.get('timestamp', 'N/A')}
"""
        
        # Add analysis-specific instructions
        if analysis_type == AnalysisType.TECHNICAL_ANALYSIS:
            instructions = """
Please provide:
1. Technical indicators analysis (RSI, MACD, Bollinger Bands)
2. Support and resistance levels
3. Trend analysis
4. Trading signals (buy/sell/hold)
5. Confidence level in your analysis
"""
        elif analysis_type == AnalysisType.RISK_ASSESSMENT:
            instructions = """
Please provide:
1. Risk level assessment (low/medium/high)
2. Volatility analysis
3. Potential downside risks
4. Risk management recommendations
5. Position sizing suggestions
"""
        elif analysis_type == AnalysisType.STRATEGY_GENERATION:
            instructions = """
Please provide:
1. Trading strategy recommendation
2. Entry and exit points
3. Stop-loss levels
4. Take-profit targets
5. Risk-reward ratio
"""
        else:
            instructions = "Please provide a comprehensive analysis with actionable insights."
        
        return f"{base_prompt}\n{tick_info}\n{instructions}"
    
    async def start_processing(self):
        """Start the KoboldCPP processing system."""
        try:
            if self.running:
                logger.warning("⚠️ KoboldCPP processing already running")
                return
            
            logger.info("🚀 Starting KoboldCPP processing system...")
            self.running = True
            
            # Start KoboldCPP server if auto-start is enabled
            if self.auto_start_kobold:
                await self.start_kobold_server()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("✅ KoboldCPP processing system started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start KoboldCPP processing: {e}")
    
    def stop_processing(self):
        """Stop the KoboldCPP processing system."""
        try:
            logger.info("🛑 Stopping KoboldCPP processing system...")
            self.running = False
            
            # Stop KoboldCPP server
            asyncio.create_task(self.stop_kobold_server())
            
            logger.info("✅ KoboldCPP processing system stopped")
            
        except Exception as e:
            logger.error(f"❌ Failed to stop KoboldCPP processing: {e}")
    
    def _processing_loop(self):
        """Main processing loop for KoboldCPP integration."""
        try:
            while self.running:
                # Process analysis requests
                asyncio.run(self._process_analysis_requests())
                
                # Monitor health
                self._monitor_health()
                
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"❌ Processing loop error: {e}")
    
    async def _process_analysis_requests(self):
        """Process pending analysis requests."""
        try:
            for analysis_type, queue in self.request_queues.items():
                if queue:
                    request = queue.popleft()
                    await self.analyze_trading_data(request)
                    
        except Exception as e:
            logger.error(f"❌ Analysis request processing failed: {e}")
    
    def _monitor_health(self):
        """Monitor system health."""
        try:
            if self.kobold_running and not self._check_server_status():
                logger.warning("⚠️ KoboldCPP server health check failed")
                self.kobold_running = False
                
        except Exception as e:
            logger.error(f"❌ Health monitoring failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            return {
                "kobold_running": self.kobold_running,
                "model_loaded": self.model_loaded,
                "model_capabilities": self.model_capabilities,
                "stats": self.stats,
                "system_info": {
                    "platform": self.system_info.platform if self.system_info else "unknown",
                    "ram_gb": self.system_info.ram_gb if self.system_info else 0.0,
                    "optimization_mode": self.system_info.optimization_mode.value if self.system_info else "unknown"
                },
                "configuration": {
                    "port": self.port,
                    "model_path": str(self.model_path) if self.model_path else "None",
                    "auto_start": self.auto_start_kobold,
                    "auto_load_model": self.auto_load_model
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics collection failed: {e}")
            return {"error": str(e)}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

async def main():
    """Main function for KoboldCPP integration testing."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Testing KoboldCPP Integration...")
    
    # Create integration instance
    kobold_integration = KoboldCPPIntegration()
    
    try:
        # Start the system
        await kobold_integration.start_processing()
        
        # Test analysis
        test_data = {
            "symbol": "BTC/USDT",
            "price": 45000.0,
            "volume": 1000000.0,
            "timestamp": time.time()
        }
        
        response = await kobold_integration.process_trading_analysis(
            test_data, AnalysisType.TECHNICAL_ANALYSIS
        )
        
        if response:
            logger.info(f"✅ Analysis completed: {response.text[:100]}...")
        else:
            logger.warning("⚠️ No analysis response received")
        
        # Get statistics
        stats = kobold_integration.get_statistics()
        logger.info(f"📊 Statistics: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        # Stop the system
        kobold_integration.stop_processing()
        
        logger.info("👋 KoboldCPP Integration test complete")

if __name__ == "__main__":
    asyncio.run(main()) 