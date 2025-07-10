import logging
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .bit_operations import BitOperations, create_bit_operations_api_endpoints
from .entropy_tracker import EntropyTracker, create_entropy_api_endpoints
from .pattern_utils import PatternUtils, create_pattern_utils_api_endpoints

#!/usr/bin/env python3
"""
Hash Recollection API Integration
================================

Main API integration module that brings together all hash_recollection
modules and provides a unified FastAPI interface for trading bot operations.
"""


# Import hash_recollection modules

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class PriceDataRequest(BaseModel):
    """Request model for price data analysis."""

    price_data: List[float] = Field(..., description="List of price values")
    symbol: Optional[str] = Field(None, description="Trading symbol")
    timeframe: Optional[str] = Field(None, description="Timeframe")


class AnalysisRequest(BaseModel):
    """Request model for comprehensive analysis."""

    price_data: List[float] = Field(..., description="List of price values")
    include_entropy: bool = Field(True, description="Include entropy analysis")
    include_bit_ops: bool = Field(True, description="Include bit operations")
    include_patterns: bool = Field(True, description="Include pattern analysis")
    symbol: Optional[str] = Field(None, description="Trading symbol")


class SignalRequest(BaseModel):
    """Request model for signal generation."""

    price_data: List[float] = Field(..., description="List of price values")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")
    symbol: Optional[str] = Field(None, description="Trading symbol")


class HashRecollectionAPI:
    """
    Main API class for hash_recollection integration.

    Provides unified access to all hash_recollection modules
    through a single FastAPI application.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the API."""
        self.config = config or {}

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Hash Recollection Trading API",
            description="Unified API for hash_recollection trading bot modules",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize modules
        self.entropy_tracker = EntropyTracker()
        self.bit_operations = BitOperations()
        self.pattern_utils = PatternUtils()

        # Setup routes
        self._setup_routes()

        logger.info("🚀 Hash Recollection API initialized")

    def _setup_routes(self):
        """Setup all API routes."""

        # Health check
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "hash_recollection_api",
                "version": "1.0.0",
                "modules": {
                    "entropy_tracker": "active",
                    "bit_operations": "active",
                    "pattern_utils": "active",
                },
            }

        # Comprehensive analysis endpoint
        @self.app.post("/analyze")
        async def comprehensive_analysis(request: AnalysisRequest):
            """Perform comprehensive analysis using all modules."""
            try:
                result = {
                    "success": True,
                    "timestamp": time.time(),
                    "symbol": request.symbol,
                    "data_points": len(request.price_data),
                }
                if request.include_entropy:
                    entropy_metrics = self.entropy_tracker.calculate_entropy(request.price_data)
                    result["entropy"] = {
                        "entropy_value": entropy_metrics.entropy_value,
                        "state": entropy_metrics.state.value,
                        "confidence": entropy_metrics.confidence,
                        "trend_direction": entropy_metrics.trend_direction,
                        "volatility_factor": entropy_metrics.volatility_factor,
                        "pattern_strength": entropy_metrics.pattern_strength,
                    }
                if request.include_bit_ops:
                    # Create bit sequence from price data
                    normalized_prices = [
                        (p - min(request.price_data)) / (max(request.price_data) - min(request.price_data))
                        for p in request.price_data
                    ]
                    bit_sequence = self.bit_operations.create_bit_sequence(normalized_prices)
                    patterns = self.bit_operations.detect_patterns(bit_sequence.bits)

                    result["bit_operations"] = {
                        "sequence_id": bit_sequence.sequence_id,
                        "bits": bit_sequence.bits,
                        "patterns_found": len(patterns),
                        "patterns": [
                            {
                                "pattern_id": p.pattern_id,
                                "pattern_type": p.pattern_type,
                                "confidence": p.confidence,
                            }
                            for p in patterns
                        ],
                    }
                if request.include_patterns:
                    trend = self.pattern_utils.analyze_trend(request.price_data)
                    patterns = self.pattern_utils.detect_patterns(request.price_data)

                    result["pattern_analysis"] = {
                        "trend": {
                            "direction": trend.trend_direction,
                            "strength": trend.strength,
                            "slope": trend.slope,
                            "r_squared": trend.r_squared,
                        },
                        "patterns_found": len(patterns),
                        "patterns": [
                            {
                                "pattern_id": p.pattern_id,
                                "pattern_type": p.pattern_type.value,
                                "confidence": p.confidence,
                            }
                            for p in patterns
                        ],
                    }
                return result

            except Exception as e:
                logger.error(f"Error in comprehensive analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Signal generation endpoint
        @self.app.post("/signal")
        async def generate_signal(request: SignalRequest):
            """Generate trading signal based on all analyses."""
            try:
                # Get entropy signal
                entropy_signal = self.entropy_tracker.generate_signal(request.price_data)

                # Analyze patterns
                trend = self.pattern_utils.analyze_trend(request.price_data)
                patterns = self.pattern_utils.detect_patterns(request.price_data)

                # Create bit sequence
                normalized_prices = [
                    (p - min(request.price_data)) / (max(request.price_data) - min(request.price_data))
                    for p in request.price_data
                ]
                bit_sequence = self.bit_operations.create_bit_sequence(normalized_prices)
                bit_patterns = self.bit_operations.detect_patterns(bit_sequence.bits)

                # Combine signals
                signal_strength = 0.0
                signal_type = "hold"
                confidence = 0.0

                # Entropy contribution
                if entropy_signal and entropy_signal.confidence >= request.confidence_threshold:
                    if entropy_signal.signal_type == "buy":
                        signal_strength += entropy_signal.strength * 0.4
                    elif entropy_signal.signal_type == "sell":
                        signal_strength -= entropy_signal.strength * 0.4
                    confidence += entropy_signal.confidence * 0.4

                # Pattern contribution
                if trend.strength > 0.6:
                    if trend.trend_direction == "up":
                        signal_strength += trend.strength * 0.3
                    elif trend.trend_direction == "down":
                        signal_strength -= trend.strength * 0.3
                    confidence += trend.strength * 0.3

                # Bit pattern contribution
                if bit_patterns:
                    avg_confidence = sum(p.confidence for p in bit_patterns) / len(bit_patterns)
                    confidence += avg_confidence * 0.3

                # Determine final signal
                if signal_strength > 0.3:
                    signal_type = "buy"
                elif signal_strength < -0.3:
                    signal_type = "sell"
                else:
                    signal_type = "hold"

                return {
                    "success": True,
                    "signal": {
                        "type": signal_type,
                        "strength": abs(signal_strength),
                        "confidence": min(confidence, 1.0),
                        "timestamp": time.time(),
                    },
                    "analysis": {
                        "entropy_signal": {
                            "type": (entropy_signal.signal_type if entropy_signal else "none"),
                            "strength": (entropy_signal.strength if entropy_signal else 0.0),
                            "confidence": (entropy_signal.confidence if entropy_signal else 0.0),
                        },
                        "trend": {
                            "direction": trend.trend_direction,
                            "strength": trend.strength,
                        },
                        "patterns_found": len(patterns) + len(bit_patterns),
                    },
                }
            except Exception as e:
                logger.error(f"Error generating signal: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Module-specific endpoints
        create_entropy_api_endpoints(self.app)
        create_bit_operations_api_endpoints(self.app)
        create_pattern_utils_api_endpoints(self.app)

        # Summary endpoints
        @self.app.get("/summary")
        async def get_system_summary():
            """Get summary of all modules."""
            try:
                return {
                    "success": True,
                    "summary": {
                        "entropy_tracker": self.entropy_tracker.get_entropy_summary(),
                        "bit_operations": self.bit_operations.get_bit_summary(),
                        "pattern_utils": self.pattern_utils.get_pattern_summary(),
                    },
                }
            except Exception as e:
                logger.error(f"Error getting system summary: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/status")
        async def get_system_status():
            """Get system status."""
            return {
                "status": "operational",
                "modules": {
                    "entropy_tracker": "active",
                    "bit_operations": "active",
                    "pattern_utils": "active",
                },
                "total_operations": {
                    "entropy_calculations": self.entropy_tracker.total_entropy_calculations,
                    "bit_operations": self.bit_operations.total_bit_operations,
                    "patterns_analyzed": self.pattern_utils.total_patterns_analyzed,
                },
            }

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the API server."""
        logger.info(f"Starting Hash Recollection API on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, debug=debug, log_level="info")


# Convenience function to create and run the API
def create_and_run_api(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Create and run the Hash Recollection API."""
    api = HashRecollectionAPI()
    api.run(host=host, port=port, debug=debug)


if __name__ == "__main__":

    create_and_run_api()
