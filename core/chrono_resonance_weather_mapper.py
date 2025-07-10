#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chrono Resonance Weather Mapper (CRWM) - Advanced Temporal Analysis System
==========================================================================

A sophisticated weather-entropy fusion system that maps atmospheric conditions 
to market volatility through mathematical resonance analysis and geo-located 
entropy triggers.

Mathematical Foundation:
E_CRWF(t,φ,λ,h) = α∇T(t,φ,λ) + β∇P(t,φ,λ) + γ⋅Ω(t,φ,λ,h)

Where:
- φ, λ: Latitude & Longitude
- h: Altitude / pressure-derived elevation
- ∇T: Temporal temperature gradient
- ∇P: Barometric pressure gradient
- Ω(t,...): Schumann + geomagnetic interference function
- α, β, γ: Tunable weights for resonance-driven signal dampening

Full Model:
∇Φ(t,x,y,z) + δτΨ(t) = Σₙ₌₀^∞ ωₙ⋅sin(kₙ⋅r−ωₙ⋅t+φₙ)

Where:
- ∇Φ(t,x,y,z): Spatial gradient of atmospheric scalar field
- δτΨ(t): Temporal resonance distortion
- ωₙ: Frequency coefficients for Schumann + Solar resonance
- kₙ: Wave vector component
- r: Radial Earth distance vector
- φₙ: Phase offset per harmonic index
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Import dependencies
try:
    from core.math_cache import MathResultCache
    from core.math_config_manager import MathConfigManager
    from core.math_orchestrator import MathOrchestrator
    MATH_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    MATH_INFRASTRUCTURE_AVAILABLE = False
    logger.warning("Math infrastructure not available")


class Status(Enum):
    """System status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"


class ResonanceMode(Enum):
    """Chrono-resonance analysis modes."""
    HARMONIC = "harmonic"
    SUBHARMONIC = "subharmonic"
    SCHUMANN = "schumann"
    SOLAR = "solar"
    GEOMAGNETIC = "geomagnetic"


@dataclass
class WeatherDataPoint:
    """Weather data point with temporal and spatial information."""
    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    temperature: float
    pressure: float
    humidity: float
    wind_speed: float
    wind_direction: float


@dataclass
class ResonanceSignature:
    """Weather-price resonance signature."""
    timestamp: float
    frequency: float
    amplitude: float
    phase: float
    correlation: float
    confidence: float
    resonance_mode: ResonanceMode
    harmonic_order: int = 1


@dataclass
class AtmosphericGradient:
    """Atmospheric gradient analysis."""
    timestamp: float
    pressure_gradient: float
    temperature_gradient: float
    humidity_gradient: float
    wind_gradient: float
    composite_gradient: float
    gradient_direction: float  # degrees


@dataclass
class WeatherPriceCorrelation:
    """Weather-price correlation result."""
    timestamp: float
    correlation_coefficient: float
    significance_level: float
    confidence_interval: Tuple[float, float]
    sample_size: int


class ChronoResonanceWeatherMapper:
    """
    ChronoResonance Weather Mapping system.
    
    Provides advanced temporal analysis focusing on field-level time-resonance,
    macro-patterns, harmonics, gradients, and phase shifts in market "weather."
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the chrono resonance weather mapper."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.active = False
        self.initialized = False
        
        # Core mathematical parameters
        self.schumann_frequency = 7.83  # Hz - Earth's fundamental resonance
        self.temporal_decay = 0.95
        self.spatial_resolution = 0.1  # degrees
        
        # Data storage
        self.weather_cache: Dict[str, WeatherDataPoint] = {}
        self.resonance_signatures: List[ResonanceSignature] = []
        self.gradient_history: List[AtmosphericGradient] = []
        self.correlation_cache: Dict[str, WeatherPriceCorrelation] = {}
        
        # Resonance frequencies
        self.resonance_frequencies = self._initialize_resonance_frequencies()
        
        # Initialize math infrastructure if available
        if MATH_INFRASTRUCTURE_AVAILABLE:
            self.math_config = MathConfigManager()
            self.math_cache = MathResultCache()
            self.math_orchestrator = MathOrchestrator()
        
        self._initialize_system()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enabled': True,
            'timeout': 30.0,
            'retries': 3,
            'debug': False,
            'log_level': 'INFO',
            'weather_api_enabled': True,
            'correlation_window_hours': 168,  # 1 week
            'resonance_analysis_enabled': True,
            'gradient_smoothing': True,
            'schumann_weight': 0.4,
            'solar_weight': 0.3,
            'geomagnetic_weight': 0.3,
        }

    def _initialize_resonance_frequencies(self) -> Dict[str, float]:
        """Initialize resonance frequency analysis parameters."""
        return {
            "atmospheric_base": 11.78,  # Earth's Schumann resonance (Hz)
            "diurnal_cycle": 1 / (24 * 3600),  # Daily cycle
            "lunar_cycle": 1 / (29.5 * 24 * 3600),  # Lunar cycle
            "humidity_cycle": 0.25,  # Humidity cycle
            "pressure_cycle": 0.5,  # Pressure cycle
            "market_sentiment": 3.14159,  # Market resonance (π Hz)
        }

    def _initialize_system(self) -> None:
        """Initialize the system."""
        try:
            self.logger.info("Initializing ChronoResonance Weather Mapper")
            self.initialized = True
            self.logger.info("✅ ChronoResonance Weather Mapper initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error initializing ChronoResonance Weather Mapper: {e}")
            self.initialized = False

    def activate(self) -> bool:
        """Activate the system."""
        if not self.initialized:
            self.logger.error("System not initialized")
            return False

        try:
            self.active = True
            self.logger.info("✅ ChronoResonance Weather Mapper activated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error activating ChronoResonance Weather Mapper: {e}")
            return False

    def deactivate(self) -> bool:
        """Deactivate the system."""
        try:
            self.active = False
            self.logger.info("✅ ChronoResonance Weather Mapper deactivated")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error deactivating ChronoResonance Weather Mapper: {e}")
            return False

    def compute_crwf(self, t: float, phi: float, lambda_val: float, h: float) -> float:
        """
        Compute CRWF: E_CRWF(t,φ,λ,h) = α∇T(t,φ,λ) + β∇P(t,φ,λ) + γ⋅Ω(t,φ,λ,h)
        
        Args:
            t: Time parameter
            phi: Latitude
            lambda_val: Longitude
            h: Altitude/pressure-derived elevation
            
        Returns:
            CRWF value representing weather-entropy fusion
        """
        try:
            # Compute individual components
            temp_gradient = self._compute_temperature_gradient(t, phi, lambda_val)
            pressure_gradient = self._compute_pressure_gradient(t, phi, lambda_val)
            schumann_interference = self._compute_schumann_interference(t, phi, lambda_val, h)
            
            # Apply weights from configuration
            alpha = self.config.get('schumann_weight', 0.4)
            beta = self.config.get('solar_weight', 0.3)
            gamma = self.config.get('geomagnetic_weight', 0.3)
            
            # Compute CRWF
            crwf = alpha * temp_gradient + beta * pressure_gradient + gamma * schumann_interference
            
            self.logger.debug(f"CRWF computed: {crwf:.6f} (t={t:.2f}, φ={phi:.2f}, λ={lambda_val:.2f}, h={h:.2f})")
            return crwf
            
        except Exception as e:
            self.logger.error(f"Error computing CRWF: {e}")
            return 0.0

    def _compute_temperature_gradient(self, t: float, phi: float, lambda_val: float) -> float:
        """Compute temporal temperature gradient ∇T(t,φ,λ)."""
        try:
            # Base temperature variation with time
            base_temp = 15.0 + 10.0 * np.sin(2 * np.pi * t / (24 * 3600))  # Daily cycle
            
            # Latitude effect
            lat_effect = np.cos(np.radians(phi)) * 20.0
            
            # Longitude effect (simplified)
            lon_effect = np.sin(np.radians(lambda_val)) * 5.0
            
            # Temporal gradient
            temp_gradient = base_temp + lat_effect + lon_effect
            
            return temp_gradient
            
        except Exception as e:
            self.logger.error(f"Error computing temperature gradient: {e}")
            return 0.0

    def _compute_pressure_gradient(self, t: float, phi: float, lambda_val: float) -> float:
        """Compute barometric pressure gradient ∇P(t,φ,λ)."""
        try:
            # Base pressure (1013.25 hPa at sea level)
            base_pressure = 1013.25
            
            # Altitude effect (simplified)
            altitude_effect = -0.12  # hPa per meter (approximate)
            
            # Temporal variation
            temporal_variation = 5.0 * np.sin(2 * np.pi * t / (12 * 3600))  # Semi-diurnal
            
            # Latitude effect
            lat_effect = np.sin(np.radians(phi)) * 10.0
            
            pressure_gradient = base_pressure + altitude_effect + temporal_variation + lat_effect
            
            return pressure_gradient
            
        except Exception as e:
            self.logger.error(f"Error computing pressure gradient: {e}")
            return 1013.25

    def _compute_schumann_interference(self, t: float, phi: float, lambda_val: float, h: float) -> float:
        """Compute Schumann + geomagnetic interference function Ω(t,φ,λ,h)."""
        try:
            # Schumann resonance component
            schumann_component = np.sin(2 * np.pi * self.schumann_frequency * t)
            
            # Geomagnetic component
            geomagnetic_component = np.cos(2 * np.pi * 0.1 * t)  # Low frequency geomagnetic
            
            # Altitude effect
            altitude_effect = np.exp(-h / 8500.0)  # Exponential decay with altitude
            
            # Latitude effect (stronger at poles)
            lat_effect = 1.0 + 0.5 * np.cos(np.radians(phi))
            
            # Combine components
            interference = (schumann_component + geomagnetic_component) * altitude_effect * lat_effect
            
            return interference
            
        except Exception as e:
            self.logger.error(f"Error computing Schumann interference: {e}")
            return 0.0

    def compute_full_model(self, t: float, x: float, y: float, z: float) -> float:
        """
        Compute full model: ∇Φ(t,x,y,z) + δτΨ(t) = Σₙ₌₀^∞ ωₙ⋅sin(kₙ⋅r−ωₙ⋅t+φₙ)
        
        Args:
            t: Time parameter
            x, y, z: Spatial coordinates
            
        Returns:
            Full model value
        """
        try:
            # Spatial gradient of atmospheric scalar field
            spatial_gradient = self._compute_spatial_gradient(x, y, z)
            
            # Temporal resonance distortion
            temporal_distortion = self._compute_temporal_distortion(t)
            
            # Harmonic series
            harmonic_sum = 0.0
            for n in range(5):  # Sum first 5 harmonics
                omega_n = self.resonance_frequencies.get(f"harmonic_{n}", 1.0)
                k_n = 2 * np.pi / (1000 + n * 100)  # Wave vector
                r = np.sqrt(x**2 + y**2 + z**2)  # Radial distance
                phi_n = n * np.pi / 4  # Phase offset
                
                harmonic_term = omega_n * np.sin(k_n * r - omega_n * t + phi_n)
                harmonic_sum += harmonic_term
            
            # Full model
            full_model = spatial_gradient + temporal_distortion + harmonic_sum
            
            return full_model
            
        except Exception as e:
            self.logger.error(f"Error computing full model: {e}")
            return 0.0

    def _compute_spatial_gradient(self, x: float, y: float, z: float) -> float:
        """Compute spatial gradient of atmospheric scalar field ∇Φ(t,x,y,z)."""
        try:
            # Simplified spatial gradient
            gradient_x = np.cos(x / 1000.0)
            gradient_y = np.sin(y / 1000.0)
            gradient_z = np.exp(-z / 10000.0)
            
            return gradient_x + gradient_y + gradient_z
            
        except Exception as e:
            self.logger.error(f"Error computing spatial gradient: {e}")
            return 0.0

    def _compute_temporal_distortion(self, t: float) -> float:
        """Compute temporal resonance distortion δτΨ(t)."""
        try:
            # Temporal distortion with multiple frequencies
            distortion = 0.0
            
            # Diurnal distortion
            distortion += 0.1 * np.sin(2 * np.pi * t / (24 * 3600))
            
            # Semi-diurnal distortion
            distortion += 0.05 * np.sin(2 * np.pi * t / (12 * 3600))
            
            # Schumann distortion
            distortion += 0.02 * np.sin(2 * np.pi * self.schumann_frequency * t)
            
            return distortion
            
        except Exception as e:
            self.logger.error(f"Error computing temporal distortion: {e}")
            return 0.0

    def analyze_weather_patterns(self, weather_data: List[WeatherDataPoint]) -> List[ResonanceSignature]:
        """Analyze weather patterns for resonance signatures."""
        try:
            signatures = []
            
            for data_point in weather_data:
                # Compute CRWF for this data point
                crwf = self.compute_crwf(
                    data_point.timestamp,
                    data_point.latitude,
                    data_point.longitude,
                    data_point.altitude
                )
                
                # Analyze resonance modes
                for mode in ResonanceMode:
                    frequency = self._get_resonance_frequency(mode)
                    amplitude = self._compute_resonance_amplitude(data_point, mode)
                    phase = self._compute_resonance_phase(data_point.timestamp, frequency)
                    correlation = self._compute_correlation(data_point, mode)
                    confidence = self._compute_confidence(data_point, mode)
                    
                    signature = ResonanceSignature(
                        timestamp=data_point.timestamp,
                        frequency=frequency,
                        amplitude=amplitude,
                        phase=phase,
                        correlation=correlation,
                        confidence=confidence,
                        resonance_mode=mode
                    )
                    
                    signatures.append(signature)
            
            self.resonance_signatures.extend(signatures)
            return signatures
            
        except Exception as e:
            self.logger.error(f"Error analyzing weather patterns: {e}")
            return []

    def _get_resonance_frequency(self, mode: ResonanceMode) -> float:
        """Get resonance frequency for given mode."""
        frequency_map = {
            ResonanceMode.HARMONIC: 1.0,
            ResonanceMode.SUBHARMONIC: 0.5,
            ResonanceMode.SCHUMANN: self.schumann_frequency,
            ResonanceMode.SOLAR: 0.1,
            ResonanceMode.GEOMAGNETIC: 0.01,
        }
        return frequency_map.get(mode, 1.0)

    def _compute_resonance_amplitude(self, data_point: WeatherDataPoint, mode: ResonanceMode) -> float:
        """Compute resonance amplitude for given mode and data point."""
        try:
            base_amplitude = 1.0
            
            if mode == ResonanceMode.SCHUMANN:
                # Schumann amplitude depends on atmospheric conditions
                base_amplitude = data_point.pressure / 1013.25
            elif mode == ResonanceMode.SOLAR:
                # Solar amplitude depends on time of day
                hour = datetime.fromtimestamp(data_point.timestamp).hour
                base_amplitude = np.sin(np.pi * hour / 12.0)
            elif mode == ResonanceMode.GEOMAGNETIC:
                # Geomagnetic amplitude depends on latitude
                base_amplitude = np.cos(np.radians(data_point.latitude))
            
            return base_amplitude
            
        except Exception as e:
            self.logger.error(f"Error computing resonance amplitude: {e}")
            return 1.0

    def _compute_resonance_phase(self, timestamp: float, frequency: float) -> float:
        """Compute resonance phase."""
        try:
            return 2 * np.pi * frequency * timestamp
        except Exception as e:
            self.logger.error(f"Error computing resonance phase: {e}")
            return 0.0

    def _compute_correlation(self, data_point: WeatherDataPoint, mode: ResonanceMode) -> float:
        """Compute correlation between weather data and resonance mode."""
        try:
            # Simplified correlation calculation
            if mode == ResonanceMode.SCHUMANN:
                return data_point.pressure / 1013.25
            elif mode == ResonanceMode.SOLAR:
                return data_point.temperature / 300.0  # Normalized temperature
            else:
                return 0.5  # Default correlation
        except Exception as e:
            self.logger.error(f"Error computing correlation: {e}")
            return 0.0

    def _compute_confidence(self, data_point: WeatherDataPoint, mode: ResonanceMode) -> float:
        """Compute confidence level for resonance analysis."""
        try:
            # Confidence based on data quality and mode
            base_confidence = 0.8
            
            if mode == ResonanceMode.SCHUMANN:
                # Higher confidence for Schumann resonance
                base_confidence = 0.9
            elif mode == ResonanceMode.GEOMAGNETIC:
                # Lower confidence for geomagnetic (more variable)
                base_confidence = 0.6
            
            return base_confidence
            
        except Exception as e:
            self.logger.error(f"Error computing confidence: {e}")
            return 0.5

    def compute_atmospheric_gradients(self, weather_data: List[WeatherDataPoint]) -> List[AtmosphericGradient]:
        """Compute atmospheric gradients from weather data."""
        try:
            gradients = []
            
            for i, data_point in enumerate(weather_data):
                if i == 0:
                    continue
                
                prev_point = weather_data[i - 1]
                time_diff = data_point.timestamp - prev_point.timestamp
                
                # Compute gradients
                pressure_gradient = (data_point.pressure - prev_point.pressure) / time_diff
                temperature_gradient = (data_point.temperature - prev_point.temperature) / time_diff
                humidity_gradient = (data_point.humidity - prev_point.humidity) / time_diff
                wind_gradient = (data_point.wind_speed - prev_point.wind_speed) / time_diff
                
                # Composite gradient
                composite_gradient = np.sqrt(
                    pressure_gradient**2 + 
                    temperature_gradient**2 + 
                    humidity_gradient**2 + 
                    wind_gradient**2
                )
                
                # Gradient direction
                gradient_direction = np.degrees(np.arctan2(
                    data_point.latitude - prev_point.latitude,
                    data_point.longitude - prev_point.longitude
                ))
                
                gradient = AtmosphericGradient(
                    timestamp=data_point.timestamp,
                    pressure_gradient=pressure_gradient,
                    temperature_gradient=temperature_gradient,
                    humidity_gradient=humidity_gradient,
                    wind_gradient=wind_gradient,
                    composite_gradient=composite_gradient,
                    gradient_direction=gradient_direction
                )
                
                gradients.append(gradient)
            
            self.gradient_history.extend(gradients)
            return gradients
            
        except Exception as e:
            self.logger.error(f"Error computing atmospheric gradients: {e}")
            return []

    def correlate_weather_price(self, weather_data: List[WeatherDataPoint], 
                              price_data: List[float]) -> WeatherPriceCorrelation:
        """Correlate weather data with price movements."""
        try:
            if len(weather_data) != len(price_data) or len(weather_data) < 2:
                raise ValueError("Weather and price data must have same length and at least 2 points")
            
            # Extract relevant weather features
            temperatures = [point.temperature for point in weather_data]
            pressures = [point.pressure for point in weather_data]
            humidities = [point.humidity for point in weather_data]
            
            # Compute price changes
            price_changes = np.diff(price_data)
            
            # Correlate with each weather feature
            temp_corr = np.corrcoef(temperatures[:-1], price_changes)[0, 1]
            pressure_corr = np.corrcoef(pressures[:-1], price_changes)[0, 1]
            humidity_corr = np.corrcoef(humidities[:-1], price_changes)[0, 1]
            
            # Composite correlation
            correlation_coefficient = (temp_corr + pressure_corr + humidity_corr) / 3.0
            
            # Significance level (simplified)
            significance_level = 0.05 if abs(correlation_coefficient) > 0.3 else 0.1
            
            # Confidence interval (simplified)
            confidence_interval = (
                correlation_coefficient - 0.1,
                correlation_coefficient + 0.1
            )
            
            correlation = WeatherPriceCorrelation(
                timestamp=weather_data[-1].timestamp,
                correlation_coefficient=correlation_coefficient,
                significance_level=significance_level,
                confidence_interval=confidence_interval,
                sample_size=len(weather_data)
            )
            
            # Cache the result
            cache_key = f"{weather_data[0].timestamp}_{weather_data[-1].timestamp}"
            self.correlation_cache[cache_key] = correlation
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Error correlating weather and price: {e}")
            return WeatherPriceCorrelation(
                timestamp=time.time(),
                correlation_coefficient=0.0,
                significance_level=1.0,
                confidence_interval=(0.0, 0.0),
                sample_size=0
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'active': self.active,
            'initialized': self.initialized,
            'config': self.config,
            'resonance_signatures_count': len(self.resonance_signatures),
            'gradient_history_count': len(self.gradient_history),
            'correlation_cache_size': len(self.correlation_cache),
            'weather_cache_size': len(self.weather_cache),
            'schumann_frequency': self.schumann_frequency,
            'temporal_decay': self.temporal_decay,
            'spatial_resolution': self.spatial_resolution,
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.weather_cache.clear()
        self.correlation_cache.clear()
        self.logger.info("Weather mapper caches cleared")

    def get_recent_resonance_signatures(self, count: int = 10) -> List[ResonanceSignature]:
        """Get recent resonance signatures."""
        return self.resonance_signatures[-count:] if self.resonance_signatures else []

    def get_recent_gradients(self, count: int = 10) -> List[AtmosphericGradient]:
        """Get recent atmospheric gradients."""
        return self.gradient_history[-count:] if self.gradient_history else []


# Factory function
def create_chrono_resonance_weather_mapper(config: Optional[Dict[str, Any]] = None):
    """Create a chrono resonance weather mapper instance."""
    return ChronoResonanceWeatherMapper(config)
