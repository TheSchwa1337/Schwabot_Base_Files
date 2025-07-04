#!/usr/bin/env python3
"""
Chrono Resonance Weather Mapping (CRWF) - Geo-Located Entropy Trigger System (GETS)

A sophisticated weather-entropy fusion system that maps atmospheric conditions to market volatility
through mathematical resonance analysis and geo-located entropy triggers.

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

import asyncio
import math
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import requests
from scipy import signal
from scipy.fft import fft, fftfreq

from .chrono_recursive_logic_function import CRLFResponse, CRLFTriggerState
from .zpe_zbe_core import ZPEVector, ZBEBalance

logger = logging.getLogger(__name__)


class WeatherPattern(Enum):
    """Weather pattern types for CRWF analysis."""
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    ATMOSPHERIC_STABILITY = "atmospheric_stability"
    WEATHER_TRANSITION = "weather_transition"
    STORM_FRONT = "storm_front"
    GEOMAGNETIC_STORM = "geomagnetic_storm"


class ResonanceMode(Enum):
    """Chrono-resonance analysis modes."""
    HARMONIC = "harmonic"
    SUBHARMONIC = "subharmonic"
    OVERTONE = "overtone"
    FUNDAMENTAL = "fundamental"
    CHAOS = "chaos"


@dataclass
class WeatherDataPoint:
    """Individual weather measurement with CRWF analysis."""
    
    timestamp: datetime
    latitude: float
    longitude: float
    altitude: float
    
    # Core weather data
    temperature: float  # Celsius
    pressure: float    # hPa
    humidity: float    # %
    wind_speed: float  # m/s
    wind_direction: float  # degrees

    # Advanced weather data
    schumann_frequency: float = 7.83  # Hz (default Schumann resonance)
    geomagnetic_index: float = 0.0    # Kp index
    solar_flux: float = 100.0         # Solar flux units

    # CRWF computed values
    temperature_gradient: float = 0.0
    pressure_gradient: float = 0.0
    entropy_score: float = 0.0
    resonance_strength: float = 0.0

    # Metadata
    weather_type: str = "unknown"
    source: str = "api"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeoLocation:
    """Geographic location with resonance properties."""
    
    latitude: float
    longitude: float
    altitude: float = 0.0
    name: str = ""
    
    # Resonance properties
    ley_line_strength: float = 0.0
    geomagnetic_density: float = 0.0
    schumann_resonance: float = 7.83
    
    # CRWF computed values
    entropy_zone_multiplier: float = 1.0
    resonance_factor: float = 1.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRWFResponse:
    """Response from CRWF computation."""
    
    # Core CRWF output
    crwf_output: float
    entropy_score: float
    resonance_strength: float
    
    # Weather analysis
    weather_pattern: WeatherPattern
    temperature_gradient: float
    pressure_gradient: float
    
    # Geo-resonance analysis
    geo_alignment_score: float
    ley_line_resonance: float
    geomagnetic_factor: float
    
    # Temporal analysis
    temporal_resonance: float
    phase_alignment: float
    
    # Integration data
    crlf_adjustment_factor: float
    market_entropy_adjustment: float
    
    # Metadata
    timestamp: datetime
    location: GeoLocation
    recommendations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeatherSignature:
    """Weather-price resonance signature."""
    
    frequency: float
    amplitude: float
    phase: float
    pattern_type: WeatherPattern
    resonance_mode: ResonanceMode
    confidence: float
    timestamp: datetime
    location: GeoLocation
    
    # CRWF analysis
    entropy_contribution: float
    market_correlation: float
    prediction_horizon: int  # hours
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeatherPriceCorrelation:
    """Weather-price correlation result."""
    
    correlation_coefficient: float
    significance_level: float
    time_lag: int  # hours
    weather_factor: str
    price_factor: str
    sample_size: int
    confidence_interval: Tuple[float, float]

    # CRWF enhanced
    entropy_weighted_correlation: float
    resonance_adjusted_correlation: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChronoResonanceWeatherMapper:
    """
    ChronoResonance Weather Mapping system with geo-located entropy triggers.
    
    Implements the full CRWF mathematical model:
    E_CRWF(t,φ,λ,h) = α∇T(t,φ,λ) + β∇P(t,φ,λ) + γ⋅Ω(t,φ,λ,h)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the CRWF system."""
        self.api_key = api_key
        self.weather_history: List[WeatherDataPoint] = []
        self.location_cache: Dict[str, GeoLocation] = {}
        
        # CRWF parameters
        self.alpha = 0.4  # Temperature gradient weight
        self.beta = 0.4   # Pressure gradient weight
        self.gamma = 0.2  # Schumann/geomagnetic weight
        
        # Resonance parameters
        self.schumann_frequencies = [7.83, 14.3, 20.8, 27.3, 33.8]  # Hz
        self.geomagnetic_threshold = 5.0  # Kp index threshold
        
        # Performance tracking
        self.computation_history: List[CRWFResponse] = []
        self.correlation_history: List[WeatherPriceCorrelation] = []
        
        logger.info("🌤️ ChronoResonance Weather Mapper initialized")
    
    def compute_crwf(
        self,
        weather_data: WeatherDataPoint,
        location: GeoLocation,
        market_entropy: float = 0.5
    ) -> CRWFResponse:
        """
        Compute the ChronoResonance Weather Function.
        
        Args:
            weather_data: Current weather data point
            location: Geographic location with resonance properties
            market_entropy: Current market entropy level
            
        Returns:
            CRWFResponse with computed weather-entropy analysis
        """
        try:
            # Compute temperature gradient ∇T(t,φ,λ)
            temp_gradient = self._compute_temperature_gradient(weather_data)
            
            # Compute pressure gradient ∇P(t,φ,λ)
            pressure_gradient = self._compute_pressure_gradient(weather_data)
            
            # Compute Schumann/geomagnetic interference Ω(t,φ,λ,h)
            schumann_interference = self._compute_schumann_interference(
                weather_data, location
            )
            
            # Compute CRWF output: E_CRWF = α∇T + β∇P + γ⋅Ω
            crwf_output = (
                self.alpha * temp_gradient +
                self.beta * pressure_gradient +
                self.gamma * schumann_interference
            )
            
            # Compute entropy score
            entropy_score = self._compute_entropy_score(weather_data, location)
            
            # Compute resonance strength
            resonance_strength = self._compute_resonance_strength(
                weather_data, location, crwf_output
            )
            
            # Determine weather pattern
            weather_pattern = self._determine_weather_pattern(
                weather_data, crwf_output
            )
            
            # Compute geo-alignment score
            geo_alignment = self._compute_geo_alignment(location, weather_data)
            
            # Compute temporal resonance
            temporal_resonance = self._compute_temporal_resonance(
                weather_data, location
            )
            
            # Compute CRLF adjustment factor
            crlf_adjustment = self._compute_crlf_adjustment(
                crwf_output, market_entropy, entropy_score
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                crwf_output, weather_pattern, entropy_score
            )
            
            # Create response
            response = CRWFResponse(
                crwf_output=crwf_output,
                entropy_score=entropy_score,
                resonance_strength=resonance_strength,
                weather_pattern=weather_pattern,
                temperature_gradient=temp_gradient,
                pressure_gradient=pressure_gradient,
                geo_alignment_score=geo_alignment['alignment_score'],
                ley_line_resonance=geo_alignment['ley_line_resonance'],
                geomagnetic_factor=geo_alignment['geomagnetic_factor'],
                temporal_resonance=temporal_resonance,
                phase_alignment=geo_alignment['phase_alignment'],
                crlf_adjustment_factor=crlf_adjustment,
                market_entropy_adjustment=entropy_score - market_entropy,
                timestamp=weather_data.timestamp,
                location=location,
                recommendations=recommendations
            )
            
            # Store in history
            self.computation_history.append(response)
            self.weather_history.append(weather_data)
            
            # Keep history manageable
            max_history = 1000
            if len(self.computation_history) > max_history:
                self.computation_history = self.computation_history[-max_history:]
                self.weather_history = self.weather_history[-max_history:]
            
            logger.debug(f"CRWF computed: {crwf_output:.4f}, Entropy: {entropy_score:.3f}")
            
            return response

        except Exception as e:
            logger.error(f"Error computing CRWF: {e}")
            return self._create_fallback_response(weather_data, location)
    
    def _compute_temperature_gradient(self, weather_data: WeatherDataPoint) -> float:
        """Compute temporal temperature gradient ∇T(t,φ,λ)."""
        if len(self.weather_history) < 2:
            return 0.0
        
        # Get recent temperature history for this location
        recent_temps = [
            w.temperature for w in self.weather_history[-24:]  # Last 24 hours
            if abs(w.latitude - weather_data.latitude) < 0.1 and
               abs(w.longitude - weather_data.longitude) < 0.1
        ]
        
        if len(recent_temps) < 2:
                return 0.0

        # Compute gradient using finite difference
        temp_gradient = np.gradient(recent_temps)
        return float(np.mean(temp_gradient))
    
    def _compute_pressure_gradient(self, weather_data: WeatherDataPoint) -> float:
        """Compute barometric pressure gradient ∇P(t,φ,λ)."""
        if len(self.weather_history) < 2:
            return 0.0

        # Get recent pressure history for this location
        recent_pressures = [
            w.pressure for w in self.weather_history[-24:]  # Last 24 hours
            if abs(w.latitude - weather_data.latitude) < 0.1 and
               abs(w.longitude - weather_data.longitude) < 0.1
        ]
        
        if len(recent_pressures) < 2:
            return 0.0
        
        # Compute gradient using finite difference
        pressure_gradient = np.gradient(recent_pressures)
        return float(np.mean(pressure_gradient))
    
    def _compute_schumann_interference(
        self,
        weather_data: WeatherDataPoint,
        location: GeoLocation
    ) -> float:
        """
        Compute Schumann + geomagnetic interference function Ω(t,φ,λ,h).
        
        This function models the interference between Schumann resonances
        and geomagnetic activity at the given location.
        """
        # Base Schumann resonance
        schumann_base = weather_data.schumann_frequency
        
        # Geomagnetic activity factor
        geomagnetic_factor = weather_data.geomagnetic_index / 9.0  # Normalize Kp index
        
        # Altitude factor (higher altitude = stronger interference)
        altitude_factor = math.exp(weather_data.altitude / 10000.0)
        
        # Solar flux factor
        solar_factor = weather_data.solar_flux / 200.0  # Normalize solar flux
        
        # Compute interference pattern
        interference = 0.0
        for i, freq in enumerate(self.schumann_frequencies):
            # Resonance strength decreases with frequency
            resonance_strength = 1.0 / (i + 1)
            
            # Phase difference between current and resonant frequency
            phase_diff = abs(schumann_base - freq) / freq
            
            # Interference contribution
            interference += (
                resonance_strength *
                math.exp(-phase_diff) *
                (1.0 + geomagnetic_factor) *
                altitude_factor *
                solar_factor
            )
        
        return float(interference)
    
    def _compute_entropy_score(
        self,
        weather_data: WeatherDataPoint,
        location: GeoLocation
    ) -> float:
        """
        Compute entropy trigger score for unbiased entropy state validation.
        
        Based on the user's entropy_trigger_score function:
        entropy_score = 0.25 * temp_var + 0.5 * pressure_drop + 0.25 * schumann_deviation
        """
        # Temperature variation
        temp_var = abs(weather_data.temperature - 15.0)  # Deviation from 15°C baseline
        
        # Pressure drop (normalized)
        pressure_drop = max(0, 1013.25 - weather_data.pressure) / 1013.25
        
        # Schumann frequency deviation
        schumann_deviation = abs(weather_data.schumann_frequency - 7.83) / 7.83
        
        # Compute entropy score
        entropy_score = (
            0.25 * temp_var / 30.0 +  # Normalize temperature variation
            0.5 * pressure_drop +
            0.25 * schumann_deviation
        )
        
        # Apply location boost
        location_boost = location.entropy_zone_multiplier
        
        return float(entropy_score * location_boost)
    
    def _compute_resonance_strength(
        self,
        weather_data: WeatherDataPoint,
        location: GeoLocation,
        crwf_output: float
    ) -> float:
        """Compute resonance strength based on CRWF output and location factors."""
        # Base resonance from CRWF output
        base_resonance = abs(crwf_output)
        
        # Location resonance factor
        location_resonance = location.resonance_factor
        
        # Weather condition resonance
        weather_resonance = 1.0
        if weather_data.pressure < 1000:  # Low pressure = higher resonance
            weather_resonance = 1.5
        elif weather_data.pressure > 1020:  # High pressure = lower resonance
            weather_resonance = 0.7
        
        # Geomagnetic resonance
        geomagnetic_resonance = 1.0 + (weather_data.geomagnetic_index / 9.0)
        
        # Combined resonance strength
        resonance_strength = (
            base_resonance *
            location_resonance *
            weather_resonance *
            geomagnetic_resonance
        )
        
        return float(np.clip(resonance_strength, 0.0, 10.0))
    
    def _determine_weather_pattern(
        self,
        weather_data: WeatherDataPoint,
        crwf_output: float
    ) -> WeatherPattern:
        """Determine weather pattern based on CRWF analysis."""
        if weather_data.pressure > 1020:
            return WeatherPattern.HIGH_PRESSURE
        elif weather_data.pressure < 1000:
            return WeatherPattern.LOW_PRESSURE
        elif weather_data.geomagnetic_index > self.geomagnetic_threshold:
            return WeatherPattern.GEOMAGNETIC_STORM
        elif abs(crwf_output) > 2.0:
            return WeatherPattern.STORM_FRONT
        elif abs(crwf_output) < 0.5:
            return WeatherPattern.ATMOSPHERIC_STABILITY
        else:
            return WeatherPattern.WEATHER_TRANSITION
    
    def _compute_geo_alignment(
        self,
        location: GeoLocation,
        weather_data: WeatherDataPoint
    ) -> Dict[str, float]:
        """Compute geo-alignment score using LeyTrace and ColdBase logic."""
        # Ley line resonance (simplified)
        ley_line_resonance = location.ley_line_strength
        
        # Geomagnetic density
        geomagnetic_factor = location.geomagnetic_density
        
        # Cold base factor (simplified)
        cold_base_factor = 1.0
        if weather_data.temperature < 0:
            cold_base_factor = 1.2
        elif weather_data.temperature > 30:
            cold_base_factor = 0.8
        
        # Phase alignment
        phase_alignment = (
            ley_line_resonance *
            geomagnetic_factor *
            cold_base_factor
        )
        
        # Overall alignment score
        alignment_score = np.clip(phase_alignment, 0.0, 1.0)
        
        return {
            'alignment_score': float(alignment_score),
            'ley_line_resonance': float(ley_line_resonance),
            'geomagnetic_factor': float(geomagnetic_factor),
            'phase_alignment': float(phase_alignment)
        }
    
    def _compute_temporal_resonance(
        self,
        weather_data: WeatherDataPoint,
        location: GeoLocation
    ) -> float:
        """Compute temporal resonance based on time and location."""
        # Time-based resonance (hour of day)
        hour = weather_data.timestamp.hour
        time_resonance = 1.0 + 0.2 * math.sin(2 * math.pi * hour / 24.0)
        
        # Seasonal resonance
        day_of_year = weather_data.timestamp.timetuple().tm_yday
        seasonal_resonance = 1.0 + 0.1 * math.sin(2 * math.pi * day_of_year / 365.0)
        
        # Location temporal factor
        location_temporal = location.resonance_factor
        
        # Combined temporal resonance
        temporal_resonance = (
            time_resonance *
            seasonal_resonance *
            location_temporal
        )
        
        return float(np.clip(temporal_resonance, 0.0, 2.0))
    
    def _compute_crlf_adjustment(
        self,
        crwf_output: float,
        market_entropy: float,
        entropy_score: float
    ) -> float:
        """Compute CRLF adjustment factor based on CRWF output."""
        # Base adjustment from CRWF output
        base_adjustment = 1.0 + (crwf_output * 0.1)
        
        # Entropy alignment adjustment
        entropy_diff = entropy_score - market_entropy
        entropy_adjustment = 1.0 + (entropy_diff * 0.2)
        
        # Combined adjustment
        adjustment = base_adjustment * entropy_adjustment
        
        return float(np.clip(adjustment, 0.5, 2.0))
    
    def _generate_recommendations(
        self,
        crwf_output: float,
        weather_pattern: WeatherPattern,
        entropy_score: float
    ) -> Dict[str, Any]:
        """Generate trading recommendations based on CRWF analysis."""
        recommendations = {
            'weather_pattern': weather_pattern.value,
            'entropy_level': 'high' if entropy_score > 0.7 else 'medium' if entropy_score > 0.3 else 'low',
            'crwf_strength': 'strong' if abs(crwf_output) > 2.0 else 'moderate' if abs(crwf_output) > 1.0 else 'weak'
        }
        
        # Pattern-specific recommendations
        if weather_pattern == WeatherPattern.GEOMAGNETIC_STORM:
            recommendations.update({
                'action': 'reduce_exposure',
                'risk_multiplier': 1.5,
                'timeout_hours': 24
            })
        elif weather_pattern == WeatherPattern.STORM_FRONT:
            recommendations.update({
                'action': 'increase_volatility_hedge',
                'volatility_multiplier': 1.3,
                'timeout_hours': 12
            })
        elif weather_pattern == WeatherPattern.HIGH_PRESSURE:
            recommendations.update({
                'action': 'stable_trading',
                'risk_multiplier': 0.8,
                'timeout_hours': 6
            })
        elif weather_pattern == WeatherPattern.LOW_PRESSURE:
            recommendations.update({
                'action': 'opportunistic_trading',
                'risk_multiplier': 1.2,
                'timeout_hours': 8
            })
        
        return recommendations
    
    def _create_fallback_response(
        self,
        weather_data: WeatherDataPoint,
        location: GeoLocation
    ) -> CRWFResponse:
        """Create a fallback response when computation fails."""
        return CRWFResponse(
            crwf_output=0.0,
            entropy_score=0.5,
            resonance_strength=0.0,
            weather_pattern=WeatherPattern.ATMOSPHERIC_STABILITY,
            temperature_gradient=0.0,
            pressure_gradient=0.0,
            geo_alignment_score=0.5,
            ley_line_resonance=0.0,
            geomagnetic_factor=0.0,
            temporal_resonance=1.0,
            phase_alignment=0.0,
            crlf_adjustment_factor=1.0,
            market_entropy_adjustment=0.0,
            timestamp=weather_data.timestamp,
            location=location,
            recommendations={'action': 'fallback', 'error': 'Computation failed'}
        )
    
    async def fetch_weather_data(
        self,
        latitude: float,
        longitude: float,
        api_key: Optional[str] = None
    ) -> Optional[WeatherDataPoint]:
        """
        Fetch weather data from OpenWeatherMap API.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            api_key: OpenWeatherMap API key
            
        Returns:
            WeatherDataPoint with current weather data
        """
        try:
            api_key = api_key or self.api_key
            if not api_key:
                logger.warning("No API key provided for weather data fetch")
                return None

            # OpenWeatherMap API call
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': api_key,
                'units': 'metric'
            }
            
            async with asyncio.timeout(10):
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            
            # Extract weather data
            weather_data = WeatherDataPoint(
                timestamp=datetime.now(),
                latitude=latitude,
                longitude=longitude,
                altitude=data.get('main', {}).get('pressure', 1013.25) / 10.0,  # Rough altitude estimate
                temperature=data['main']['temp'],
                pressure=data['main']['pressure'],
                humidity=data['main']['humidity'],
                wind_speed=data['wind']['speed'],
                wind_direction=data['wind'].get('deg', 0.0),
                weather_type=data['weather'][0]['main'],
                source='openweathermap'
            )
            
            logger.info(f"Weather data fetched for {latitude:.2f}, {longitude:.2f}")
            return weather_data

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
        return None

    def get_location(self, latitude: float, longitude: float, name: str = "") -> GeoLocation:
        """Get or create a GeoLocation with computed resonance properties."""
        location_key = f"{latitude:.3f},{longitude:.3f}"
        
        if location_key in self.location_cache:
            return self.location_cache[location_key]
        
        # Create new location with computed properties
        location = GeoLocation(
            latitude=latitude,
            longitude=longitude,
            name=name,
            ley_line_strength=self._compute_ley_line_strength(latitude, longitude),
            geomagnetic_density=self._compute_geomagnetic_density(latitude, longitude),
            entropy_zone_multiplier=self._compute_entropy_zone_multiplier(latitude, longitude),
            resonance_factor=self._compute_resonance_factor(latitude, longitude)
        )
        
        self.location_cache[location_key] = location
        return location
    
    def _compute_ley_line_strength(self, latitude: float, longitude: float) -> float:
        """Compute ley line strength at given coordinates (simplified)."""
        # Simplified ley line computation based on known ley line intersections
        # In a real implementation, this would use actual ley line data
        
        # Tiger, GA coordinates (34.8°N, 83.4°W) - user's root node
        tiger_lat, tiger_lon = 34.8, -83.4
        
        # Distance from Tiger, GA
        distance = math.sqrt((latitude - tiger_lat)**2 + (longitude - tiger_lon)**2)
        
        # Ley line strength decreases with distance from known intersections
        ley_strength = math.exp(-distance / 10.0)  # 10 degree decay
        
        return float(np.clip(ley_strength, 0.0, 1.0))
    
    def _compute_geomagnetic_density(self, latitude: float, longitude: float) -> float:
        """Compute geomagnetic density at given coordinates."""
        # Simplified geomagnetic density computation
        # Higher density near poles, lower near equator
        
        # Distance from magnetic equator (simplified)
        magnetic_latitude = abs(latitude)
        
        # Geomagnetic density increases with magnetic latitude
        geomagnetic_density = 0.3 + 0.7 * (magnetic_latitude / 90.0)
        
        return float(np.clip(geomagnetic_density, 0.0, 1.0))
    
    def _compute_entropy_zone_multiplier(self, latitude: float, longitude: float) -> float:
        """Compute entropy zone multiplier for location."""
        # Simplified entropy zone computation
        # Higher entropy in equatorial regions, lower in polar regions
        
        # Distance from equator
        equator_distance = abs(latitude)
        
        # Entropy multiplier decreases with distance from equator
        entropy_multiplier = 1.5 - 0.5 * (equator_distance / 90.0)
        
        return float(np.clip(entropy_multiplier, 0.5, 1.5))
    
    def _compute_resonance_factor(self, latitude: float, longitude: float) -> float:
        """Compute resonance factor for location."""
        # Simplified resonance factor computation
        # Combines ley line strength and geomagnetic density
        
        ley_strength = self._compute_ley_line_strength(latitude, longitude)
        geomagnetic_density = self._compute_geomagnetic_density(latitude, longitude)
        
        # Resonance factor is geometric mean of ley strength and geomagnetic density
        resonance_factor = math.sqrt(ley_strength * geomagnetic_density)
        
        return float(np.clip(resonance_factor, 0.0, 1.0))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive CRWF performance summary."""
        if not self.computation_history:
            return {'error': 'No computation history available'}
        
        recent_responses = self.computation_history[-100:]  # Last 100 computations
        
        return {
            'total_computations': len(self.computation_history),
            'average_crwf_output': np.mean([r.crwf_output for r in recent_responses]),
            'average_entropy_score': np.mean([r.entropy_score for r in recent_responses]),
            'average_resonance_strength': np.mean([r.resonance_strength for r in recent_responses]),
            'weather_pattern_distribution': self._get_weather_pattern_distribution(),
            'geo_alignment_trend': self._get_geo_alignment_trend(),
            'crwf_output_statistics': self._get_crwf_statistics(),
            'location_cache_size': len(self.location_cache),
            'weather_history_size': len(self.weather_history)
        }
    
    def _get_weather_pattern_distribution(self) -> Dict[str, int]:
        """Get distribution of weather patterns."""
        distribution = {}
        for response in self.computation_history:
            pattern = response.weather_pattern.value
            distribution[pattern] = distribution.get(pattern, 0) + 1
        return distribution
    
    def _get_geo_alignment_trend(self) -> List[float]:
        """Get recent geo-alignment trend."""
        recent = self.computation_history[-20:]
        return [r.geo_alignment_score for r in recent] if recent else []
    
    def _get_crwf_statistics(self) -> Dict[str, float]:
        """Get CRWF output statistics."""
        outputs = [r.crwf_output for r in self.computation_history]
        if not outputs:
            return {}
        
        return {
            'mean': np.mean(outputs),
            'std': np.std(outputs),
            'min': np.min(outputs),
            'max': np.max(outputs),
            'median': np.median(outputs)
        }


def create_crwf_mapper(api_key: Optional[str] = None) -> ChronoResonanceWeatherMapper:
    """Factory function to create a CRWF mapper instance."""
    return ChronoResonanceWeatherMapper(api_key)


# Example usage and testing
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create CRWF mapper
    crwf = create_crwf_mapper()
    
    # Test location (Tiger, GA - user's root node)
    test_location = crwf.get_location(34.8, -83.4, "Tiger, GA")
    
    # Create test weather data
    test_weather = WeatherDataPoint(
        timestamp=datetime.now(),
        latitude=34.8,
        longitude=-83.4,
        altitude=300.0,
        temperature=20.0,
        pressure=1013.25,
        humidity=60.0,
        wind_speed=5.0,
        wind_direction=180.0,
        schumann_frequency=7.83,
        geomagnetic_index=2.0,
        solar_flux=100.0
    )
    
    # Compute CRWF
    response = crwf.compute_crwf(test_weather, test_location)
    
    print(f"CRWF Output: {response.crwf_output:.4f}")
    print(f"Entropy Score: {response.entropy_score:.3f}")
    print(f"Weather Pattern: {response.weather_pattern.value}")
    print(f"Geo Alignment: {response.geo_alignment_score:.3f}")
    print(f"CRLF Adjustment: {response.crlf_adjustment_factor:.3f}")
    
    # Get performance summary
    summary = crwf.get_performance_summary()
    print(f"\nPerformance Summary: {summary}") 