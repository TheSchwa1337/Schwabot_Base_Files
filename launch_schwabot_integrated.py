import logging
import math
import random
import sys
from datetime import datetime, timedelta

from core.chrono_resonance_weather_mapper import WeatherDataPoint
from core.data_pipeline_visualizer import DataCategory
from core.schwabot_integrated_launcher import SchwabotIntegratedLauncher
from core.secure_api_coordinator import SecureAPICoordinator

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schwabot Integrated Launcher Demo - Complete System Demonstration."

This script demonstrates the complete Schwabot integrated system including:
1. Secure API management and key storage
2. Data pipeline visualization
3. ChronoResonance Weather Mapping (CRWM)
4. Profit-driven trading strategy
5. System monitoring and controls

Usage:
    python launch_schwabot_integrated.py
"""


# Set up logging
logging.basicConfig()
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("schwabot_launcher.log")],
)

logger=logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are available."""
    required_modules=[]
    optional_modules=[]
        ("numpy", "NumPy for mathematical operations"),
        ("scipy", "SciPy for advanced mathematical functions"),
        ("psutil", "PSUtil for system monitoring"),
        ("cryptography", "Cryptography for secure API key storage"),
        ("requests", "Requests for API communication"),
        ("aiohttp", "AioHTTP for async API operations"),
    ]

    missing_required=[]
    missing_optional=[]

    # Check required modules
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(module)

    # Check optional modules
    for module, description in optional_modules:
        try:
            __import__(module)
            logger.info(f"✅ {description}")
        except ImportError:
            missing_optional.append((module, description))
            logger.warning(f"⚠️ Optional dependency missing: {module} - {description}")

    if missing_required:
        logger.error(f"❌ Missing required dependencies: {', '.join(missing_required)}")
        return False

    if missing_optional:
        logger.info()
            "📝 Some optional features may not be available due to missing dependencies"
        )
        logger.info()
            "To install all dependencies, run: pip install numpy scipy psutil cryptography requests aiohttp"
        )

    return True


def create_demo_launcher():
    """Create a simplified demo launcher that works without all dependencies."""

    print("🚀 Schwabot Integrated System Demo")
    print("=" * 50)

    # Initialize demo components
    components={}
        "api_coordinator": None,
        "data_pipeline": None,
        "crwm_mapper": None,
        "profit_engine": None,
    }

    try:
        # Try to import and initialize API coordinator
        try:

            components["api_coordinator"]=SecureAPICoordinator()
            logger.info("✅ Secure API Coordinator initialized")
        except ImportError as e:
            logger.warning(f"⚠️ API Coordinator not available: {e}")

        # Try to import and initialize data pipeline visualizer
        try:
                DataPipelineVisualizer,
                DataCategory,
                DataTier,
            )

            components["data_pipeline"]=DataPipelineVisualizer()
            logger.info("✅ Data Pipeline Visualizer initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Data Pipeline Visualizer not available: {e}")

        # Try to import and initialize CRWM
        try:
                ChronoResonanceWeatherMapper,
            )

            components["crwm_mapper"]=ChronoResonanceWeatherMapper()
            logger.info("✅ ChronoResonance Weather Mapper initialized")
        except ImportError as e:
            logger.warning(f"⚠️ CRWM not available: {e}")

        # Try to import profit optimization
        try:
                EnhancedProfitTradingStrategy,
            )

            components["profit_engine"] = EnhancedProfitTradingStrategy()
            logger.info("✅ Enhanced Profit Trading Strategy initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Profit Engine not available: {e}")

    except Exception as e:
        logger.error(f"❌ Error initializing components: {e}")

    return components


def run_demo_without_ui(components):
    """Run a demo without UI components."""

    print("\n🎯 Running System Demo (Console, Mode)")
    print("-" * 40)

    # Demo API Coordinator
    if components["api_coordinator"]:
        print("\n🔐 API Coordinator Demo:")
        api_coord = components["api_coordinator"]
        status = api_coord.get_api_status()
        print(f"  Total providers configured: {status['total_providers']}")
        print(f"  Performance stats: {status['performance_stats']}")

    # Demo Data Pipeline
    if components["data_pipeline"]:
        print("\n💾 Data Pipeline Demo:")
        pipeline = components["data_pipeline"]

        # Add some demo data

        for i in range(10):
            category = random.choice(list(DataCategory))
            size = random.randint(1024, 10240)
            unit_id = pipeline.add_data_unit(category, size)
            if unit_id:
                print(f"  Added data unit: {unit_id[:20]}... ({size} bytes)")

        status = pipeline.get_pipeline_status()
        print(f"  Total units: {status['total_units']}")
        print(f"  Total size: {status['total_size_bytes']} bytes")

    # Demo CRWM
    if components["crwm_mapper"]:
        print("\n🌤️ ChronoResonance Weather Mapping Demo:")
        crwm = components["crwm_mapper"]

        # Add demo weather data

        base_time = datetime.now() - timedelta(hours=2)
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i * 10)
            temp = 20 + 5 * math.sin(i * 0.5) + random.uniform(-1, 1)
            pressure = 1013 + random.uniform(-10, 10)
            humidity = 60 + random.uniform(-20, 20)
            wind_speed = 3 + random.uniform(0, 5)

            weather_point = WeatherDataPoint()
                timestamp=timestamp,
                location="Demo Location",
                temperature=temp,
                pressure=pressure,
                humidity=humidity,
                wind_speed=wind_speed,
                wind_direction=random.uniform(0, 360),
                weather_type="partly_cloudy",
            )

            crwm.add_weather_data(weather_point)

            # Add corresponding price data
            price = 45000 + temp * 100 + random.uniform(-200, 200)
            crwm.add_price_data(timestamp, price)

        status = crwm.get_crwm_status()
        print(f"  Weather data points: {status['data_points']['weather_history']}")
        print(f"  Price data points: {status['data_points']['price_history']}")

        # Get weather signature
        signature = crwm.get_weather_signature("1h")
        if signature:
            print()
                f"  Current temperature: {"}
                    signature['current_conditions']['temperature']:.1f}°C"
            )
            print(f"  Trading signal: {signature['trading_signals']['direction']}")

    # Demo Profit Engine
    if components["profit_engine"]:
        print("\n💰 Profit Trading Strategy Demo:")
        profit_engine = components["profit_engine"]

        # Run demo analysis
        try:
            # Simulate market data

            demo_market_data = {}
                "btc_price": 45000 + random.uniform(-1000, 1000),
                "volume_24h": 2500000000 + random.uniform(-500000000, 500000000),
                "price_change_24h": random.uniform(-5, 5),
                "volatility": random.uniform(0.2, 0.8),
            }

            result = profit_engine.analyze_market_conditions(demo_market_data)
            print(f"  Market analysis result: {result}")

        except Exception as e:
            logger.warning(f"Profit engine demo error: {e}")
            print("  Profit engine initialized (demo data simulation, failed)")

    print("\n✅ Demo completed successfully!")


def run_ui_demo(components):
    """Run demo with UI components if available."""

    print("\n🎮 Starting UI Demo...")

    try:
        # Try to run the main integrated launcher

        print("🚀 Starting Schwabot Integrated Control Center...")
        launcher = SchwabotIntegratedLauncher()
        launcher.run()

    except ImportError as e:
        logger.warning(f"UI components not available: {e}")
        print("⚠️ UI mode not available, running console demo instead...")
        run_demo_without_ui(components)

    except Exception as e:
        logger.error(f"Error running UI demo: {e}")
        print("❌ UI demo failed, falling back to console mode...")
        run_demo_without_ui(components)


def main():
    """Main entry point for the integrated launcher demo."""

    print("🔧 Checking system requirements...")

    # Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("❌ Critical dependencies missing. Please install required packages.")
        return 1

    print("\n🏗️ Initializing Schwabot components...")

    # Initialize components
    components = create_demo_launcher()

    # Count available components
    available_components = sum(1 for comp in components.values() if comp is not None)
    total_components = len(components)

    print()
        f"\n📊 System Status: {available_components}/{total_components} components available"
    )

    if available_components == 0:
        print("❌ No components available. Please check installation.")
        return 1

    # Ask user for demo mode
    print("\n🎯 Select demo mode:")
    print("1. Console demo (works with any, setup)")
    print("2. UI demo (requires, tkinter)")
    print("3. Full integration test")

    try:
        choice = input("\nEnter choice (1-3, default=1): ").strip()
        if not choice:
            choice = "1"

        if choice == "1":
            run_demo_without_ui(components)
        elif choice == "2":
            run_ui_demo(components)
        elif choice == "3":
            print("\n🧪 Running full integration test...")
            run_demo_without_ui(components)
            if any(comp for comp in components.values()):
                print("\n🎮 Attempting UI demo...")
                run_ui_demo(components)
        else:
            print("Invalid choice, running console demo...")
            run_demo_without_ui(components)

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"❌ Demo error: {e}")
        return 1

    print("\n🎉 Schwabot Integrated System Demo completed!")
    print()
        "📖 For more information, check the documentation files in the project directory."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
