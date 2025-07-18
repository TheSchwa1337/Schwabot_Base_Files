import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

#!/usr/bin/env python3
"""
Mathematical Relay Sequencing Demo
=================================

Comprehensive demonstration of the mathematical relay sequencing system
with precise time log management across all math libraries and unified math states.

This demo showcases:
- BTC price hash synchronization with microsecond precision
- Bit-depth tensor switching with phase tracking
- Dual-channel switching with handoff timing
- Profit optimization with basket-tier navigation
- Cross-library mathematical relay integration
- Real-time sequencing validation and error recovery
- Comprehensive time log management and export
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger(__name__)


class MathematicalRelaySequencingDemo:
    """Comprehensive demo of mathematical relay sequencing system."""

    def __init__(self):
        """Initialize the demo system."""
        self.sequencer=None
        self.demo_results=[]
        self.start_time=datetime.now()

    def initialize_system(self) -> bool:
        """Initialize the mathematical relay sequencer."""
        print("🔧 Initializing Mathematical Relay Sequencing System")
        print("=" * 60)

        try:
                MathematicalRelaySequencer,
                TimeLogLevel,
            )

            # Initialize sequencer with microsecond precision
            self.sequencer = MathematicalRelaySequencer()
                mode="demo", log_level="INFO", time_log_level=TimeLogLevel.MICROSECOND
            )

            print("✅ Mathematical relay sequencer initialized successfully")
            print(f"✅ Mode: {self.sequencer.mode}")
            print(f"✅ Time log level: {self.sequencer.time_log_level.value}")
            print(f"✅ Start time: {self.sequencer.start_time.isoformat()}")

            # Check system integrations
            integrations = []
                ("Relay Navigator", self.sequencer.relay_navigator),
                ("Relay Integration", self.sequencer.relay_integration),
                ("Trend Manager", self.sequencer.trend_manager),
                ("Basket Engine", self.sequencer.basket_engine),
                ("QuickTime Manager", self.sequencer.quicktime_manager),
            ]
            for name, integration in integrations:
                status = "✅ Available" if integration else "⚠️ Not Available"
                print(f"{status} {name}")

            return True

        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            return False

    def demo_btc_price_hash_sequencing(self) -> Dict[str, Any]:
        """Demonstrate BTC price hash sequencing with precise timing."""
        print("\n🪙 BTC Price Hash Sequencing Demo")
        print("=" * 40)

        try:
            # Simulate BTC price updates with different phases
            btc_scenarios = []
                {}
                    "price": 50000.0,
                    "volume": 1000.0,
                    "phase": 32,
                    "description": "Standard 32-bit",
                },
                {}
                    "price": 51000.0,
                    "volume": 1500.0,
                    "phase": 16,
                    "description": "High volume 16-bit",
                },
                {}
                    "price": 52000.0,
                    "volume": 2000.0,
                    "phase": 8,
                    "description": "Spike 8-bit",
                },
                {}
                    "price": 53000.0,
                    "volume": 2500.0,
                    "phase": 4,
                    "description": "Surge 4-bit",
                },
                {}
                    "price": 54000.0,
                    "volume": 3000.0,
                    "phase": 2,
                    "description": "Peak 2-bit",
                },
            ]
            results = []

            for i, scenario in enumerate(btc_scenarios):
                print(f"\n📊 Scenario {i + 1}: {scenario['description']}")
                print(f"   Price: ${scenario['price']:,.2f}")
                print(f"   Volume: {scenario['volume']:,.0f}")
                print(f"   Phase: {scenario['phase']}-bit")

                # Execute BTC price hash sequence
                result = self.sequencer.sequence_btc_price_hash()
                    btc_price=scenario["price"],
                    btc_volume=scenario["volume"],
                    phase=scenario["phase"],
                    additional_data={}
                        "scenario": i + 1,
                        "description": scenario["description"],
                        "demo_type": "btc_price_hash",
                    },
                )

                if result.get("success", False):
                    duration = result.get("total_duration_seconds", 0)
                    btc_hash = result.get("btc_hash", "")[:16]
                    print(f"   ✅ Completed in {duration:.6f}s")
                    print(f"   🔗 Hash: {btc_hash}...")

                    results.append()
                        {}
                            "scenario": i + 1,
                            "description": scenario["description"],
                            "success": True,
                            "duration": duration,
                            "btc_hash": btc_hash,
                            "price": scenario["price"],
                            "volume": scenario["volume"],
                            "phase": scenario["phase"],
                        }
                    )
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                    results.append()
                        {}
                            "scenario": i + 1,
                            "description": scenario["description"],
                            "success": False,
                            "error": result.get("error", "Unknown error"),
                        }
                    )

                time.sleep(0.1)  # Small delay between scenarios

            # Calculate statistics
            successful_results = [r for r in results if r["success"]]
            if successful_results:
                avg_duration = sum(r["duration"] for r in successful_results) / len(successful_results)
                print("\n📈 BTC Sequencing Statistics:")
                print()
                    f"   Successful scenarios: {len(successful_results)}/{len(btc_scenarios)}"
                )
                print(f"   Average duration: {avg_duration:.6f}s")
                print()
                    f"   Success rate: {"}
                        len(successful_results) / len(btc_scenarios) * 100:.1f}%"
                )

            return {"type": "btc_price_hash", "results": results, "success": True}

        except Exception as e:
            print(f"❌ Error in BTC price hash sequencing demo: {e}")
            return {"type": "btc_price_hash", "error": str(e), "success": False}

    def demo_bit_depth_switching(self) -> Dict[str, Any]:
        """Demonstrate bit depth switching with phase tracking."""
        print("\n🔄 Bit Depth Switching Demo")
        print("=" * 35)

        try:
            # Test different bit depth transitions
            bit_depth_scenarios = []
                {}
                    "from": 32,
                    "to": 16,
                    "channel": "primary",
                    "description": "32→16 Primary",
                },
                {}
                    "from": 16,
                    "to": 8,
                    "channel": "secondary",
                    "description": "16→8 Secondary",
                },
                {}
                    "from": 8,
                    "to": 4,
                    "channel": "fallback",
                    "description": "8→4 Fallback",
                },
                {}
                    "from": 4,
                    "to": 2,
                    "channel": "primary",
                    "description": "4→2 Primary",
                },
                {}
                    "from": 2,
                    "to": 32,
                    "channel": "secondary",
                    "description": "2→32 Secondary",
                },
            ]
            results = []

            for i, scenario in enumerate(bit_depth_scenarios):
                print(f"\n🔄 Scenario {i + 1}: {scenario['description']}")
                print(f"   From: {scenario['from']}-bit")
                print(f"   To: {scenario['to']}-bit")
                print(f"   Channel: {scenario['channel']}")

                # Execute bit depth switch sequence
                result = self.sequencer.sequence_bit_depth_switch()
                    from_bit_depth=scenario["from"],
                    to_bit_depth=scenario["to"],
                    channel=scenario["channel"],
                    metadata={}
                        "scenario": i + 1,
                        "description": scenario["description"],
                        "demo_type": "bit_depth_switch",
                    },
                )

                if result.get("success", False):
                    duration = result.get("total_duration_seconds", 0)
                    print(f"   ✅ Completed in {duration:.6f}s")

                    results.append()
                        {}
                            "scenario": i + 1,
                            "description": scenario["description"],
                            "success": True,
                            "duration": duration,
                            "from_bit_depth": scenario["from"],
                            "to_bit_depth": scenario["to"],
                            "channel": scenario["channel"],
                        }
                    )
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                    results.append()
                        {}
                            "scenario": i + 1,
                            "description": scenario["description"],
                            "success": False,
                            "error": result.get("error", "Unknown error"),
                        }
                    )

                time.sleep(0.1)  # Small delay between scenarios

            # Calculate statistics
            successful_results = [r for r in results if r["success"]]
            if successful_results:
                avg_duration = sum(r["duration"] for r in successful_results) / len(successful_results)
                print("\n📈 Bit Depth Switching Statistics:")
                print()
                    f"   Successful switches: {len(successful_results)}/{len(bit_depth_scenarios)}"
                )
                print(f"   Average duration: {avg_duration:.6f}s")
                print()
                    f"   Success rate: {"}
                        len(successful_results) / len(bit_depth_scenarios) * 100:.1f}%"
                )

            return {"type": "bit_depth_switch", "results": results, "success": True}

        except Exception as e:
            print(f"❌ Error in bit depth switching demo: {e}")
            return {"type": "bit_depth_switch", "error": str(e), "success": False}

    def demo_profit_optimization(self) -> Dict[str, Any]:
        """Demonstrate profit optimization with basket-tier navigation."""
        print("\n💰 Profit Optimization Demo")
        print("=" * 35)

        try:
            # Test different profit optimization scenarios
            profit_scenarios = []
                {}
                    "target": 0.2,
                    "tier": "low",
                    "btc_price": 50000.0,
                    "description": "Low Risk 2%",
                },
                {}
                    "target": 0.5,
                    "tier": "medium",
                    "btc_price": 51000.0,
                    "description": "Medium Risk 5%",
                },
                {}
                    "target": 0.10,
                    "tier": "high",
                    "btc_price": 52000.0,
                    "description": "High Risk 10%",
                },
                {}
                    "target": 0.15,
                    "tier": "aggressive",
                    "btc_price": 53000.0,
                    "description": "Aggressive 15%",
                },
                {}
                    "target": 0.20,
                    "tier": "extreme",
                    "btc_price": 54000.0,
                    "description": "Extreme 20%",
                },
            ]
            results = []

            for i, scenario in enumerate(profit_scenarios):
                print(f"\n💰 Scenario {i + 1}: {scenario['description']}")
                print(f"   Target: {scenario['target'] * 100:.1f}%")
                print(f"   Tier: {scenario['tier']}")
                print(f"   BTC Price: ${scenario['btc_price']:,.2f}")

                # Execute profit optimization sequence
                result = self.sequencer.sequence_profit_optimization()
                    profit_target=scenario["target"],
                    basket_tier=scenario["tier"],
                    btc_price=scenario["btc_price"],
                    metadata={}
                        "scenario": i + 1,
                        "description": scenario["description"],
                        "demo_type": "profit_optimization",
                    },
                )

                if result.get("success", False):
                    duration = result.get("total_duration_seconds", 0)
                    print(f"   ✅ Completed in {duration:.6f}s")

                    results.append()
                        {}
                            "scenario": i + 1,
                            "description": scenario["description"],
                            "success": True,
                            "duration": duration,
                            "target": scenario["target"],
                            "tier": scenario["tier"],
                            "btc_price": scenario["btc_price"],
                        }
                    )
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                    results.append()
                        {}
                            "scenario": i + 1,
                            "description": scenario["description"],
                            "success": False,
                            "error": result.get("error", "Unknown error"),
                        }
                    )

                time.sleep(0.1)  # Small delay between scenarios

            # Calculate statistics
            successful_results = [r for r in results if r["success"]]
            if successful_results:
                avg_duration = sum(r["duration"] for r in successful_results) / len(successful_results)
                print("\n📈 Profit Optimization Statistics:")
                print()
                    f"   Successful optimizations: {len(successful_results)}/{len(profit_scenarios)}"
                )
                print(f"   Average duration: {avg_duration:.6f}s")
                print()
                    f"   Success rate: {"}
                        len(successful_results) / len(profit_scenarios) * 100:.1f}%"
                )

            return {"type": "profit_optimization", "results": results, "success": True}

        except Exception as e:
            print(f"❌ Error in profit optimization demo: {e}")
            return {"type": "profit_optimization", "error": str(e), "success": False}

    def demo_quicktime_events(self) -> Dict[str, Any]:
        """Demonstrate QuickTime event handling with sequencing."""
        print("\n⚡ QuickTime Events Demo")
        print("=" * 30)

        try:
            if not self.sequencer.quicktime_manager:
                print("⚠️ QuickTime manager not available, skipping demo")
                return {"type": "quicktime_events", "skipped": True, "success": True}

            # Simulate various QuickTime events
            quicktime_scenarios = []
                {}
                    "event_type": "price_spike",
                    "context": {}
                        "btc_price": 55000.0,
                        "volume": 3000.0,
                        "basket_id": "spike_basket_1",
                        "tier": "high",
                        "bit_depth": 32,
                        "channel": "primary",
                        "sub_ring": 0,
                    },
                    "description": "Price Spike Event",
                },
                {}
                    "event_type": "volume_surge",
                    "context": {}
                        "btc_price": 56000.0,
                        "volume": 4000.0,
                        "basket_id": "surge_basket_2",
                        "tier": "medium",
                        "bit_depth": 16,
                        "channel": "secondary",
                        "sub_ring": 1,
                    },
                    "description": "Volume Surge Event",
                },
                {}
                    "event_type": "market_crash",
                    "context": {}
                        "btc_price": 45000.0,
                        "volume": 5000.0,
                        "basket_id": "crash_basket_3",
                        "tier": "extreme",
                        "bit_depth": 8,
                        "channel": "fallback",
                        "sub_ring": 2,
                    },
                    "description": "Market Crash Event",
                },
            ]
            results = []

            for i, scenario in enumerate(quicktime_scenarios):
                print(f"\n⚡ Scenario {i + 1}: {scenario['description']}")
                print(f"   Event Type: {scenario['event_type']}")
                print(f"   BTC Price: ${scenario['context']['btc_price']:,.2f}")
                print(f"   Volume: {scenario['context']['volume']:,.0f}")
                print(f"   Basket: {scenario['context']['basket_id']}")

                # Trigger QuickTime event
                self.sequencer.quicktime_manager.detect_and_log_event()
                    event_type=scenario["event_type"], context=scenario["context"]
                )

                print("   ✅ Event triggered")
                results.append()
                    {}
                        "scenario": i + 1,
                        "description": scenario["description"],
                        "event_type": scenario["event_type"],
                        "success": True,
                    }
                )

                time.sleep(0.2)  # Delay between events

            # Wait for event processing
            time.sleep(1.0)

            # Get event log
            event_log = self.sequencer.quicktime_manager.get_event_log()
            print("\n📊 QuickTime Event Statistics:")
            print(f"   Events triggered: {len(quicktime_scenarios)}")
            print(f"   Event log entries: {len(event_log)}")

            return {"type": "quicktime_events", "results": results, "success": True}

        except Exception as e:
            print(f"❌ Error in QuickTime events demo: {e}")
            return {"type": "quicktime_events", "error": str(e), "success": False}

    def demo_time_log_analysis(self) -> Dict[str, Any]:
        """Demonstrate comprehensive time log analysis."""
        print("\n⏱️ Time Log Analysis Demo")
        print("=" * 35)

        try:
            # Get comprehensive statistics
            statistics = self.sequencer.get_sequencing_statistics()

            if "error" in statistics:
                print(f"❌ Error getting statistics: {statistics['error']}")
                return {}
                    "type": "time_log_analysis",
                    "error": statistics["error"],
                    "success": False,
                }

            print("📊 Overall System Statistics:")
            print(f"   Active sequences: {statistics.get('active_sequences', 0)}")
            print(f"   Completed sequences: {statistics.get('completed_sequences', 0)}")
            print(f"   Total sequences: {statistics.get('total_sequences', 0)}")
            print(f"   Time logs count: {statistics.get('time_logs_count', 0)}")
            print()
                f"   Average duration: {"}
                    statistics.get('average_duration_seconds', 0):.6f}s"
            )
            print(f"   Uptime: {statistics.get('uptime_seconds', 0):.1f}s")

            # Show sequence type distribution
            type_dist = statistics.get("sequence_type_distribution", {})
            if type_dist:
                print("\n📈 Sequence Type Distribution:")
                for seq_type, count in type_dist.items():
                    percentage = count / statistics.get("total_sequences", 1) * 100
                    print(f"   {seq_type}: {count} ({percentage:.1f}%)")

            # Show sequence status distribution
            status_dist = statistics.get("sequence_status_distribution", {})
            if status_dist:
                print("\n📊 Sequence Status Distribution:")
                for status, count in status_dist.items():
                    percentage = count / statistics.get("total_sequences", 1) * 100
                    print(f"   {status}: {count} ({percentage:.1f}%)")

            # Get recent time logs for analysis
            recent_logs = self.sequencer.get_time_logs(limit=20)

            if recent_logs:
                print()
                    f"\n📝 Recent Time Log Analysis (Last {len(recent_logs)} entries):"
                )

                # Group by operation type
                operation_counts = {}
                total_duration = 0

                for log in recent_logs:
                    operation = log.get("operation", "unknown")
                    duration = log.get("duration_microseconds", 0)

                    operation_counts[operation] = operation_counts.get(operation, 0) + 1
                    total_duration += duration

                # Show operation distribution
                for operation, count in sorted(operation_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {operation}: {count} times")

                avg_duration = total_duration / len(recent_logs) if recent_logs else 0
                print(f"   Average operation duration: {avg_duration:.2f}μs")

            return {}
                "type": "time_log_analysis",
                "statistics": statistics,
                "success": True,
            }

        except Exception as e:
            print(f"❌ Error in time log analysis demo: {e}")
            return {"type": "time_log_analysis", "error": str(e), "success": False}

    def demo_data_export(self) -> Dict[str, Any]:
        """Demonstrate data export and persistence."""
        print("\n💾 Data Export Demo")
        print("=" * 25)

        try:
            # Export sequencing data
            print("📤 Exporting comprehensive sequencing data...")
            export_filename = self.sequencer.export_sequencing_data()

            if os.path.exists(export_filename):
                print(f"✅ Data exported to: {export_filename}")

                # Read and analyze exported data
                with open(export_filename, "r") as f:
                    export_data = json.load(f)

                # Show export summary
                sequencer_info = export_data.get("sequencer_info", {})
                statistics = export_data.get("statistics", {})

                print("\n📊 Export Summary:")
                print(f"   File size: {os.path.getsize(export_filename)} bytes")
                print(f"   Mode: {sequencer_info.get('mode', 'unknown')}")
                print()
                    f"   Time log level: {sequencer_info.get('time_log_level', 'unknown')}"
                )
                print(f"   Uptime: {sequencer_info.get('uptime_seconds', 0):.1f}s")
                print(f"   Total sequences: {statistics.get('total_sequences', 0)}")
                print(f"   Time logs: {statistics.get('time_logs_count', 0)}")

                # Show export structure
                print("\n📁 Export Structure:")
                for key, value in export_data.items():
                    if isinstance(value, list):
                        print(f"   {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"   {key}: {len(value)} keys")
                    else:
                        print(f"   {key}: {type(value).__name__}")

                # Clean up export file
                os.remove(export_filename)
                print("\n✅ Export file cleaned up")

                return {
                "type": "data_export",
                "filename": export_filename,
                "file_size": os.path.getsize(export_filename)
                    if os.path.exists(export_filename)
                    else 0,
                "success": True,
                }
            else:
                print(f"❌ Export file not created: {export_filename}")
                return {
                "type": "data_export",
                "error": "Export file not created",
                "success": False,
                }

        except Exception as e:
            print(f"❌ Error in data export demo: {e}")
            return {"type": "data_export", "error": str(e), "success": False}

    def run_comprehensive_demo(self) -> bool:
        """Run the comprehensive mathematical relay sequencing demo."""
        print("🧮 Mathematical Relay Sequencing Demo")
        print("=" * 60)
        print("Comprehensive demonstration of mathematical relay sequencing")
        print("with precise time log management across all library systems.")
        print()

        # Initialize system
        if not self.initialize_system():
            return False

        # Run all demos
        demos = [
            ("BTC Price Hash Sequencing", self.demo_btc_price_hash_sequencing),
            ("Bit Depth Switching", self.demo_bit_depth_switching),
            ("Profit Optimization", self.demo_profit_optimization),
            ("QuickTime Events", self.demo_quicktime_events),
            ("Time Log Analysis", self.demo_time_log_analysis),
            ("Data Export", self.demo_data_export),
        ]
        for demo_name, demo_func in demos:
            print(f"\n{'=' * 60}")
            print(f"🎯 Running: {demo_name}")
            print(f"{'=' * 60}")

            try:
                result = demo_func()
                self.demo_results.append(result)

                if result.get("success", False):
                    print(f"✅ {demo_name} completed successfully")
                else:
                    print(
                        f"❌ {demo_name} failed: {result.get('error', 'Unknown error')}"
                    )

            except Exception as e:
                print(f"❌ {demo_name} crashed: {e}")
                self.demo_results.append(
                    {"type": demo_name.lower().replace(" ", "_"), "error": str(e), "success": False}
                )

            time.sleep(0.5)  # Brief pause between demos

        # Print comprehensive summary
        self.print_demo_summary()

        return True

    def print_demo_summary(self):
        """Print comprehensive demo summary."""
        print(f"\n{'=' * 60}")
        print("📊 COMPREHENSIVE DEMO SUMMARY")
        print(f"{'=' * 60}")

        # Calculate overall statistics
        total_demos = len(self.demo_results)
        successful_demos = len([r for r in self.demo_results if r.get("success", False)])
        failed_demos = total_demos - successful_demos

        print("🎯 Overall Results:")
        print(f"   Total demos: {total_demos}")
        print(f"   Successful: {successful_demos}")
        print(f"   Failed: {failed_demos}")
        print(f"   Success rate: {successful_demos / total_demos * 100:.1f}%")

        # Show individual demo results
        print("\n📋 Individual Demo Results:")
        for result in self.demo_results:
            demo_type = result.get("type", "unknown")
            success = result.get("success", False)
            status = "✅ PASS" if success else "❌ FAIL"

            if success and "results" in result:
                results = result["results"]
                successful_results = len([r for r in results if r.get("success", False)])
                total_results = len(results)
                print()
                    f"   {status} {demo_type}: {successful_results}/{total_results} scenarios"
                )
            else:
                print(f"   {status} {demo_type}")

        # Show final system statistics
        if self.sequencer:
            print("\n📈 Final System Statistics:")
            final_stats = self.sequencer.get_sequencing_statistics()

            if "error" not in final_stats:
                print(f"   Total sequences: {final_stats.get('total_sequences', 0)}")
                print(f"   Time logs: {final_stats.get('time_logs_count', 0)}")
                print()
                    f"   Average duration: {"}
                        final_stats.get('average_duration_seconds', 0):.6f}s"
                )
                print(f"   Total uptime: {final_stats.get('uptime_seconds', 0):.1f}s")

                # Show most common sequence types
                type_dist = final_stats.get("sequence_type_distribution", {})
                if type_dist:
                    most_common = max(type_dist.items(), key=lambda x: x[1])
                    print()
                        f"   Most common sequence: {most_common[0]} ({")}
                            most_common[1]
                        } times)"
                    )

        # Demo duration
        demo_duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n⏱️ Demo Duration: {demo_duration:.1f} seconds")

        if successful_demos == total_demos:
            print("\n🎉 All demos completed successfully!")
            print("Mathematical relay sequencing system is working correctly.")
        else:
            print("\n⚠️ Some demos failed. Please review the implementation.")

        print(f"\n{'=' * 60}")


def main():
    """Main demo function."""
    demo = MathematicalRelaySequencingDemo()
    success = demo.run_comprehensive_demo()

    if success:
        print("🎉 Demo completed successfully!")
        return True
    else:
        print("❌ Demo failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
