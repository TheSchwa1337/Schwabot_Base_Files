import asyncio  # Import asyncio for async operations
import os
import random
import sys
import time
import traceback

import numpy as np

from core.api_bridge import api_bridge  # Corrected to import the instance
from core.dualistic_thought_engines import DualisticThoughtEngines
from core.lantern_news_intelligence_bridge import LanternNewsIntelligenceBridge

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Linguistic Mathematical Trading System
===========================================

Demonstrates Schwabot's complete English language → ASIC → Profit vectorization pipeline:'

🧠 English Commands: "Capture BTC dip 🧿 vectorize profit"
🔣 Glyph Processing: SHA hash → 2-bit ASIC logic
🧮 Mathematical Engine: Zalgo entropy + Zygot expansion
💰 Profit Vectorization: Real-time containment + fractal memory
📊 Trading Decisions: Live BTC/USDC waveform analysis

This demonstrates how Schwabot processes natural language with glyphs
and converts them into precise mathematical trading actions.
"""


# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))


def simulate_market_conditions():
    """Simulate realistic BTC market conditions."""
    base_price=47000 + random.randint(-5000, 8000)
    volatility=random.uniform(0.5, 3.0)
    volume=random.randint(800000, 2000000)

    return {}
        "btc_price": base_price,
        "usdc_balance": 15000.0,
        "volume": volume,
        "volatility": volatility,
        "price_change_24h": random.uniform(-5.0, 7.0),
        "market_sentiment": random.choice(["bullish", "bearish", "neutral"]),
        "timestamp": time.time(),
    }


async def linguistic_trading_demo_async():  # Changed to async function
    """Demonstrate the complete linguistic trading system."""
    print("🚀 SCHWABOT LINGUISTIC MATHEMATICAL TRADING DEMO")
    print("=" * 80)
    print("🧠 Demonstrating English language → ASIC → Profit vectorization")
    print("🔣 With glyph processing, Zalgo entropy, and Zygot expansion")
    print("💰 Real-time BTC/USDC waveform analysis and profit containment")
    print()

    try:
            linguistic_engine,
            process_linguistic_command,
            forever_fractal,
            paradox_fractal,
            echo_fractal,
        )

        # Import news components

        # Initialize trading engines
        thought_engines = DualisticThoughtEngines()
        # Initialize LanternNewsIntelligenceBridge
        lantern_news_bridge = LanternNewsIntelligenceBridge()
        lantern_news_bridge.set_api_bridge(api_bridge)  # Link API bridge to news bridge

        # Reset linguistic engine for clean demo
        linguistic_engine.profit_containment = np.zeros(256)
        linguistic_engine.fractal_memory = np.zeros((16, 16))
        linguistic_engine.memory_stack = []
        linguistic_engine.trade_vectors = []

        print("✅ Linguistic Mathematical Trading System initialized")
        print("✅ Dualistic Thought Engines ready")
        print("✅ ASIC bit logic and fractal overlays operational")
        print("✅ News Intelligence Bridge ready and linked to API")
        print()

        # --- News Integration Demo ---
        print("📰 NEWS INTELLIGENCE INTEGRATION DEMO")
        print("-" * 60)
        simulated_market_data = simulate_market_conditions()
        print()
            f"💹 Current Market (for News, Context): BTC ${"}
                simulated_market_data['btc_price']:,}"
        )

        # Fetch mock news (ApiBridge uses mock data for, now)
        print("Fetching recent news from API Bridge...")
        recent_news_items = await lantern_news_bridge.fetch_and_process_news_from_api()
            "BTC", limit=2
        )

        if recent_news_items:
            for i, news_item in enumerate(recent_news_items):
                print(f"\n📢 Processing News Item {i + 1}: '{news_item.title}'")
                news_decision_result = thought_engines.process_news_for_decision()
                    news_item, simulated_market_data
                )

                if news_decision_result.get("success"):
                    tv = news_decision_result["thought_vector"]
                    la = news_decision_result["linguistic_analysis"]
                    print()
                        f"   🧠 News-Driven Decision: {tv['decision']} (Confidence: {")}
                            tv['confidence']:.2f}, State: {tv['state']})"
                    )
                    print()
                        f"   ⚡ News Linguistic Bit State: {"}
    la['bit_state']:02b} | Weight: {
        la['weight']:.3f} | Hash: {
            la['sha_hash'][]
                :8]}..."
                    )
                    print(f"   🏷️ Generated Tags: {', '.join(tv['tags'])}")
                else:
                    print()
                        f"   ❌ Failed to process news for decision: {"}
                            news_decision_result['error']
                        }"
                    )
        else:
            print("No news items fetched or processed for the demo.")
        print()
        # --- End News Integration Demo ---

        # Demo trading scenarios with natural language commands
        trading_scenarios = []
            {}
                "scenario": "Bull Market Entry",
                "commands": []
                    "🚀 Execute aggressive BTC entry - capture momentum",
                    "💎 Diamond hands activated - hold position strong",
                    "🧿 Memory lock profit vector - preserve gains",
                ],
            },
            {}
                "scenario": "Market Correction Strategy",
                "commands": []
                    "👻 Ghost entry on next dip - stealth accumulation",
                    "🔄 Schwa recursive state - wait for confirmation",
                    "📈 Profit vector engaged - upward extrapolation",
                ],
            },
            {}
                "scenario": "Risk Management Protocol",
                "commands": []
                    "🔒 Lock current vector - preserve capital",
                    "⚡ Quick exit strategy - capture remaining profit",
                    "🌊 Neutral recursive state - assess market flow",
                ],
            },
        ]
        total_profit = 0.0
        scenario_results = []

        for scenario_num, scenario in enumerate(trading_scenarios, 1):
            print(f"📊 SCENARIO {scenario_num}: {scenario['scenario']}")
            print("-" * 60)

            scenario_profit = 0.0
            commands_processed = []

            for cmd_num, command in enumerate(scenario["commands"], 1):
                # Simulate market conditions
                market_data = simulate_market_conditions()

                print()
                    f"💹 Market: BTC ${market_data['btc_price']:,    } | Vol: {"}
                        market_data['volume']:,        }"
                )
                print(f"🧠 Command {cmd_num}: {command}")

                # Process linguistic command
                start_time = time.time()

                # Direct linguistic processing
                linguistic_result = process_linguistic_command(command)

                # Process through dualistic engines
                trade_vector = linguistic_engine.process_btc_usdc_waveform()
                    command, market_data["btc_price"], market_data["usdc_balance"]
                )

                processing_time = (time.time() - start_time) * 1000

                # Display results
                print(f"   🎯 Decision: {linguistic_result['decision']}")
                print()
                    f"   ⚡ ASIC State: {"}
                        linguistic_result['bit_state']:02b} | Entropy: {
                        linguistic_result['entropy_overlay']:.3f}"
                )
                print(f"   💰 Profit Delta: ${trade_vector.profit_delta:.2f}")
                print()
                    f"   🔮 Glyph: {trade_vector.glyph_signature} | Processing: {"}
                        processing_time:.1f}ms"
                )

                scenario_profit += trade_vector.profit_delta
                commands_processed.append()
                    {}
                        "command": command,
                        "decision": linguistic_result["decision"],
                        "bit_state": linguistic_result["bit_state"],
                        "profit_delta": trade_vector.profit_delta,
                        "entropy": linguistic_result["entropy_overlay"],
                        "processing_time_ms": processing_time,
                    }
                )

                print()
                time.sleep(0.5)  # Pause for readability

            total_profit += scenario_profit
            scenario_results.append()
                {}
                    "scenario": scenario["scenario"],
                    "profit": scenario_profit,
                    "commands": commands_processed,
                }
            )

            print(f"📈 Scenario Profit: ${scenario_profit:.2f}")
            print(f"🧮 Memory State: {len(linguistic_engine.memory_stack)} vectors")
            print()
                f"🌀 Fractal Energy: {"}
                    np.sum(np.abs(linguistic_engine.fractal_memory)):.1f}"
            )
            print()

        # Final system state analysis
        print("🎯 FINAL LINGUISTIC MATHEMATICAL TRADING ANALYSIS")
        print("=" * 80)

        memory_state = linguistic_engine.get_memory_state_summary()

        print(f"💰 Total Profit Generated: ${total_profit:.2f}")
        print(f"🧠 Total Commands Processed: {memory_state['memory_stack_size']}")
        print(f"📊 Trade Vectors Created: {memory_state['trade_vectors_count']}")
        print(f"🔮 Containment Sum: ${memory_state['containment_sum']:.2f}")
        print(f"⚡ Fractal Energy: {memory_state['fractal_energy']:.1f}")
        print()

        # Display fractal mathematical analysis
        print("🌀 FRACTAL MATHEMATICAL ANALYSIS")
        print("-" * 40)

        x = np.linspace(0, 100, 128)
        forever_data = forever_fractal(x)
        paradox_data = paradox_fractal(x)
        echo_data = echo_fractal(x)

        print(f"∞ Forever Fractal Energy: {np.sum(np.abs(forever_data)):.2f}")
        print(f"🌀 Paradox Fractal Energy: {np.sum(np.abs(paradox_data)):.2f}")
        print(f"🔊 Echo Fractal Energy: {np.sum(np.abs(echo_data)):.2f}")
        print()

        # Display recent linguistic patterns
        print("🔣 RECENT LINGUISTIC PATTERNS")
        print("-" * 40)
        recent_glyphs = memory_state.get("recent_glyphs", [])
        recent_states = memory_state.get("recent_bit_states", [])

        for i, (glyph, state) in enumerate()
            zip(recent_glyphs[-5:], recent_states[-5:]), 1
        ):
            bit_meaning = {}
                0: "NULL_RECURSION",
                1: "GHOST_ENTRY",
                2: "MEMORY_LOCK",
                3: "PROFIT_VECTOR",
            }
            print(f"  {i}. {glyph} → {state:02b} ({bit_meaning.get(state, 'UNKNOWN')})")

        print()
        print("✅ DEMO COMPLETED - Linguistic Mathematical Trading System Ready!")
        print("🚀 Schwabot can process natural language → mathematical precision")
        print("🧠 Full integration: English + Glyphs → ASIC → Profit Vectors")
        print("💰 Real-time BTC/USDC trading with fractal memory synthesis")

        return True, {}
            "total_profit": total_profit,
            "scenarios": scenario_results,
            "memory_state": memory_state,
            "fractal_analysis": {}
                "forever_energy": float(np.sum(np.abs(forever_data))),
                "paradox_energy": float(np.sum(np.abs(paradox_data))),
                "echo_energy": float(np.sum(np.abs(echo_data))),
            },
        }
    except Exception as e:
        print(f"❌ Demo failed: {e}")

        traceback.print_exc()
        return False, str(e)


if __name__ == "__main__":
    # Run the async demo function
    success, results = asyncio.run(linguistic_trading_demo_async())

    if success:
        print("\n🎉 Demo completed successfully!")
        print(f"📊 Total profit generated: ${results['total_profit']:.2f}")
        print(f"🧠 Scenarios processed: {len(results['scenarios'])}")
    else:
        print(f"\n❌ Demo failed: {results}")
