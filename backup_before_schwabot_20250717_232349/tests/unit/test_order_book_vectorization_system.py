import asyncio
import logging
import time
from typing import Any, Dict, List

import numpy as np

from core.api_bridge import APIBridge
from core.entry_exit_logic import EntryExitLogic
from core.order_book_vectorizer import OrderBookVectorizer
from core.strategy_bit_mapper import StrategyBitMapper

# -*- coding: utf-8 -*-
"""
Comprehensive Test for Order Book Vectorization System.

Tests the complete pipeline:
1. Order Book Vectorization (16-bit)
2. Strategy Bit Mapping
3. API Bridge Integration
4. Entry/Exit Logic
5. Full Integration Example

Demonstrates the system working with live-like data and
shows how all components integrate together.
"""



# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrderBookVectorizationSystem:
    """
    Complete Order Book Vectorization System.

    Integrates all components for end-to-end testing and demonstration.
    """

    def __init__(self):
        """Initialize the complete system."""
        self.order_book_vectorizer = OrderBookVectorizer()
        self.strategy_bit_mapper = StrategyBitMapper()
        self.api_bridge = APIBridge()
        self.entry_exit_logic = EntryExitLogic()

        logger.info("OrderBookVectorizationSystem initialized")

    async def run_full_integration_test():-> Dict[str, Any]:
        """
        Run full integration test of the complete system.

        Returns:
            Dictionary with comprehensive test results
        """
        logger.info("Starting full integration test...")

        test_results = {}
            "test_timestamp": time.time(),
            "components_tested": [],
            "performance_metrics": {},
            "integration_results": {},
            "errors": [],
        }
        try:
            # Test 1: Order Book Vectorization
            logger.info("Testing Order Book Vectorization...")
            vectorization_results = await self._test_order_book_vectorization()
            test_results["components_tested"].append("order_book_vectorization")
            test_results["integration_results"]["vectorization"] = vectorization_results

            # Test 2: Strategy Bit Mapping
            logger.info("Testing Strategy Bit Mapping...")
            bit_mapping_results = await self._test_strategy_bit_mapping()
            test_results["components_tested"].append("strategy_bit_mapping")
            test_results["integration_results"]["bit_mapping"] = bit_mapping_results

            # Test 3: API Bridge
            logger.info("Testing API Bridge...")
            api_results = await self._test_api_bridge()
            test_results["components_tested"].append("api_bridge")
            test_results["integration_results"]["api_bridge"] = api_results

            # Test 4: Entry/Exit Logic
            logger.info("Testing Entry/Exit Logic...")
            entry_exit_results = await self._test_entry_exit_logic()
            test_results["components_tested"].append("entry_exit_logic")
            test_results["integration_results"]["entry_exit_logic"] = entry_exit_results

            # Test 5: Full Integration Pipeline
            logger.info("Testing Full Integration Pipeline...")
            pipeline_results = await self._test_full_pipeline()
            test_results["components_tested"].append("full_pipeline")
            test_results["integration_results"]["full_pipeline"] = pipeline_results

            # Collect performance metrics
            test_results["performance_metrics"] = self._collect_performance_metrics()

            logger.info("Full integration test completed successfully!")

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            test_results["errors"].append(str(e))

        return test_results

    async def _test_order_book_vectorization():-> Dict[str, Any]:
        """Test order book vectorization functionality."""
        results = {}
            "success": False,
            "vectorization_tests": [],
            "performance_metrics": {},
        }
        try:
            # Create sample order book data
            sample_order_book = self._generate_sample_order_book()

            # Test 16-bit vectorization
            vector_16 = self.order_book_vectorizer.vectorize_order_book()
                sample_order_book, bit_depth=16, symbol="BTC/USDC"
            )

            vectorization_test = {}
                "test_name": "16-bit_vectorization",
                "success": True,
                "vector_shape": vector_16.shape,
                "vector_dtype": str(vector_16.dtype),
                "vector_mean": float(np.mean(vector_16)),
                "vector_std": float(np.std(vector_16)),
            }
            results["vectorization_tests"].append(vectorization_test)

            # Test 32-bit vectorization
            vector_32 = self.order_book_vectorizer.vectorize_order_book()
                sample_order_book, bit_depth=32, symbol="BTC/USDC"
            )

            vectorization_test_32 = {}
                "test_name": "32-bit_vectorization",
                "success": True,
                "vector_shape": vector_32.shape,
                "vector_dtype": str(vector_32.dtype),
                "vector_mean": float(np.mean(vector_32)),
                "vector_std": float(np.std(vector_32)),
            }
            results["vectorization_tests"].append(vectorization_test_32)

            # Test vector metrics
            metrics = self.order_book_vectorizer.compute_vector_metrics(vector_16)
            results["performance_metrics"] = metrics

            # Test batch vectorization
            order_books = [sample_order_book] * 3
            symbols = ["BTC/USDC", "ETH/USDC", "XRP/USDC"]
            batch_vectors = self.order_book_vectorizer.vectorize_order_book_batch()
                order_books, symbols, bit_depth=16
            )

            batch_test = {}
                "test_name": "batch_vectorization",
                "success": len(batch_vectors) == 3,
                "batch_size": len(batch_vectors),
                "symbols_processed": list(batch_vectors.keys()),
            }
            results["vectorization_tests"].append(batch_test)

            results["success"] = True

        except Exception as e:
            logger.error(f"Order book vectorization test failed: {e}")
            results["error"] = str(e)

        return results

    async def _test_strategy_bit_mapping():-> Dict[str, Any]:
        """Test strategy bit mapping functionality."""
        results = {}
            "success": False,
            "mapping_tests": [],
            "performance_metrics": {},
        }
        try:
            base_strategy = 0b1010  # 10 in decimal

            # Test flip expansion
            flip_strategies = self.strategy_bit_mapper.expand_strategy_bits()
                base_strategy, target_depth=8, mode="flip"
            )

            flip_test = {}
                "test_name": "flip_expansion",
                "success": len(flip_strategies) == 2,  # 8-bit = 2 strategies
                "strategies_count": len(flip_strategies),
                "strategies": flip_strategies,
            }
            results["mapping_tests"].append(flip_test)

            # Test mirror expansion
            mirror_strategies = self.strategy_bit_mapper.expand_strategy_bits()
                base_strategy, target_depth=8, mode="mirror"
            )

            mirror_test = {}
                "test_name": "mirror_expansion",
                "success": len(mirror_strategies) == 2,
                "strategies_count": len(mirror_strategies),
                "strategies": mirror_strategies,
            }
            results["mapping_tests"].append(mirror_test)

            # Test random expansion
            random_strategies = self.strategy_bit_mapper.expand_strategy_bits()
                base_strategy, target_depth=8, mode="random"
            )

            random_test = {}
                "test_name": "random_expansion",
                "success": len(random_strategies) == 2,
                "strategies_count": len(random_strategies),
                "strategies": random_strategies,
            }
            results["mapping_tests"].append(random_test)

            # Test Ferris expansion
            ferris_strategies = self.strategy_bit_mapper.expand_strategy_bits()
                base_strategy, target_depth=8, mode="ferris", ferris_phase=np.pi / 4
            )

            ferris_test = {}
                "test_name": "ferris_expansion",
                "success": len(ferris_strategies) == 2,
                "strategies_count": len(ferris_strategies),
                "strategies": ferris_strategies,
            }
            results["mapping_tests"].append(ferris_test)

            # Test self-similarity detection
            similarity_result = self.strategy_bit_mapper.detect_self_similarity()
                flip_strategies, similarity_threshold=0.8
            )

            similarity_test = {}
                "test_name": "self_similarity_detection",
                "success": True,
                "similarity_result": similarity_result,
            }
            results["mapping_tests"].append(similarity_test)

            # Get performance metrics
            metrics = self.strategy_bit_mapper.get_strategy_metrics(flip_strategies)
            results["performance_metrics"] = metrics

            results["success"] = True

        except Exception as e:
            logger.error(f"Strategy bit mapping test failed: {e}")
            results["error"] = str(e)

        return results

    async def _test_api_bridge():-> Dict[str, Any]:
        """Test API bridge functionality."""
        results = {}
            "success": False,
            "api_tests": [],
            "performance_metrics": {},
        }
        try:
            # Test order book fetching
            order_book = await self.api_bridge.fetch_order_book("BTC/USDC", limit=10)

            order_book_test = {}
                "test_name": "order_book_fetch",
                "success": "bids" in order_book and "asks" in order_book,
                "bids_count": len(order_book.get("bids", [])),
                "asks_count": len(order_book.get("asks", [])),
            }
            results["api_tests"].append(order_book_test)

            # Test price data fetching
            price_data = await self.api_bridge.fetch_price_data("BTC/USDC")

            price_test = {}
                "test_name": "price_data_fetch",
                "success": "price" in price_data,
                "price": price_data.get("price", 0),
                "symbol": price_data.get("symbol", ""),
            }
            results["api_tests"].append(price_test)

            # Test news sentiment fetching
            news_data = await self.api_bridge.fetch_news_sentiment("BTC", limit=5)

            news_test = {}
                "test_name": "news_sentiment_fetch",
                "success": len(news_data) > 0,
                "news_count": len(news_data),
                "avg_sentiment": np.mean()
                    [item.get("sentiment", 0) for item in news_data]
                ),
            }
            results["api_tests"].append(news_test)

            # Get performance metrics
            performance = self.api_bridge.get_api_performance_summary()
            results["performance_metrics"] = performance

            results["success"] = True

        except Exception as e:
            logger.error(f"API bridge test failed: {e}")
            results["error"] = str(e)

        return results

    async def _test_entry_exit_logic():-> Dict[str, Any]:
        """Test entry/exit logic functionality."""
        results = {}
            "success": False,
            "logic_tests": [],
            "performance_metrics": {},
        }
        try:
            # Create test data
            test_vector = np.random.rand(16)
            test_ferris_phase = np.pi / 4
            test_ghost_input = 0.5

            # Test entry signal
            entry_result = self.entry_exit_logic.compute_entry_signal()
                test_vector, test_ferris_phase, test_ghost_input
            )

            entry_test = {}
                "test_name": "entry_signal",
                "success": "should_enter" in entry_result,
                "should_enter": entry_result.get("should_enter", False),
                "signal_strength": entry_result.get("signal_strength", 0),
                "position_size": entry_result.get("position_size", 0),
            }
            results["logic_tests"].append(entry_test)

            # Test exit signal
            exit_result = self.entry_exit_logic.compute_exit_signal()
                test_vector,
                test_ferris_phase,
                test_ghost_input,
                entry_price=62000.0,
                current_price=62500.0,
                time_held=3600.0,
            )

            exit_test = {}
                "test_name": "exit_signal",
                "success": "should_exit" in exit_result,
                "should_exit": exit_result.get("should_exit", False),
                "signal_strength": exit_result.get("signal_strength", 0),
                "price_change_pct": exit_result.get("price_change_pct", 0),
            }
            results["logic_tests"].append(exit_test)

            # Get performance metrics
            performance = self.entry_exit_logic.get_trading_performance_summary()
            results["performance_metrics"] = performance

            results["success"] = True

        except Exception as e:
            logger.error(f"Entry/exit logic test failed: {e}")
            results["error"] = str(e)

        return results

    async def _test_full_pipeline():-> Dict[str, Any]:
        """Test the complete pipeline integration."""
        results = {}
            "success": False,
            "pipeline_steps": [],
            "final_result": {},
        }
        try:
            # Step 1: Fetch order book data
            logger.info("Pipeline Step 1: Fetching order book data...")
            order_book = await self.api_bridge.fetch_order_book("BTC/USDC", limit=8)

            step1 = {}
                "step": "fetch_order_book",
                "success": "bids" in order_book and "asks" in order_book,
                "data_shape": f"bids: {len(order_book.get('bids', []))}, asks: {len(order_book.get('asks', []))}",
            }
            results["pipeline_steps"].append(step1)

            # Step 2: Vectorize order book
            logger.info("Pipeline Step 2: Vectorizing order book...")
            vector = self.order_book_vectorizer.vectorize_order_book()
                order_book, bit_depth=16
            )

            step2 = {}
                "step": "vectorize_order_book",
                "success": vector.shape == (16,),
                "vector_shape": vector.shape,
                "vector_mean": float(np.mean(vector)),
            }
            results["pipeline_steps"].append(step2)

            # Step 3: Get Ferris phase
            logger.info("Pipeline Step 3: Computing Ferris phase...")
            ferris_phase = self._compute_ferris_phase()

            step3 = {}
                "step": "compute_ferris_phase",
                "success": True,
                "ferris_phase": ferris_phase,
            }
            results["pipeline_steps"].append(step3)

            # Step 4: Get Ghost input
            logger.info("Pipeline Step 4: Computing Ghost input...")
            ghost_input = self._compute_ghost_input(vector)

            step4 = {}
                "step": "compute_ghost_input",
                "success": True,
                "ghost_input": ghost_input,
            }
            results["pipeline_steps"].append(step4)

            # Step 5: Compute entry signal
            logger.info("Pipeline Step 5: Computing entry signal...")
            entry_signal = self.entry_exit_logic.compute_entry_signal()
                vector, ferris_phase, ghost_input
            )

            step5 = {}
                "step": "compute_entry_signal",
                "success": "should_enter" in entry_signal,
                "should_enter": entry_signal.get("should_enter", False),
                "signal_strength": entry_signal.get("signal_strength", 0),
            }
            results["pipeline_steps"].append(step5)

            # Step 6: Expand strategy bits if entry signal
            logger.info("Pipeline Step 6: Expanding strategy bits...")
            if entry_signal.get("should_enter", False):
                base_strategy = 0b1010
                expanded_strategies = self.strategy_bit_mapper.expand_strategy_bits()
                    base_strategy, target_depth=8, mode="flip"
                )

                step6 = {}
                    "step": "expand_strategy_bits",
                    "success": len(expanded_strategies) == 2,
                    "strategies_count": len(expanded_strategies),
                    "strategies": expanded_strategies,
                }
            else:
                step6 = {}
                    "step": "expand_strategy_bits",
                    "success": True,
                    "note": "No entry signal, skipping strategy expansion",
                }
            results["pipeline_steps"].append(step6)

            # Final result
            results["final_result"] = {}
                "entry_decision": entry_signal.get("should_enter", False),
                "signal_strength": entry_signal.get("signal_strength", 0),
                "position_size": entry_signal.get("position_size", 0),
                "risk_score": entry_signal.get("risk_assessment", {}).get()
                    "risk_score", 0.5
                ),
            }
            results["success"] = True

        except Exception as e:
            logger.error(f"Full pipeline test failed: {e}")
            results["error"] = str(e)

        return results

    def _generate_sample_order_book():-> Dict[str, List]:
        """Generate sample order book data for testing."""
        base_price = 62000.0
        spread = 0.01

        bids = []
        asks = []

        for i in range(8):
            bid_price = base_price * (1 - spread / 2 - i * 0.001)
            ask_price = base_price * (1 + spread / 2 + i * 0.001)

            bid_volume = np.random.uniform(0.1, 2.0)
            ask_volume = np.random.uniform(0.1, 2.0)

            bids.append([bid_price, bid_volume])
            asks.append([ask_price, ask_volume])

        return {"bids": bids, "asks": asks}

    def _compute_ferris_phase():-> float:
        """Compute Ferris wheel phase for testing."""
        # Simulate Ferris wheel phase based on current time
        current_time = time.time()
        period = 144  # 144 ticks per cycle
        phase = (current_time % period) / period * 2 * np.pi
        return phase

    def _compute_ghost_input():-> float:
        """Compute Ghost input for testing."""
        # Simulate Ghost overlay based on vector characteristics
        vector_entropy = -np.sum(vector * np.log2(vector + 1e-12))
        ghost_input = np.tanh(vector_entropy / 10.0)  # Normalize
        return ghost_input

    def _collect_performance_metrics():-> Dict[str, Any]:
        """Collect performance metrics from all components."""
        return {}
            "order_book_vectorizer": self.order_book_vectorizer.get_performance_summary(),
            "strategy_bit_mapper": self.strategy_bit_mapper.get_performance_summary(),
            "api_bridge": self.api_bridge.get_api_performance_summary(),
            "entry_exit_logic": self.entry_exit_logic.get_trading_performance_summary(),
        }


async def main():
    """Main test function."""
    logger.info("Starting Order Book Vectorization System Test")

    # Initialize system
    system = OrderBookVectorizationSystem()

    # Run full integration test
    results = await system.run_full_integration_test()

    # Print results
    print("\n" + "=" * 80)
    print("ORDER BOOK VECTORIZATION SYSTEM TEST RESULTS")
    print("=" * 80)

    print(f"\nTest Timestamp: {results['test_timestamp']}")
    print(f"Components Tested: {', '.join(results['components_tested'])}")

    if results["errors"]:
        print(f"\nErrors: {results['errors']}")

    # Print component results
    for component, component_results in results["integration_results"].items():
        print(f"\n{component.upper()} RESULTS:")
        print("-" * 40)

        if component_results.get("success", False):
            print(f"✅ {component} tests passed")

            if "vectorization_tests" in component_results:
                for test in component_results["vectorization_tests"]:
                    print()
                        f"  - {test['test_name']}: {'✅' if test['success'] else '❌'}"
                    )

            if "mapping_tests" in component_results:
                for test in component_results["mapping_tests"]:
                    print()
                        f"  - {test['test_name']}: {'✅' if test['success'] else '❌'}"
                    )

            if "api_tests" in component_results:
                for test in component_results["api_tests"]:
                    print()
                        f"  - {test['test_name']}: {'✅' if test['success'] else '❌'}"
                    )

            if "logic_tests" in component_results:
                for test in component_results["logic_tests"]:
                    print()
                        f"  - {test['test_name']}: {'✅' if test['success'] else '❌'}"
                    )

            if "pipeline_steps" in component_results:
                for step in component_results["pipeline_steps"]:
                    print(f"  - {step['step']}: {'✅' if step['success'] else '❌'}")
        else:
            print(f"❌ {component} tests failed")
            if "error" in component_results:
                print(f"  Error: {component_results['error']}")

    # Print performance summary
    print("\nPERFORMANCE SUMMARY:")
    print("-" * 40)

    for component, metrics in results["performance_metrics"].items():
        print(f"\n{component}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Print final pipeline result
    if "full_pipeline" in results["integration_results"]:
        pipeline_results = results["integration_results"]["full_pipeline"]
        if ()
            pipeline_results.get("success", False)
            and "final_result" in pipeline_results
        ):
            final_result = pipeline_results["final_result"]
            print("\nFINAL PIPELINE RESULT:")
            print("-" * 40)
            print()
                f"Entry Decision: {'✅ ENTER' if final_result['entry_decision'] else '❌ HOLD'}"
            )
            print(f"Signal Strength: {final_result['signal_strength']:.4f}")
            print(f"Position Size: {final_result['position_size']:.4f}")
            print(f"Risk Score: {final_result['risk_score']:.4f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
