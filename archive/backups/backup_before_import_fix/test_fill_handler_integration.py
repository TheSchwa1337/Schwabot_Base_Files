#!/usr/bin/env python3
"""
Test Fill Handler Integration - Advanced Crypto Trading Fill Management

This test demonstrates the integration of the fill handler with the secure exchange manager
to handle partial fills, retries, and crypto-specific trading challenges.

Features tested:
- Fill event parsing for different exchanges
- Partial fill handling and retry logic
- Order state management
- Fill statistics and performance tracking
- State persistence and recovery
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Any, Dict

# Setup logging
logging.basicConfig()
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the modules
    try:
    from core.fill_handler import FillEvent, FillHandler, FillStatus, OrderState, create_fill_handler
    from core.secure_exchange_manager import ExchangeType, SecureExchangeManager, TradeResult
    MODULES_AVAILABLE = True
    except ImportError as e:
    logger.error(f"Required modules not available: {e}")
    MODULES_AVAILABLE = False


class FillHandlerIntegrationTest:
    """Test suite for fill handler integration."""

    def __init__(self):
        """Initialize the test suite."""
        self.fill_handler = None
        self.exchange_manager = None
        self.test_results = []

        logger.info("🧪 Fill Handler Integration Test initialized")

    async def setup(self):
        """Setup test environment."""
        if not MODULES_AVAILABLE:
            logger.error("❌ Required modules not available")
            return False

        try:
            # Initialize fill handler
            self.fill_handler = await create_fill_handler({)}
                'retry_config': {}
                    'max_retries': 3,
                    'base_delay': 1.0,
                    'max_delay': 10.0,
                    'exponential_base': 2.0,
                    'jitter_factor': 0.1
                }
            })

            # Initialize exchange manager
            self.exchange_manager = SecureExchangeManager()

            logger.info("✅ Test environment setup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

    async def test_fill_event_parsing(self):
        """Test fill event parsing for different exchanges."""
        logger.info("🔍 Testing fill event parsing...")

        test_cases = []
            {}
                "name": "Binance Fill",
                "data": {}
                    "orderId": "123456789",
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "fills": []
                        {}
                            "tradeId": "987654321",
                            "qty": "0.01",
                            "price": "50000.0",
                            "commission": "0.00001",
                            "commissionAsset": "BTC",
                            "takerOrMaker": "taker"
                        }
                    ]
                }
            },
            {}
                "name": "Bitget Fill",
                "data": {}
                    "orderId": "456789123",
                    "tradeId": "321654987",
                    "symbol": "BTCUSDT",
                    "side": "SELL",
                    "baseVolume": "0.02",
                    "fillPrice": "51000.0",
                    "fillFee": "0.102",
                    "fillFeeCoin": "USDT",
                    "tradeScope": "maker"
                }
            },
            {}
                "name": "Phemex Fill",
                "data": {}
                    "orderID": "789123456",
                    "execID": "147258369",
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "execQty": "0.03",
                    "execPriceRp": "52000.0",
                    "execFeeRv": "0.156",
                    "feeCurrency": "USDT",
                    "execStatus": "taker"
                }
            }
        ]

        for test_case in test_cases:
            try:
                fill_event = await self.fill_handler.process_fill_event(test_case["data"])

                # Validate fill event
                assert fill_event.order_id, "Order ID should be set"
                assert fill_event.trade_id, "Trade ID should be set"
                assert fill_event.amount > 0, "Amount should be positive"
                assert fill_event.price > 0, "Price should be positive"

                logger.info(f"✅ {test_case['name']}: {fill_event.amount} @ {fill_event.price}")

                self.test_results.append({)}
                    "test": f"Fill Event Parsing - {test_case['name']}",
                    "status": "PASS",
                    "details": f"Parsed {fill_event.amount} @ {fill_event.price}"
                })

            except Exception as e:
                logger.error(f"❌ {test_case['name']} failed: {e}")
                self.test_results.append({)}
                    "test": f"Fill Event Parsing - {test_case['name']}",
                    "status": "FAIL",
                    "details": str(e)
                })

    async def test_partial_fill_handling(self):
        """Test partial fill handling and retry logic."""
        logger.info("🔄 Testing partial fill handling...")

        # Create a test order state
        order_id = "test_partial_order_123"
        order_state = OrderState()
            order_id=order_id,
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            original_amount=Decimal("0.1")  # 0.1 BTC
        )

        self.fill_handler.active_orders[order_id] = order_state

        # Simulate partial fill
        partial_fill_data = {}
            "orderId": order_id,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "fills": []
                {}
                    "tradeId": "partial_fill_1",
                    "qty": "0.06",  # 60% filled
                    "price": "50000.0",
                    "commission": "0.00003",
                    "commissionAsset": "BTC",
                    "takerOrMaker": "taker"
                }
            ]
        }

        try:
            # Process partial fill
            result = await self.fill_handler.handle_partial_fill(order_id, partial_fill_data)

            # Check results
            assert result["status"] in ["partial_fill_processed", "partial_fill_retry_scheduled"], \
                f"Unexpected status: {result['status']}"

            # Check order state
            updated_order = self.fill_handler.get_order_state(order_id)
            assert updated_order.fill_percentage == 60.0, f"Expected 60% fill, got {updated_order.fill_percentage}%"
            assert updated_order.status == FillStatus.PARTIAL, "Order should be in partial status"

            logger.info(f"✅ Partial fill handled: {updated_order.fill_percentage}% filled")

            self.test_results.append({)}
                "test": "Partial Fill Handling",
                "status": "PASS",
                "details": f"{updated_order.fill_percentage}% filled, status: {result['status']}"
            })

        except Exception as e:
            logger.error(f"❌ Partial fill handling failed: {e}")
            self.test_results.append({)}
                "test": "Partial Fill Handling",
                "status": "FAIL",
                "details": str(e)
            })

    async def test_order_state_management(self):
        """Test order state management and updates."""
        logger.info("📊 Testing order state management...")

        order_id = "test_order_state_456"

        # Create initial order state
        order_state = OrderState()
            order_id=order_id,
            symbol="ETHUSDT",
            side="sell",
            order_type="limit",
            original_amount=Decimal("1.0")  # 1 ETH
        )

        self.fill_handler.active_orders[order_id] = order_state

        # Simulate multiple fills
        fills = []
            {"qty": "0.3", "price": "3000.0", "commission": "0.009", "tradeId": "fill_1"},
            {"qty": "0.4", "price": "3001.0", "commission": "0.012", "tradeId": "fill_2"},
            {"qty": "0.3", "price": "3002.0", "commission": "0.009", "tradeId": "fill_3"}
        ]

        try:
            for i, fill_data in enumerate(fills):
                fill_event_data = {}
                    "orderId": order_id,
                    "symbol": "ETHUSDT",
                    "side": "SELL",
                    "fills": [fill_data]
                }

                await self.fill_handler.process_fill_event(fill_event_data)

            # Check final order state
            final_order = self.fill_handler.get_order_state(order_id)

            assert final_order.is_complete, "Order should be complete"
            assert final_order.status == FillStatus.COMPLETE, "Order should be in complete status"
            assert final_order.fill_percentage == 100.0, "Order should be 100% filled"

            # Check average price calculation
            expected_avg_price = (0.3 * 3000 + 0.4 * 3001 + 0.3 * 3002) / 1.0
            assert abs(float(final_order.average_price) - expected_avg_price) < 0.1, \
                f"Average price calculation error: {final_order.average_price} vs {expected_avg_price}"

            logger.info(f"✅ Order state management: {final_order.fill_percentage}% @ {final_order.average_price}")

            self.test_results.append({)}
                "test": "Order State Management",
                "status": "PASS",
                "details": f"100% filled @ {final_order.average_price}"
            })

        except Exception as e:
            logger.error(f"❌ Order state management failed: {e}")
            self.test_results.append({)}
                "test": "Order State Management",
                "status": "FAIL",
                "details": str(e)
            })

    async def test_fill_statistics(self):
        """Test fill statistics and performance tracking."""
        logger.info("📈 Testing fill statistics...")

        try:
            stats = self.fill_handler.get_fill_statistics()

            # Check required fields
            required_fields = []
                "total_fills_processed", "total_retries", "total_fees",
                "active_orders", "completed_orders", "partial_orders"
            ]

            for field in required_fields:
                assert field in stats, f"Missing field: {field}"

            logger.info(f"✅ Fill statistics: {stats['total_fills_processed']} fills, ")
                       f"{stats['total_retries']} retries, {stats['total_fees']} fees")

            self.test_results.append({)}
                "test": "Fill Statistics",
                "status": "PASS",
                "details": f"{stats['total_fills_processed']} fills processed"
            })

        except Exception as e:
            logger.error(f"❌ Fill statistics failed: {e}")
            self.test_results.append({)}
                "test": "Fill Statistics",
                "status": "FAIL",
                "details": str(e)
            })

    async def test_state_persistence(self):
        """Test state export and import functionality."""
        logger.info("💾 Testing state persistence...")

        try:
            # Export current state
            exported_state = self.fill_handler.export_state()

            # Validate exported state structure
            assert "active_orders" in exported_state, "Missing active_orders in export"
            assert "fill_history" in exported_state, "Missing fill_history in export"
            assert "statistics" in exported_state, "Missing statistics in export"

            # Create new fill handler and import state
            new_fill_handler = await create_fill_handler()
            new_fill_handler.import_state(exported_state)

            # Compare statistics
            original_stats = self.fill_handler.get_fill_statistics()
            imported_stats = new_fill_handler.get_fill_statistics()

            assert original_stats["total_fills_processed"] == imported_stats["total_fills_processed"], \
                "Fill count mismatch after import"

            logger.info(f"✅ State persistence: {len(exported_state['active_orders'])} orders exported/imported")

            self.test_results.append({)}
                "test": "State Persistence",
                "status": "PASS",
                "details": f"{len(exported_state['active_orders'])} orders persisted"
            })

        except Exception as e:
            logger.error(f"❌ State persistence failed: {e}")
            self.test_results.append({)}
                "test": "State Persistence",
                "status": "FAIL",
                "details": str(e)
            })

    async def test_exchange_manager_integration(self):
        """Test integration with secure exchange manager."""
        logger.info("🔗 Testing exchange manager integration...")

        try:
            # Test fill handler availability
            await self.exchange_manager._initialize_fill_handler()

            if self.exchange_manager.fill_handler:
                logger.info("✅ Fill handler integrated with exchange manager")

                # Test fill statistics through exchange manager
                stats = await self.exchange_manager.get_fill_statistics()
                assert "total_fills_processed" in stats, "Fill statistics not available"

                self.test_results.append({)}
                    "test": "Exchange Manager Integration",
                    "status": "PASS",
                    "details": "Fill handler successfully integrated"
                })
            else:
                logger.warning("⚠️ Fill handler not available in exchange manager")
                self.test_results.append({)}
                    "test": "Exchange Manager Integration",
                    "status": "SKIP",
                    "details": "Fill handler not available"
                })

        except Exception as e:
            logger.error(f"❌ Exchange manager integration failed: {e}")
            self.test_results.append({)}
                "test": "Exchange Manager Integration",
                "status": "FAIL",
                "details": str(e)
            })

    async def run_all_tests(self):
        """Run all tests."""
        logger.info("🚀 Starting Fill Handler Integration Tests")
        logger.info("=" * 60)

        # Setup
        if not await self.setup():
            logger.error("❌ Test setup failed")
            return

        # Run tests
        await self.test_fill_event_parsing()
        await self.test_partial_fill_handling()
        await self.test_order_state_management()
        await self.test_fill_statistics()
        await self.test_state_persistence()
        await self.test_exchange_manager_integration()

        # Report results
        self.report_results()

    def report_results(self):
        """Report test results."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed_tests = sum(1 for result in self.test_results if result["status"] == "FAIL")
        skipped_tests = sum(1 for result in self.test_results if result["status"] == "SKIP")

        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests} ✅")
        logger.info(f"Failed: {failed_tests} ❌")
        logger.info(f"Skipped: {skipped_tests} ⚠️")

        # Show detailed results
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
            logger.info(f"{status_icon} {result['test']}: {result['details']}")

        # Overall result
        if failed_tests == 0:
            logger.info("\n🎉 ALL TESTS PASSED! Fill handler integration is working correctly.")
        else:
            logger.warning(f"\n⚠️ {failed_tests} tests failed. Please check the implementation.")

        logger.info("=" * 60)


async def main():
    """Main test runner."""
    if not MODULES_AVAILABLE:
        logger.error("❌ Required modules not available. Please install dependencies.")
        return

    test_suite = FillHandlerIntegrationTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 