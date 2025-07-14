import asyncio
import hashlib
import json
import random
import threading
import time
from dataclasses import asdict
from typing import Any, Dict

import websockets
from aleph_unitizer_lib import AlephUnitizer, TesseractPortal
from mathlib_v2 import CoreMathLibV2
from rittle_gemm import RingLayer, RittleGEMM
from schwabot_stop import SchwabotStopBook, StopPatternState


class SchwabootDashboardEngine:
    def __init__(self, websocket_port: int = 8765):
        # Initialize core modules
        self.math_lib = CoreMathLibV2()
        self.stop_book = SchwabotStopBook()
        self.rittle_gemm = RittleGEMM()
        self.aleph_unitizer = AlephUnitizer("dashboard_memory")
        self.tesseract_portal = TesseractPortal(self.aleph_unitizer)

        # State management
        self.websocket_port = websocket_port
        self.connected_clients = set()
        self.is_running = False
        self.data_thread = None
        self.loop = asyncio.get_event_loop()  # Ensure an event loop is available

        # Market data simulation
        self.current_price = 50000.0
        self.price_history = []
        self.volume_history = []
        self.high_history = []
        self.low_history = []

        # TPF State
        self.tpf_state = "INITIALIZING"
        self.paradox_visible = False
        self.stabilized = False
        self.phase = 0  # To simulate phase progression

        # Signal tracking
        self.active_signals = []
        self.hash_stream = []
        self.timing_hashes = []

        # Ring values (RITTLE-GEMM)
        self.ring_values = {}
            "R1": 0.0,
            "R2": 0.0,
            "R3": 0.0,
            "R4": 0.0,
            "R5": 0.0,
            "R6": 0.0,
            "R7": 0.0,
            "R8": 0.0,
            "R9": 0.0,
            "R10": 0.0,
        }
        # Stop Patterns
        self.stop_patterns = []

        # Performance tracking
        self.performance_metrics = {}
            "total_signals": 0,
            "successful_signals": 0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }
        self.mock_detonation_active = False  # Visual flag for detonation

    async def register_client(self, websocket, path):
        """Register a new WebSocket client"""
        self.connected_clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.connected_clients)}")

        # Send initial state
        await self.send_initial_state(websocket)

        try:
            # Handle incoming messages
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected. Total clients: {len(self.connected_clients)}")
            pass
        finally:
            self.connected_clients.discard(websocket)

    async def send_initial_state(self, websocket):
        """Send the current comprehensive state to a newly connected client."""
        initial_state_data = {}
            "update_type": "initial_state",
            "data": {}
                "market_data": {}
                    "price": self.current_price,
                    "volume": self.volume_history[-1] if self.volume_history else 0,
                    "rsi": self.math_lib.calculate_rsi(np.array(self.price_history))
                    if len(self.price_history) > self.math_lib.rsi_period
                    else 50,
                    "entropy": self.math_lib.calculate_entropy()
                        np.array(self.price_history)
                    )
                    if self.price_history
                    else 0.5,
                    "drift": np.sin(self.phase * 0.1) * 2,  # Simulated drift
                    "vwap": self.math_lib.calculate_vwap()
                        np.array(self.price_history), np.array(self.volume_history)
                    )
                    if self.price_history and self.volume_history
                    else self.current_price,
                    "atr": self.math_lib.calculate_atr()
                        np.array(self.high_history),
                        np.array(self.low_history),
                        np.array(self.price_history),
                    )
                    if len(self.price_history) > self.math_lib.atr_period
                    else 0,
                    "kellyFraction": 0.25,  # Placeholder
                },
                "ring_values": self.ring_values,
                "hash_stream": self.hash_stream,
                "timing_hashes": self.timing_hashes,
                "glyph_signals": self.active_signals,
                # Convert dataclasses to dicts
                "stop_patterns": []
                    asdict(p) for p in self.stop_book.get_active_patterns()
                ],
                "price_history": []
                    {"timestamp": i, "price": p, "vwap": vwap}
                    for i, (p, vwap) in enumerate()
                        zip()
                            self.price_history,
                            self.math_lib.calculate_vwap()
                                np.array(self.price_history),
                                np.array(self.volume_history),
                                return_array=True,
                            ).tolist(),
                        )
                    )
                ]
                if self.price_history and self.volume_history
                else [],
                "tpf_state": self.tpf_state,
                "paradox_visible": self.paradox_visible,
                "stabilized": self.stabilized,
                "phase": self.phase,
                "performance_metrics": self.performance_metrics,
            },
        }
        await websocket.send(json.dumps(initial_state_data))

    async def handle_client_message(self, websocket, message: str):
        """Handle incoming commands from the frontend."""
        try:
            command = json.loads(message)
            command_type = command.get("command_type")
            data = command.get("data", {})

            print(f"Received command: {command_type}")

            if command_type == "generate_signal":
                # Backend logic to generate a signal based on current market data
                # For now, simulate a signal and send it back
                signal_type = ()
                    "BUY"
                    if data.get("rsi", 0) < 40
                    else "SELL"
                    if data.get("rsi", 0) > 60
                    else "HOLD"
                )
                confidence = data.get("confidence", 0.7)  # Placeholder

                # Simulate a glyph signal
                glyph_signal = {}
                    "timestamp": time.time() * 1000,
                    "type": signal_type,
                    "confidence": confidence,
                    "price": data.get("price", self.current_price),
                    "tpfState": self.tpf_state,
                    "hashTrigger": hashlib.sha256()
                        str(time.time()).encode()
                    ).hexdigest()[:8],
                }
                self.active_signals.append(glyph_signal)
                await self.broadcast()
                    {"update_type": "glyph_signals", "data": glyph_signal}
                )

            elif command_type == "trigger_detonation":
                print("1337 P4TT3RN D3T0N4T10N PR0T0C0L Triggered!")
                self.mock_detonation_active = True  # Set flag for visual effect
                # In a real scenario, this would kick off complex backend logic
                await self.broadcast()
                    {"update_type": "detonation_status", "data": {"active": True}}
                )
                await asyncio.sleep(3)  # Simulate detonation process
                self.mock_detonation_active = False
                await self.broadcast()
                    {"update_type": "detonation_status", "data": {"active": False}}
                )

            elif command_type == "clear_hash_stream":
                self.hash_stream = []
                # Send empty array
                await self.broadcast({"update_type": "hash_stream", "data": []})

            elif command_type == "clear_glyph_signals":
                self.active_signals = []
                # Send empty array
                await self.broadcast({"update_type": "glyph_signals", "data": []})

            else:
                print(f"Unknown command type: {command_type}")

        except json.JSONDecodeError:
            print(f"Received invalid JSON: {message}")
        except Exception as e:
            print(f"Error handling client message: {e}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if self.connected_clients:
            disconnected_clients = set()
            for websocket in self.connected_clients:
                try:
                    await websocket.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(websocket)
                except Exception as e:
                    print(f"Error sending to client: {e}")
                    disconnected_clients.add(websocket)
            self.connected_clients.difference_update(disconnected_clients)

    async def data_update_loop(self):
        """Continuously generate and send data updates to connected clients."""
        while self.is_running:
            # Simulate market data
            price_change = (random.random() - 0.5) * 100
            self.current_price += price_change
            self.current_price = max(100.0, self.current_price)  # Keep price realistic

            volume_change = (random.random() - 0.5) * 50
            current_volume = max()
                100,
                (self.volume_history[-1] if self.volume_history else 1000)
                + volume_change,
            )

            self.price_history.append(self.current_price)
            self.volume_history.append(current_volume)
            self.high_history.append(self.current_price + abs(price_change * 0.5))
            self.low_history.append(self.current_price - abs(price_change * 0.5))

            # Keep history to a manageable size
            max_history = 200
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.high_history = self.high_history[-max_history:]
            self.low_history = self.low_history[-max_history:]

            # Calculate indicators using CoreMathLibV2
            rsi = 50.0
            entropy = 0.5
            vwap = self.current_price
            atr = 0.0

            if len(self.price_history) > self.math_lib.rsi_period:
                rsi = self.math_lib.calculate_rsi(np.array(self.price_history))
            if len(self.price_history) > 1:
                entropy = self.math_lib.calculate_entropy(np.array(self.price_history))
            if len(self.price_history) > 0 and len(self.volume_history) > 0:
                vwap = self.math_lib.calculate_vwap()
                    np.array(self.price_history), np.array(self.volume_history)
                )
            if len(self.high_history) > self.math_lib.atr_period:
                atr = self.math_lib.calculate_atr()
                    np.array(self.high_history),
                    np.array(self.low_history),
                    np.array(self.price_history),
                )

            market_data_update = {}
                "price": self.current_price,
                "volume": current_volume,
                "rsi": rsi,
                "entropy": entropy,
                "drift": np.sin(time.time() * 0.01) * 2,  # Simulating drift
                "vwap": vwap,
                "atr": atr,
                # Simulated kelly
                "kellyFraction": max(0, min(1, 0.25 + (random.random() - 0.5) * 0.1)),
            }
            await self.broadcast()
                {"update_type": "market_data", "data": market_data_update}
            )

            # Simulate Hash Stream
            hash_value = hashlib.sha256(str(time.time()).encode()).hexdigest()
            entropy_tag = random.randint(0, 144)
            hash_data = {}
                "timestamp": time.time() * 1000,
                "hash": hash_value,
                "entropy": entropy_tag,
                "confidence": random.random(),
                "pattern": [random.randint(0, 15) for _ in range(8)],
            }
            self.hash_stream.append(hash_data)
            self.hash_stream = self.hash_stream[-50:]  # Keep last 50
            await self.broadcast({"update_type": "hash_stream", "data": hash_data})

            # Simulate Timing Hashes (from TPF, State)
            self.phase = (self.phase + 1) % 100

            # Update TPF state logic (can be driven by calculated metrics, later)
            if self.phase == 30:
                self.tpf_state = "PARADOX_DETECTED"
                self.paradox_visible = True
                self.stabilized = False
            elif self.phase == 70:
                self.tpf_state = "TPF_STABILIZED"
                self.stabilized = True
                self.paradox_visible = False
            elif self.phase == 99:
                self.tpf_state = "INITIALIZING"
                self.paradox_visible = False
                self.stabilized = False

            timing_hash_data = {}
                "timestamp": time.time() * 1000,
                "hash": hashlib.sha256(str(self.current_price).encode()).hexdigest(),
                "state": self.tpf_state,
            }
            self.timing_hashes.append(timing_hash_data)
            self.timing_hashes = self.timing_hashes[-20:]  # Keep last 20
            await self.broadcast()
                {"update_type": "timing_hashes", "data": timing_hash_data}
            )

            # Broadcast TPF state update for the paradox visualizer
            await self.broadcast()
                {}
                    "update_type": "tpf_state_update",
                    "data": {}
                        "tpf_state": self.tpf_state,
                        "paradox_visible": self.paradox_visible,
                        "stabilized": self.stabilized,
                        "phase": self.phase,
                    },
                }
            )

            # Simulate RITTLE-GEMM Ring Values
            for ring in self.ring_values:
                self.ring_values[ring] = max()
                    -2.0,
                    min()
                        2.0,
                        self.ring_values[ring] * 0.9 + (random.random() - 0.5) * 0.2,
                    ),
                )
            await self.broadcast()
                {"update_type": "ring_values", "data": self.ring_values}
            )

            # Simulate Stop Patterns (simplified)
            if random.random() < 0.1:  # 10% chance to update stop patterns
                # Clear and re-add for simplicity, or manage actual patterns
                self.stop_patterns = []
                num_patterns = random.randint(0, 3)
                for _ in range(num_patterns):
                    state = random.choice()
                        []
                            StopPatternState.ACTIVE,
                            StopPatternState.TRIGGERED,
                            StopPatternState.WARNING,
                        ]
                    )
                    self.stop_patterns.append()
                        {}
                            "id": hashlib.sha256()
                                str(random.random()).encode()
                            ).hexdigest()[:6],
                            "state": state.value,
                        }
                    )
                await self.broadcast()
                    {"update_type": "stop_patterns", "data": self.stop_patterns}
                )

            await asyncio.sleep(0.2)  # Update every 200ms

    def start(self):
        """Starts the WebSocket server and data update loop."""
        self.is_running = True
        print(f"Starting Schwaboot Dashboard Engine on port {self.websocket_port}...")

        # Start the WebSocket server in a separate thread
        self.data_thread = threading.Thread(target=self._run_async_loop)
        self.data_thread.start()

    def _run_async_loop(self):
        """Helper to run the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        try:
            # Start the WebSocket server
            start_server = websockets.serve()
                self.register_client, "localhost", self.websocket_port
            )
            self.loop.run_until_complete(start_server)
            print(f"WebSocket server started on ws://localhost:{self.websocket_port}")

            # Run the data update loop
            self.loop.run_until_complete(self.data_update_loop())
        except Exception as e:
            print(f"Error in async loop: {e}")
        finally:
            self.loop.close()
            print("Asyncio event loop closed.")

    def stop(self):
        """Stops the engine."""
        self.is_running = False
        if self.data_thread and self.data_thread.is_alive():
            print("Stopping Schwaboot Dashboard Engine...")
            # Use run_coroutine_threadsafe to stop the loop from another thread
            future = asyncio.run_coroutine_threadsafe()
                self._shutdown_websocket_server(), self.loop
            )
            future.result()  # Wait for shutdown to complete
            self.data_thread.join()
            print("Schwaboot Dashboard Engine stopped.")

    async def _shutdown_websocket_server(self):
        """Gracefully shut down the WebSocket server."""
        for ws in list(self.connected_clients):  # Create a copy for safe iteration
            try:
                await ws.close()
            except Exception as e:
                print(f"Error closing client websocket: {e}")
        # There's no direct `stop` method for websockets.serve,'
        # it stops when the event loop is closed or all tasks are cancelled.
        # So we just ensure all clients are disconnected.


if __name__ == "__main__":
    # Ensure all necessary modules exist for import
    # This is a temporary check for development. In a proper setup,
    # you'd ensure these paths are correct or handle ModuleNotFoundError.'
    try:
    pass
    except ImportError as e:
        print()
            f"ERROR: Missing a required module. Please ensure all backend modules are in your Python path. {e}"
        )
        print()
            "Expected modules: mathlib_v2.py, schwabot_stop.py, rittle_gemm.py, aleph_unitizer_lib.py"
        )
        exit(1)

    engine = SchwabootDashboardEngine()
    engine.start()

    try:
        # Keep the main thread alive so the background thread can run
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()
        print("Dashboard engine shutdown initiated by user.")
