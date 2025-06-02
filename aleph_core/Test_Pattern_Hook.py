# tests/test_pattern_hook.py

import unittest
from ufs_app import UFSApp
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import sqlite3
from pathlib import Path
import yaml
from kafka import KafkaProducer
import dash
from dash import dcc, html
import plotly.express as px
import pymongo
import numpy as np
import cupy as cp  # For GPU acceleration
from dataclasses import dataclass
import logging
import threading
from queue import Queue
import asyncio

@dataclass
class PatternEvent:
    pattern_name: str
    pattern_hash: str
    metadata: Dict
    timestamp: float
    gpu_processed: bool = False

@dataclass
class ProfitTrajectory:
    entry_price: float
    current_price: float
    target_price: float
    stop_loss: float
    timestamp: float
    confidence: float
    basket_state: Dict[str, float]  # Token balances
    pattern_hash: str
    lattice_phase: str  # ALPHA/BETA/GAMMA/OMEGA

class EventBus:
    def __init__(self, max_size: int = 10000):
        self.queue = Queue(maxsize=max_size)
        self.subscribers = {}
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_events)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _process_events(self):
        while self.running:
            try:
                event = self.queue.get(timeout=1)
                for callback in self.subscribers.get(event.pattern_name, []):
                    callback(event)
            except Queue.Empty:
                continue
                
    def publish(self, event: PatternEvent):
        if not self.queue.full():
            self.queue.put(event)
            
    def subscribe(self, pattern_name: str, callback):
        if pattern_name not in self.subscribers:
            self.subscribers[pattern_name] = []
        self.subscribers[pattern_name].append(callback)

class PatternHookManager:
    def __init__(self):
        self.event_bus = EventBus()
        self.gpu_available = self._check_gpu()
        self.pattern_cache = {}
        self.hook_registry = {}
        
    def _check_gpu(self) -> bool:
        try:
            cp.array([1])
            return True
        except:
            return False
            
    def initialize(self):
        self.event_bus.start()
        
    def register(self, pattern_name: str, callback, use_gpu: bool = False):
        if pattern_name not in self.hook_registry:
            self.hook_registry[pattern_name] = []
        self.hook_registry[pattern_name].append({
            'callback': callback,
            'use_gpu': use_gpu and self.gpu_available
        })
        
    def trigger(self, pattern_name: str, pattern_hash: str, metadata: Dict):
        event = PatternEvent(
            pattern_name=pattern_name,
            pattern_hash=pattern_hash,
            metadata=metadata,
            timestamp=datetime.utcnow().timestamp()
        )
        
        if self.gpu_available:
            # Process pattern matching on GPU
            try:
                gpu_data = cp.array(metadata.get('pattern_data', []))
                # GPU processing logic here
                event.gpu_processed = True
            except Exception as e:
                logging.error(f"GPU processing failed: {e}")
                
        self.event_bus.publish(event)

# Initialize global hook manager
HOOK_MANAGER = PatternHookManager()

class PatternHookTest(unittest.TestCase):
    def setUp(self):
        self.hook_mgr = UFSApp.get_hook_manager()
        self.payloads = []

        def test_callback(pattern_name, pattern_hash, metadata):
            self.payloads.append({
                "pattern_name": pattern_name,
                "pattern_hash": pattern_hash,
                "metadata": metadata
            })

        self.hook_mgr.register("on_pattern_matched", test_callback)

    def test_pattern_matched_hook(self):
        # Simulate pattern match
        self.hook_mgr.trigger(
            "on_pattern_matched",
            pattern_name="XRP_Breakout",
            pattern_hash="abc123",
            metadata={"confidence": 0.98, "matched_nodes": 4}
        )

        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(self.payloads[0]["pattern_name"], "XRP_Breakout")
        self.assertGreaterEqual(self.payloads[0]["metadata"]["confidence"], 0.95)

    def tearDown(self):
        self.payloads.clear()

def register_hook():
    try:
        HOOK_MANAGER.initialize()
        HOOK_MANAGER.register("on_pattern_matched", log_pattern_match)
        print("[Hook] Successfully registered on_pattern_matched hook.")
    except Exception as e:
        print(f"[Error] Failed to register on_pattern_matched hook: {e}")

def log_pattern_match(pattern_name, pattern_hash, metadata):
    try:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pattern": pattern_name,
            "hash": pattern_hash,
            "confidence": metadata.get("confidence", 0),
            "nodes": metadata.get("matched_nodes", 0)
        }

        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as fp:
                data = json.load(fp)
        else:
            data = []

        data.append(entry)

        with open(LOG_FILE, "w") as fp:
            json.dump(data, fp, indent=4)

        print(f"[Logger] Pattern '{pattern_name}' recorded.")
    except Exception as e:
        print(f"[Error] Failed to log pattern match: {e}")

register_hook()

class BasketSwapper:
    def __init__(self):
        self.trajectories = {}
        self.swap_history = []
        self.cooldown_periods = {}
        
    def calculate_swap_opportunity(self, 
                                 current_trajectory: ProfitTrajectory,
                                 market_data: Dict) -> Optional[Dict]:
        """Calculate optimal swap opportunity based on trajectory"""
        if not self._check_cooldown(current_trajectory.pattern_hash):
            return None
            
        # Calculate profit potential
        profit_potential = (current_trajectory.target_price - current_trajectory.current_price) / current_trajectory.current_price
        
        # Check if swap is beneficial
        if profit_potential > 0.01:  # 1% minimum profit threshold
            return {
                'from_token': self._get_current_token(current_trajectory),
                'to_token': self._determine_target_token(current_trajectory, market_data),
                'amount': self._calculate_swap_amount(current_trajectory),
                'expected_profit': profit_potential
            }
        return None
        
    def _check_cooldown(self, pattern_hash: str) -> bool:
        """Check if pattern is in cooldown period"""
        if pattern_hash in self.cooldown_periods:
            cooldown_end = self.cooldown_periods[pattern_hash]
            if datetime.utcnow().timestamp() < cooldown_end:
                return False
        return True
        
    def _get_current_token(self, trajectory: ProfitTrajectory) -> str:
        """Get current token with highest balance"""
        return max(trajectory.basket_state.items(), key=lambda x: x[1])[0]
        
    def _determine_target_token(self, 
                              trajectory: ProfitTrajectory,
                              market_data: Dict) -> str:
        """Determine optimal target token based on market conditions"""
        # Implement token selection logic based on market data
        # This is a simplified version
        tokens = ['XRP', 'USDC', 'BTC', 'ETH']
        return max(tokens, key=lambda t: market_data.get(f'{t}_score', 0))
        
    def _calculate_swap_amount(self, trajectory: ProfitTrajectory) -> float:
        """Calculate optimal swap amount based on risk management"""
        current_token = self._get_current_token(trajectory)
        balance = trajectory.basket_state[current_token]
        return balance * 0.95  # Use 95% of balance for swap

def match_patterns(input_data: dict) -> Optional[dict]:
    for pattern in KNOWN_PATTERNS:
        try:
            if pattern.matches(input_data):
                # Create a dictionary to hold the match data
                match_data = {
                    "pattern_name": pattern.name,
                    "pattern_hash": pattern.get_hash(),
                    "metadata": {
                        "confidence": pattern.confidence,
                        "matched_nodes": pattern.node_count,
                        "timestamp": pattern.timestamp,
                        "pattern_data": pattern.get_pattern_data(),
                        "trajectory": {
                            "entry_price": input_data.get('price', 0),
                            "target_price": pattern.get_target_price(),
                            "stop_loss": pattern.get_stop_loss(),
                            "basket_state": input_data.get('basket_state', {})
                        }
                    }
                }

                # Trigger the hook with the match data
                HOOK_MANAGER.trigger(
                    "on_pattern_matched",
                    pattern_name=match_data["pattern_name"],
                    pattern_hash=match_data["pattern_hash"],
                    metadata=match_data["metadata"]
                )

                return match_data
        except Exception as e:
            print(f"[Error] Failed to process pattern {pattern.name}: {e}")

    return None

def infer_strategy(pattern_name: str, metadata: dict) -> str:
    if "XRP" in pattern_name and metadata["confidence"] > 0.9:
        return "alpha-resolve"
    elif "BTC" in pattern_name and metadata["confidence"] > 0.85:
        return "beta-trade"
    else:
        return "none"

MEMORY_DB = "mongodb://localhost:27017/pattern_memory"

def store_pattern(pattern_name, hash_str, strategy_hint, confidence):
    try:
        client = pymongo.MongoClient(MEMORY_DB)
        db = client["pattern_memory"]
        collection = db["patterns"]

        collection.insert_one({
            "pattern": pattern_name,
            "strategy": strategy_hint,
            "confidence": confidence,
            "recorded": datetime.utcnow().isoformat()
        })

        print(f"[Memory] Stored pattern '{pattern_name}' ➝ strategy '{strategy_hint}'")
    except Exception as e:
        print(f"[Error] Failed to store pattern in memory: {e}")

def retrieve_strategy_for_pattern(hash_str):
    try:
        client = pymongo.MongoClient(MEMORY_DB)
        db = client["pattern_memory"]
        collection = db["patterns"]

        result = collection.find_one({"hash": hash_str})
        return result.get("strategy", "none")
    except Exception as e:
        print(f"[Error] Failed to retrieve strategy for pattern: {e}")

def convert_logs_to_yaml():
    from core.pattern_memory import retrieve_strategy_for_pattern

    try:
        conn = sqlite3.connect(MEMORY_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patterns")
        data = cursor.fetchall()

        with open("logs/pattern_logs.yaml", "w") as yf:
            for entry in data:
                strategy_hint = retrieve_strategy_for_pattern(entry[1])
                yaml.dump({
                    "timestamp": entry[3],
                    "pattern": entry[0],
                    "hash": entry[1],
                    "confidence": entry[2],
                    "strategy": strategy_hint
                }, yf, sort_keys=False)
        print("[Export] YAML log ready at logs/pattern_logs.yaml")
    except Exception as e:
        print(f"[Error] Failed to export logs: {e}")

convert_logs_to_yaml()

def send_pattern_match_to_kafka(pattern_name, pattern_hash, metadata):
    try:
        producer = KafkaProducer(bootstrap_servers='localhost:9092')
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "pattern": pattern_name,
            "hash": pattern_hash,
            "confidence": metadata.get("confidence", 0),
            "nodes": metadata.get("matched_nodes", 0)
        }
        producer.send('pattern_matches', value=message)
        producer.flush()
        print(f"[Kafka] Sent pattern match to Kafka: {message}")
    except Exception as e:
        print(f"[Error] Failed to send pattern match to Kafka: {e}")

def register_kafka_hook():
    try:
        HOOK_MANAGER.register("on_pattern_matched", lambda *args, **kwargs: send_pattern_match_to_kafka(*args, **kwargs))
        print("[Kafka] Successfully registered on_pattern_matched Kafka hook.")
    except Exception as e:
        print(f"[Error] Failed to register on_pattern_matched Kafka hook: {e}")

register_kafka_hook()

app = dash.Dash(__name__)

@app.callback(
    dash.dependencies.Output('pattern-heatmap', 'figure'),
    [dash.dependencies.Input('pattern-memory-db', 'data')])
def update_heatmap(pattern_memory):
    # Process pattern memory data to generate a heatmap
    fig = px.scatter_matrix(
        pattern_memory,
        dimensions=['confidence', 'nodes'],
        color='strategy',
        title='Pattern Heatmap'
    )
    return fig

app.layout = html.Div([
    dcc.Graph(id='pattern-heatmap'),
    dcc.Interval(id='interval-component', interval=60000, n_intervals=0),
    dcc.Store(id='pattern-memory-db')
])

if __name__ == '__main__':
    app.run_server(debug=True)
