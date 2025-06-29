import asyncio
import collections
import dataclasses as dc
import hashlib
import json
import time


@dc.dataclass
class ShareEvent:
    pool: str
    diff: float
    ts: float


class StratumSniffer:
    def __init__(self, host: str, port: int, pool_name: str):
        self.host = host
        self.port = port
        self.pool_name = pool_name
        self.share_q = collections.deque(maxlen=120)  # Stores recent share events
        self.last_block_notify_ts = None

    async def run(self, event_cb=None):
        """"""
        Connects to a Stratum mining pool and sniffs for share and difficulty events.
        Calls event_cb with ShareEvent on each share.
        """"""
        try:
            reader, writer = await asyncio.open_connection(self.host, self.port)
            # Simple subscribe; many pools accept json-RPC login {id:1, method:"mining.subscribe"}
            # For a real implementation, you'd need to handle authentication (mining.authorize)'
            subscribe_msg = {"id": 1, "method": "mining.subscribe", "params": []}
            writer.write(json.dumps(subscribe_msg).encode() + b"\n")
            await writer.drain()

            print(f"Connected to {self.pool_name} ({self.host}:{self.port})")

            # Read initial response (usually subscription details)
            first_line = await reader.readline()
            # print(f"Initial response from {self.pool_name}: {first_line.decode().strip()}")

            while True:
                line = await reader.readline()
                if not line:  # Connection closed
                    print(f"Connection to {self.pool_name} closed.")
                    break

                try:
                    msg = json.loads(line.decode().strip())
                    m = msg.get("method")

                    if m == "mining.set_difficulty":
                        diff = float(msg["params"][0])
                        # print(f"[{self.pool_name}] New difficulty: {diff}")
                        # You might want to publish this event to an internal bus as well
                    elif m == "mining.notify":
                        # params[4] is often the share difficulty in many implementations
                        # For simplicity, we'll use a dummy diff or a stored one if available.'
                        # In a real scenario, you'd parse params for actual block details and difficulty.'
                        current_diff = 1.0  # Placeholder, should come from mining.set_difficulty
                        share_event = ShareEvent(self.pool_name, current_diff, time.time())
                        self.share_q.append(share_event)
                        # print(f"[{self.pool_name}] Mining notify - Share appended. Q size: {len(self.share_q)}")
                        if event_cb:
                            await event_cb(share_event)
                        self.last_block_notify_ts = time.time()
                    # Add more handlers for other stratum methods like mining.submit, mining.set_target etc.
                except json.JSONDecodeError:
                    print(f"[{self.pool_name}] JSON Decode Error: {line.decode().strip()}")
                except Exception as e:
                    print(f"[{self.pool_name}] Error processing message: {e} | Message: {line.decode().strip()}")
        except Exception as e:
            print(f"StratumSniffer for {self.pool_name} encountered an error: {e}")

    def get_recent_shares(self) -> collections.deque:
        """"""
        Returns the deque of recent share events.
        """"""
        return self.share_q

    def get_last_block_notify_time(self) -> float | None:
        """"""
        Returns the timestamp of the last mining.notify event.
        """"""
        return self.last_block_notify_ts


# Example usage (for testing/demonstration, will be removed in final integration)
if __name__ == "__main__":

    async def test_event_callback(event):
        print(f"Received Share Event: Pool={event.pool}, Diff={event.diff}, TS={event.ts}")

    async def main():
        # Replace with actual Stratum pool details
        # F2Pool BTC Stratum server, port for BTC (check their official docs for current info)
        f2pool_sniffer = StratumSniffer("stratum.f2pool.com", 3333, "F2Pool")
        # foundry_sniffer = StratumSniffer("stratum-na.foundrypool.com", 3335, "FoundryPool")

        # Run sniffers concurrently
        await asyncio.gather()
            f2pool_sniffer.run(test_event_callback),
                # foundry_sniffer.run(test_event_callback),
                    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Sniffer stopped.")
