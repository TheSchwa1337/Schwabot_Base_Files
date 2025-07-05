from core.unified_math_system import generate_unified_hash
from core.unified_trade_router import UnifiedTradeRouter


class VisualExecutionNode:
    """
    Visual Execution Node for GUI integration.
    Generates a visual packet and routes the trade signal.
    """

    def __init__(self, asset: str, price: float):
        self.asset = asset
        self.price = price
        self.router = UnifiedTradeRouter()

    def generate_visual_packet(self) -> dict:
        """Generate a packet with hash and display string for GUI."""
        signal = {"asset": self.asset, "price": self.price, "entropy": 0.88, "drift": 0.04}
        signal_hash = generate_unified_hash(
            [signal["asset"], signal["price"], signal["entropy"], signal["drift"]],
            time_slot="15min",
        )
        return {
            "hash": signal_hash,
            "visual_display": f"Signal: {
                self.asset} at ${
                self.price} | E:{
                signal['entropy']} D:{
                    signal['drift']}",
        }

    def execute(self) -> dict:
        """Route the generated visual packet as a trade signal."""
        packet = self.generate_visual_packet()
        # Route using actual price and packet hash
        self.router.route_trade_signal(
            price=self.price, volume=1.0, asset=self.asset, metadata={"packet_hash": packet["hash"]}
        )
        return packet
