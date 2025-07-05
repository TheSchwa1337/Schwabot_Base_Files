class GlyphRouter:
    def __init__(self):
        self.glyph_map = {
            "accumulate": "🌀",
            "breakout": "⚡",
            "fallback": "🛡️",
            "pump_guard": "🚨",
            "dip_harvest": "🌘",
            "ghost": "👻",
            "loopback": "🔁",
            "deferral": "⏳"
        }

    def get_glyph(self, strategy_name: str) -> str:
        return self.glyph_map.get(strategy_name.lower(), "❔")

    def route_by_vector(self, profit_vector) -> str:
        max_val = max(profit_vector)
        if max_val >= 0.5:
            return "⚡"
        elif max_val >= 0.25:
            return "🌀"
        elif max_val < 0.05:
            return "🛡️"
        return "⏳" 