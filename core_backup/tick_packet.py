class TickPacket:
    def __init__()
        self,
            price: float,
                volume: float,
                timestamp: float,
                delta: float,
                entropy: float,
                profit_bias: float,
                coherence: float,
                eco_signal: float,
                ):
        self.price = price
        self.volume = volume
        self.timestamp = timestamp
        self.delta = delta
        self.entropy = entropy
        self.profit_bias = profit_bias
        self.coherence = coherence
        self.eco_signal = eco_signal

    def to_dict(self):
        return self.__dict__
