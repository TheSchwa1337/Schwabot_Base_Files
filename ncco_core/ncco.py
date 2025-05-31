class NCCO:
    def __init__(self, id: int, price_delta: float, base_price: float, bit_mode: int, score: float = 0.0, pre_commit_id: str = None):
        self.id = id
        self.price_delta = price_delta
        self.base_price = base_price
        self.bit_mode = bit_mode
        self.score = score
        self.pre_commit_id = pre_commit_id

    def __repr__(self):
        return f"NCCO(id={self.id}, price_delta={self.price_delta}, base_price={self.base_price}, bit_mode={self.bit_mode}, score={self.score}, pre_commit_id={self.pre_commit_id})" 