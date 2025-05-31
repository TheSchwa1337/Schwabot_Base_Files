class FillConjunctionEngine:
    def __init__(self, harmony_memory):
        self.harmony_memory = harmony_memory
        self.live_book = {}
        self.ghost_log = []
        self.phantom_log = []

    def ingest_ncco(self, ncco):
        # Example: store NCCO in live_book
        self.live_book[ncco.id] = ncco

    def commit_order(self, pre_order_id):
        # Example: move from live_book to ghost_log
        if pre_order_id in self.live_book:
            self.ghost_log.append(self.live_book.pop(pre_order_id))

    def handle_fill_state(self, order_id, fill_state, fill_price, fill_size, fill_time):
        # Example: move from ghost_log to phantom_log
        for i, ncco in enumerate(self.ghost_log):
            if ncco.id == order_id:
                self.phantom_log.append(self.ghost_log.pop(i))
                break 