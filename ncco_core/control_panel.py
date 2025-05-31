class AdvancedControlPanel:
    def batch_adjust(self, nccos, market_signal):
        for ncco in nccos:
            # Example: adjust score based on market signal
            ncco.score *= (1 + market_signal)
        return nccos 