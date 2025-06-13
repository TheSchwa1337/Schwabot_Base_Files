class SchwabotMetrics:
    def __init__(self):
        self.zygot_metrics = {
            'drift_resonance': [],
            'alignment_score': [],
            'shell_states': []
        }
        self.gan_metrics = {
            'anomaly_scores': [],
            'filter_confidence': []
        }
        self.fill_metrics = {
            'order_latency': [],
            'fill_rates': []
        }
        
    def record_zygot_metric(self, metric_name: str, value: float):
        """Record ZygotShell metric"""
        if metric_name in self.zygot_metrics:
            self.zygot_metrics[metric_name].append(value)
