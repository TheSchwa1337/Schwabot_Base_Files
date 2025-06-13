class SchwabotConfig:
    def __init__(self):
        self.zygot_config = {
            'drift_threshold': 0.5,
            'alignment_threshold': 0.7,
            'shell_radius': 144.44
        }
        self.gan_config = {
            'input_dim': 32,
            'latent_dim': 16,
            'learning_rate': 0.001
        }
        self.hook_config = {
            'ack_timeout': 1.0,
            'max_retries': 3,
            'backoff': 0.1
        }
