class SchwabotStateManager:
    def __init__(self):
        self.zygot_state = None
        self.gan_state = None
        self.fill_state = None
        self.ncco_state = None
        
    def update_zygot_state(self, state):
        """Update ZygotShell state"""
        self.zygot_state = state
        self._propagate_state_change('zygot', state)
        
    def update_gan_state(self, state):
        """Update GAN filter state"""
        self.gan_state = state
        self._propagate_state_change('gan', state)
