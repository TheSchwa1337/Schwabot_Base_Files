import pytest
from core.hooks.phase_engine_hooks import SchwabotPhaseEngineHooks
from core.hooks.oracle_bus import OracleBus
from core.hooks.zygot_shell import ZygotShell
from core.hooks.gan_filter import GANFilter

class TestSchwabotIntegration:
    @pytest.mark.asyncio
    async def test_zygot_integration(self):
        """Test ZygotShell integration with hooks"""
        hooks = SchwabotPhaseEngineHooks(OracleBus())
        zygot = ZygotShell({})
        
        # Test drift resonance calculation
        hook_id = await hooks.on_tick_start(32000.0, 1)
        assert hooks._phase_state['drift_resonance'] is not None
        
    @pytest.mark.asyncio
    async def test_gan_integration(self):
        """Test GAN filter integration"""
        hooks = SchwabotPhaseEngineHooks(OracleBus())
        gan = GANFilter({})
        
        # Test anomaly detection
        hook_id = await hooks.on_tick_start(32000.0, 1)
        assert hooks._phase_state['anomaly_score'] is not None
