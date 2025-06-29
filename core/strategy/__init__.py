# This is a package initializer file for the 'core/strategy' directory. 

from .glyph_strategy_core import GlyphStrategyCore
from .entry_exit_portal import EntryExitPortal
from .flip_switch_logic_lattice import FlipSwitchLogicLattice
from .loss_anticipation_curve import LossAnticipationCurve
from .zygot_zalgo_entropy_dual_key_gate import ZygotZalgoEntropyDualKeyGate
from .volume_weighted_hash_oscillator import VolumeWeightedHashOscillator
from .multi_phase_strategy_weight_tensor import MultiPhaseStrategyWeightTensor, MarketPhase
from .glyph_gate_engine import GlyphGateEngine

def create_glyph_trading_system(
    simulation_mode: bool = True,
    enable_fractal_memory: bool = True,
    enable_gear_shifting: bool = True,
    enable_risk_management: bool = True,
    enable_portfolio_tracking: bool = True
):
    """Factory function to create an integrated glyph trading system."""
    glyph_core = GlyphStrategyCore(
        enable_fractal_memory=enable_fractal_memory,
        enable_gear_shifting=enable_gear_shifting
    )
    portal = EntryExitPortal(
        glyph_core=glyph_core,
        enable_risk_management=enable_risk_management,
        enable_portfolio_tracking=enable_portfolio_tracking
    )
    return glyph_core, portal 