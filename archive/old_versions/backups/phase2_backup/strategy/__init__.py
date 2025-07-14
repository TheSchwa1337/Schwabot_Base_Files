"""Module for Schwabot trading system."""

from .entry_exit_portal import EntryExitPortal
from .flip_switch_logic_lattice import FlipSwitchLogicLattice
from .glyph_gate_engine import GlyphGateEngine
from .glyph_strategy_core import GlyphStrategyCore

"""
Strategy Module Package Initializer

This module contains the strategic intelligence components for Schwabot trading.
"""

    try:
    from .entry_exit_portal import EntryExitPortal
        except ImportError:
        EntryExitPortal = None

            try:
            from .flip_switch_logic_lattice import FlipSwitchLogicLattice
                except ImportError:
                FlipSwitchLogicLattice = None

                    try:
                    from .glyph_gate_engine import GlyphGateEngine
                        except ImportError:
                        GlyphGateEngine = None

                            try:
                            from .glyph_strategy_core import GlyphStrategyCore
                                except ImportError:
                                GlyphStrategyCore = None

                                # Version info
                                __version__ = "1.0.0"
                                __author__ = "Schwabot Development Team"

                                # Export list
                                __all__ = []
                                "EntryExitPortal",
                                "FlipSwitchLogicLattice",
                                "GlyphGateEngine",
                                "GlyphStrategyCore",
                                "create_glyph_trading_system",
                                ]


                                def create_glyph_trading_system()
                                enable_fractal_memory = True,
                                enable_gear_shifting = True,
                                enable_risk_management = True,
                                enable_portfolio_tracking = True,
                                    ):
                                    """
                                    Factory function to create an integrated glyph trading system.

                                        Args:
                                        enable_fractal_memory: Enable fractal memory patterns
                                        enable_gear_shifting: Enable dynamic gear shifting
                                        enable_risk_management: Enable risk management features
                                        enable_portfolio_tracking: Enable portfolio tracking

                                            Returns:
                                            Tuple of (glyph_core, portal) components
                                            """
                                                try:
                                                    if GlyphStrategyCore is None or EntryExitPortal is None:
                                                return None, None

                                                glyph_core = GlyphStrategyCore()
                                                enable_fractal_memory = enable_fractal_memory,
                                                enable_gear_shifting = enable_gear_shifting,
                                                )

                                                portal = EntryExitPortal()
                                                glyph_core = glyph_core,
                                                enable_risk_management = enable_risk_management,
                                                enable_portfolio_tracking = enable_portfolio_tracking,
                                                )

                                            return glyph_core, portal

                                                except Exception:
                                                # Fallback to simplified system
                                            return None, None
