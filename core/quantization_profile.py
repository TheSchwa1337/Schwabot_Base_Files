from dataclasses import dataclass

@dataclass
class QuantizationProfile:
    """Data-class representing the parameters that govern lattice quantisation.

    This lives in its own module so that tests and other code can import it
    without depending on EnhancedFractalCore's internal definition.
    """
    decay_power: float = 1.5
    terms: int = 12
    dimension: int = 8
    epsilon_q: float = 0.003
    precision: float = 1e-3 