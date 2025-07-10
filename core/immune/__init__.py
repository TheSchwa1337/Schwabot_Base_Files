from .qsc_gate import QSCGate

__all__ = ["QSCGate"]

# Optionally, add immune system factory or registration hooks here

def create_immune_gate(*args, **kwargs):
    """Factory for QSCGate (immune system gate)"""
    return QSCGate(*args, **kwargs) 