from .galileo_tensor_field import GalileoTensorField

__all__ = ["GalileoTensorField"]

def create_entropy_field(*args, **kwargs):
    """Factory for GalileoTensorField (entropy field)"""
    return GalileoTensorField(*args, **kwargs) 