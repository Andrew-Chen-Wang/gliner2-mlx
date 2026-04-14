"""GLiNER2 inference on Apple Silicon via MLX."""

__version__ = "0.1.0"

from .convert import convert_weights as convert_weights
from .engine import GLiNER2MLX as GLiNER2MLX
