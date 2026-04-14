"""Span representation layers ported to MLX.

Implements SpanMarkerV0 and SpanRepLayer matching gliner's architecture.
"""

import mlx.core as mx
import mlx.nn as nn


def _create_projection_layer(hidden_size: int, dropout: float, out_dim: int | None = None) -> list:
    """Creates a two-layer projection: Linear -> ReLU -> Dropout -> Linear."""
    if out_dim is None:
        out_dim = hidden_size
    return [
        nn.Linear(hidden_size, out_dim * 4),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim * 4, out_dim),
    ]


class ProjectionLayer(nn.Module):
    """Wrapper around projection layers for nn.Module compatibility."""

    def __init__(self, hidden_size: int, dropout: float, out_dim: int | None = None):
        super().__init__()
        self.layers = _create_projection_layer(hidden_size, dropout, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


def extract_elements(sequence: mx.array, indices: mx.array) -> mx.array:
    """Extract elements from a sequence using indices.

    Args:
        sequence: (B, L, D)
        indices: (B, K)

    Returns:
        (B, K, D)
    """
    B, _L, D = sequence.shape
    K = indices.shape[1]
    # Expand indices to (B, K, D) for gather
    expanded_indices = mx.broadcast_to(indices[:, :, None], (B, K, D))
    return mx.take_along_axis(sequence, expanded_indices.astype(mx.int32), axis=1)


class SpanMarkerV0(nn.Module):
    """Marks and projects span endpoints using MLPs.

    Projects start and end positions separately, concatenates, then
    applies a final projection.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = ProjectionLayer(hidden_size, dropout)
        self.project_end = ProjectionLayer(hidden_size, dropout)
        self.out_project = ProjectionLayer(hidden_size * 2, dropout, hidden_size)

    def __call__(self, h: mx.array, span_idx: mx.array) -> mx.array:
        """Compute span representations.

        Args:
            h: (B, L, D) token representations
            span_idx: (B, N, 2) span indices (start, end)

        Returns:
            (B, L, max_width, D) span representations
        """
        B, L, D = h.shape

        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        cat = nn.relu(mx.concatenate([start_span_rep, end_span_rep], axis=-1))
        return self.out_project(cat).reshape(B, L, self.max_width, D)


class SpanRepLayer(nn.Module):
    """Factory wrapper for span representation layers.

    Only supports 'markerV0' mode (used by GLiNER2).
    """

    def __init__(self, hidden_size: int, max_width: int, span_mode: str = "markerV0", **kwargs):
        super().__init__()
        if span_mode != "markerV0":
            raise ValueError(f"Only 'markerV0' mode is supported, got {span_mode!r}")
        self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, **kwargs)

    def __call__(self, x: mx.array, *args) -> mx.array:
        return self.span_rep_layer(x, *args)
