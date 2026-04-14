"""Tests for span representation layers."""

import mlx.core as mx
import pytest

from gliner2_mlx.span_rep import SpanMarkerV0, SpanRepLayer, extract_elements


class TestExtractElements:
    def test_basic_extraction(self):
        seq = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
        indices = mx.array([[0, 2, 1]])  # (1, 3)
        result = extract_elements(seq, indices)
        mx.eval(result)
        assert result.shape == (1, 3, 2)
        assert result[0, 0, 0].item() == 1.0
        assert result[0, 1, 0].item() == 5.0
        assert result[0, 2, 0].item() == 3.0

    def test_batch_extraction(self):
        B, L, D = 2, 5, 4
        seq = mx.random.normal((B, L, D))
        indices = mx.array([[0, 1, 2], [3, 4, 0]])  # (2, 3)
        result = extract_elements(seq, indices)
        mx.eval(result)
        assert result.shape == (2, 3, 4)


class TestSpanMarkerV0:
    def test_output_shape(self):
        hidden_size = 32
        max_width = 4
        marker = SpanMarkerV0(hidden_size, max_width, dropout=0.0)

        B, L = 2, 8
        h = mx.random.normal((B, L, hidden_size))
        N = L * max_width
        # Build span indices
        span_idx = mx.zeros((B, N, 2), dtype=mx.int32)
        out = marker(h, span_idx)
        mx.eval(out)
        assert out.shape == (B, L, max_width, hidden_size)

    def test_single_sample(self):
        hidden_size = 16
        max_width = 3
        marker = SpanMarkerV0(hidden_size, max_width, dropout=0.0)

        h = mx.random.normal((1, 5, hidden_size))
        span_idx = mx.zeros((1, 15, 2), dtype=mx.int32)
        out = marker(h, span_idx)
        mx.eval(out)
        assert out.shape == (1, 5, max_width, hidden_size)


class TestSpanRepLayer:
    def test_markerV0_mode(self):
        layer = SpanRepLayer(hidden_size=32, max_width=4, span_mode="markerV0", dropout=0.0)
        h = mx.random.normal((1, 6, 32))
        span_idx = mx.zeros((1, 24, 2), dtype=mx.int32)
        out = layer(h, span_idx)
        mx.eval(out)
        assert out.shape == (1, 6, 4, 32)

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Only 'markerV0'"):
            SpanRepLayer(hidden_size=32, max_width=4, span_mode="query")
