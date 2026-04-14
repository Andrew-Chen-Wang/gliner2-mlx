"""Tests for the MLX Extractor model."""

import mlx.core as mx
import pytest

from gliner2_mlx.model import Extractor


@pytest.fixture
def small_extractor(small_deberta_config):
    """Create a small Extractor for testing."""
    return Extractor(
        encoder_config=small_deberta_config,
        max_width=4,
        counting_layer="count_lstm",
    )


class TestExtractor:
    def test_encode_shape(self, small_extractor, sample_input):
        input_ids, attention_mask = sample_input
        out = small_extractor.encode(input_ids, attention_mask)
        mx.eval(out)
        assert out.shape == (2, 16, 64)  # (batch, seq, hidden)

    def test_compute_span_rep_shape(self, small_extractor):
        token_embs = mx.random.normal((10, 64))  # (text_len, hidden)
        result = small_extractor.compute_span_rep(token_embs)
        mx.eval(result["span_rep"])
        mx.eval(result["span_mask"])
        assert result["span_rep"].shape == (10, 4, 64)  # (text_len, max_width, hidden)
        assert result["span_mask"].shape[1] == 10 * 4

    def test_compute_span_rep_batched(self, small_extractor):
        embs = [
            mx.random.normal((8, 64)),
            mx.random.normal((12, 64)),
        ]
        results = small_extractor.compute_span_rep_batched(embs)
        for _i, res in enumerate(results):
            mx.eval(res["span_rep"])
        assert len(results) == 2
        assert results[0]["span_rep"].shape == (8, 4, 64)
        assert results[1]["span_rep"].shape == (12, 4, 64)

    def test_classifier_shape(self, small_extractor):
        x = mx.random.normal((3, 64))
        out = small_extractor.classifier(x)
        mx.eval(out)
        assert out.shape == (3, 1)

    def test_count_pred_shape(self, small_extractor):
        x = mx.random.normal((1, 64))
        out = small_extractor.count_pred(x)
        mx.eval(out)
        assert out.shape == (1, 20)

    def test_count_embed_shape(self, small_extractor):
        pc_emb = mx.random.normal((3, 64))
        out = small_extractor.count_embed(pc_emb, 2)
        mx.eval(out)
        assert out.shape == (2, 3, 64)

    def test_counting_layer_options(self, small_deberta_config):
        for layer_type in ("count_lstm", "count_lstm_moe", "count_lstm_v2"):
            model = Extractor(
                encoder_config=small_deberta_config,
                max_width=4,
                counting_layer=layer_type,
            )
            pc_emb = mx.random.normal((2, 64))
            out = model.count_embed(pc_emb, 3)
            mx.eval(out)
            assert out.shape == (3, 2, 64)

    def test_invalid_counting_layer(self, small_deberta_config):
        with pytest.raises(ValueError, match="Unknown counting_layer"):
            Extractor(
                encoder_config=small_deberta_config,
                max_width=4,
                counting_layer="invalid",
            )
