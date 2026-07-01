"""Tests for MLX weight quantization in GLiNER2MLX.

The fast unit tests build a tiny Extractor and exercise
``GLiNER2MLX._quantize_model`` directly, so they need no model download.
The slow integration test verifies ``from_pretrained(..., quantize=True)``
end-to-end and is gated behind the ``slow`` marker.
"""

import mlx.nn as nn
import pytest

from gliner2_mlx.engine import GLiNER2MLX
from gliner2_mlx.model import Extractor

# Top-level task-head prefixes that are skipped unless quantize_heads=True.
HEAD_PREFIXES = ("span_rep", "classifier", "count_pred", "count_embed")


def build_extractor(config) -> Extractor:
    """Build a small Extractor from a DebertaV2Config for testing."""
    return Extractor(encoder_config=config, max_width=8, counting_layer="count_lstm")


def quantized_modules(model: Extractor) -> dict[str, nn.Module]:
    """Return {path: module} for every quantized layer in the model."""
    return {
        path: mod for path, mod in model.named_modules() if isinstance(mod, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }


def count_quantized(model: Extractor) -> tuple[int, int]:
    """Return (num QuantizedLinear, num QuantizedEmbedding)."""
    q = quantized_modules(model)
    n_linear = sum(isinstance(m, nn.QuantizedLinear) for m in q.values())
    n_embed = sum(isinstance(m, nn.QuantizedEmbedding) for m in q.values())
    return n_linear, n_embed


class TestQuantizeModel:
    def test_no_quantization_by_default_baseline(self, small_deberta_config):
        """A freshly built model has no quantized layers."""
        model = build_extractor(small_deberta_config)
        assert count_quantized(model) == (0, 0)

    def test_default_quantizes_encoder_and_embeddings_not_heads(self, small_deberta_config):
        """Default config: encoder Linears + embeddings quantized, heads skipped."""
        model = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model,
            q_bits=8,
            q_group_size=64,
            quantize_embeddings=True,
            quantize_heads=False,
        )

        q = quantized_modules(model)
        assert q, "expected some layers to be quantized"

        # The huge win: word embeddings must be quantized.
        assert isinstance(model.encoder.embeddings.word_embeddings, nn.QuantizedEmbedding)

        # Encoder Linear layers are quantized.
        n_linear, n_embed = count_quantized(model)
        assert n_linear > 0
        assert n_embed > 0

        # Task heads must NOT be quantized by default.
        for path in q:
            assert not path.startswith(HEAD_PREFIXES), f"head layer was quantized: {path}"

    def test_rel_embeddings_never_quantized(self, small_deberta_config):
        """Disentangled-attention rel_embeddings must stay full precision.

        Its weight is consumed as a raw matrix (get_rel_embedding), so
        quantizing it breaks the encoder forward pass.
        """
        model = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model,
            q_bits=8,
            q_group_size=64,
            quantize_embeddings=True,
            quantize_heads=True,
        )
        assert not isinstance(model.encoder.encoder.rel_embeddings, nn.QuantizedEmbedding)
        for path in quantized_modules(model):
            assert not path.endswith("rel_embeddings"), f"rel_embeddings was quantized: {path}"

    def test_all_quantized_layers_are_under_encoder_by_default(self, small_deberta_config):
        model = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model,
            q_bits=8,
            q_group_size=64,
            quantize_embeddings=True,
            quantize_heads=False,
        )
        for path in quantized_modules(model):
            assert path.startswith("encoder"), f"unexpected quantized layer outside encoder: {path}"

    def test_quantize_embeddings_false_skips_embeddings(self, small_deberta_config):
        model = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model,
            q_bits=8,
            q_group_size=64,
            quantize_embeddings=False,
            quantize_heads=False,
        )
        n_linear, n_embed = count_quantized(model)
        assert n_embed == 0, "embeddings should not be quantized when disabled"
        assert n_linear > 0, "Linear layers should still be quantized"
        assert not isinstance(model.encoder.embeddings.word_embeddings, nn.QuantizedEmbedding)

    def test_quantize_heads_true_quantizes_heads(self, small_deberta_config):
        model = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model,
            q_bits=8,
            q_group_size=64,
            quantize_embeddings=True,
            quantize_heads=True,
        )
        # The classifier head must now contain quantized layers.
        classifier_quant = any(isinstance(m, nn.QuantizedLinear) for _, m in model.classifier.named_modules())
        assert classifier_quant, "classifier head should be quantized when quantize_heads=True"

        # And more layers overall than the heads-skipped case.
        heads_on = count_quantized(model)

        model2 = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model2,
            q_bits=8,
            q_group_size=64,
            quantize_embeddings=True,
            quantize_heads=False,
        )
        heads_off = count_quantized(model2)
        assert heads_on[0] > heads_off[0], "enabling heads should quantize more Linear layers"

    def test_incompatible_group_size_skips_all(self, small_deberta_config):
        """A group size that divides no layer dimension quantizes nothing."""
        # hidden_size=64 and intermediate_size=128; 96 divides neither.
        model = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model,
            q_bits=8,
            q_group_size=96,
            quantize_embeddings=True,
            quantize_heads=True,
        )
        assert count_quantized(model) == (0, 0)

    @pytest.mark.parametrize(
        ("q_bits", "q_group_size"),
        [(8, 64), (4, 64), (4, 32)],
    )
    def test_bits_and_group_size_recorded(self, small_deberta_config, q_bits, q_group_size):
        """Quantized layers carry the requested bit width and group size."""
        model = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model,
            q_bits=q_bits,
            q_group_size=q_group_size,
            quantize_embeddings=True,
            quantize_heads=False,
        )
        q = quantized_modules(model)
        assert q, "expected some layers to be quantized"
        for path, mod in q.items():
            assert mod.bits == q_bits, f"{path}: bits {mod.bits} != {q_bits}"
            assert mod.group_size == q_group_size, f"{path}: group_size {mod.group_size} != {q_group_size}"

    def test_quantized_layers_have_scales(self, small_deberta_config):
        """Quantized layers expose affine scales derived from the loaded weights."""
        model = build_extractor(small_deberta_config)
        GLiNER2MLX._quantize_model(
            model,
            q_bits=8,
            q_group_size=64,
            quantize_embeddings=True,
            quantize_heads=False,
        )
        for path, mod in quantized_modules(model).items():
            assert hasattr(mod, "scales"), f"{path}: quantized layer missing scales"
            assert mod.scales.size > 0


@pytest.mark.slow
class TestQuantizeFromPretrained:
    """End-to-end checks that require downloading the real model."""

    def test_from_pretrained_quantized_encoder(self):
        model = GLiNER2MLX.from_pretrained("fastino/gliner2-base-v1", quantize=True, q_bits=8)
        extractor = model.mlx_model

        # The big embedding table and encoder Linears should be quantized.
        assert isinstance(extractor.encoder.embeddings.word_embeddings, nn.QuantizedEmbedding)
        n_linear, n_embed = count_quantized(extractor)
        assert n_linear > 0
        assert n_embed > 0

        # Heads remain full precision by default.
        for path in quantized_modules(extractor):
            assert not path.startswith(HEAD_PREFIXES)

    def test_quantized_model_still_runs_inference(self):
        model = GLiNER2MLX.from_pretrained("fastino/gliner2-base-v1", quantize=True, q_bits=8)
        result = model.extract_entities(
            "Apple released iPhone 15 in September 2023.",
            ["company", "product", "date"],
        )
        assert "entities" in result

    def test_non_quantized_from_pretrained_has_no_quantized_layers(self):
        model = GLiNER2MLX.from_pretrained("fastino/gliner2-base-v1")
        assert count_quantized(model.mlx_model) == (0, 0)
