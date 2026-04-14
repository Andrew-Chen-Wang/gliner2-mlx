"""Tests that verify weight conversion and loading actually work.

These tests create mock PyTorch-shaped weight dicts and verify they
load into the MLX model without errors and produce the expected
parameter tree.
"""

import mlx.core as mx
import numpy as np

from gliner2_mlx.convert import _remap_key, _split_fused_qkv
from gliner2_mlx.deberta_v2 import DebertaV2Config
from gliner2_mlx.model import Extractor


def _make_mock_pt_weights(hidden_size=64, num_layers=2, max_width=4, vocab_size=100):
    """Build a mock PyTorch state_dict (as numpy arrays) for a small model."""
    D = hidden_size
    inter = D * 4  # intermediate_size
    att_span = 256  # position_buckets

    weights = {}

    # Encoder embeddings
    weights["encoder.embeddings.word_embeddings.weight"] = np.random.randn(vocab_size, D).astype(np.float32)
    weights["encoder.embeddings.LayerNorm.weight"] = np.ones(D, dtype=np.float32)
    weights["encoder.embeddings.LayerNorm.bias"] = np.zeros(D, dtype=np.float32)

    # Encoder rel_embeddings + LayerNorm
    weights["encoder.encoder.rel_embeddings.weight"] = np.random.randn(att_span * 2, D).astype(np.float32)
    weights["encoder.encoder.LayerNorm.weight"] = np.ones(D, dtype=np.float32)
    weights["encoder.encoder.LayerNorm.bias"] = np.zeros(D, dtype=np.float32)

    # Encoder layers
    for i in range(num_layers):
        prefix = f"encoder.encoder.layer.{i}"
        # Attention
        for proj in ("query_proj", "key_proj", "value_proj"):
            weights[f"{prefix}.attention.self.{proj}.weight"] = np.random.randn(D, D).astype(np.float32)
            weights[f"{prefix}.attention.self.{proj}.bias"] = np.zeros(D, dtype=np.float32)
        for proj in ("pos_key_proj", "pos_query_proj"):
            weights[f"{prefix}.attention.self.{proj}.weight"] = np.random.randn(D, D).astype(np.float32)
            weights[f"{prefix}.attention.self.{proj}.bias"] = np.zeros(D, dtype=np.float32)
        # Attention output
        weights[f"{prefix}.attention.output.dense.weight"] = np.random.randn(D, D).astype(np.float32)
        weights[f"{prefix}.attention.output.dense.bias"] = np.zeros(D, dtype=np.float32)
        weights[f"{prefix}.attention.output.LayerNorm.weight"] = np.ones(D, dtype=np.float32)
        weights[f"{prefix}.attention.output.LayerNorm.bias"] = np.zeros(D, dtype=np.float32)
        # Intermediate
        weights[f"{prefix}.intermediate.dense.weight"] = np.random.randn(inter, D).astype(np.float32)
        weights[f"{prefix}.intermediate.dense.bias"] = np.zeros(inter, dtype=np.float32)
        # Output
        weights[f"{prefix}.output.dense.weight"] = np.random.randn(D, inter).astype(np.float32)
        weights[f"{prefix}.output.dense.bias"] = np.zeros(D, dtype=np.float32)
        weights[f"{prefix}.output.LayerNorm.weight"] = np.ones(D, dtype=np.float32)
        weights[f"{prefix}.output.LayerNorm.bias"] = np.zeros(D, dtype=np.float32)

    # SpanRepLayer (markerV0) — nn.Sequential: Linear, ReLU, Dropout, Linear
    for proj in ("project_start", "project_end"):
        weights[f"span_rep.span_rep_layer.{proj}.0.weight"] = np.random.randn(D * 4, D).astype(np.float32)
        weights[f"span_rep.span_rep_layer.{proj}.0.bias"] = np.zeros(D * 4, dtype=np.float32)
        weights[f"span_rep.span_rep_layer.{proj}.3.weight"] = np.random.randn(D, D * 4).astype(np.float32)
        weights[f"span_rep.span_rep_layer.{proj}.3.bias"] = np.zeros(D, dtype=np.float32)
    # out_project: input is D*2
    weights["span_rep.span_rep_layer.out_project.0.weight"] = np.random.randn(D * 4, D * 2).astype(np.float32)
    weights["span_rep.span_rep_layer.out_project.0.bias"] = np.zeros(D * 4, dtype=np.float32)
    weights["span_rep.span_rep_layer.out_project.3.weight"] = np.random.randn(D, D * 4).astype(np.float32)
    weights["span_rep.span_rep_layer.out_project.3.bias"] = np.zeros(D, dtype=np.float32)

    # Classifier MLP: Linear(D, D*2), ReLU, Linear(D*2, 1)
    weights["classifier.0.weight"] = np.random.randn(D * 2, D).astype(np.float32)
    weights["classifier.0.bias"] = np.zeros(D * 2, dtype=np.float32)
    weights["classifier.2.weight"] = np.random.randn(1, D * 2).astype(np.float32)
    weights["classifier.2.bias"] = np.zeros(1, dtype=np.float32)

    # Count pred MLP: Linear(D, D*2), ReLU, Linear(D*2, 20)
    weights["count_pred.0.weight"] = np.random.randn(D * 2, D).astype(np.float32)
    weights["count_pred.0.bias"] = np.zeros(D * 2, dtype=np.float32)
    weights["count_pred.2.weight"] = np.random.randn(20, D * 2).astype(np.float32)
    weights["count_pred.2.bias"] = np.zeros(20, dtype=np.float32)

    # CountLSTM
    weights["count_embed.pos_embedding.weight"] = np.random.randn(20, D).astype(np.float32)
    weights["count_embed.gru.weight_ih_l0"] = np.random.randn(3 * D, D).astype(np.float32)
    weights["count_embed.gru.weight_hh_l0"] = np.random.randn(3 * D, D).astype(np.float32)
    weights["count_embed.gru.bias_ih_l0"] = np.zeros(3 * D, dtype=np.float32)
    weights["count_embed.gru.bias_hh_l0"] = np.zeros(3 * D, dtype=np.float32)
    # Projector MLP: Linear(D*2, D*4), ReLU, Linear(D*4, D)
    weights["count_embed.projector.0.weight"] = np.random.randn(D * 4, D * 2).astype(np.float32)
    weights["count_embed.projector.0.bias"] = np.zeros(D * 4, dtype=np.float32)
    weights["count_embed.projector.2.weight"] = np.random.randn(D, D * 4).astype(np.float32)
    weights["count_embed.projector.2.bias"] = np.zeros(D, dtype=np.float32)

    return weights


def _make_mock_pt_weights_v2(hidden_size=64, num_layers=2, max_width=4, vocab_size=100):
    """Build mock weights for count_lstm_v2 (includes DownscaledTransformer)."""
    weights = _make_mock_pt_weights(hidden_size, num_layers, max_width, vocab_size)
    D = hidden_size
    H = 128  # DownscaledTransformer hidden_size

    # Remove CountLSTM projector (v2 doesn't have it)
    for k in list(weights.keys()):
        if k.startswith("count_embed.projector."):
            del weights[k]

    # Add DownscaledTransformer weights
    # in_projector
    weights["count_embed.transformer.in_projector.weight"] = np.random.randn(H, D).astype(np.float32)
    weights["count_embed.transformer.in_projector.bias"] = np.zeros(H, dtype=np.float32)
    # TransformerEncoder layers (PyTorch uses nn.TransformerEncoder -> .layers.N)
    for i in range(2):
        prefix = f"count_embed.transformer.transformer.layers.{i}"
        # self_attn: PyTorch fused QKV
        weights[f"{prefix}.self_attn.in_proj_weight"] = np.random.randn(3 * H, H).astype(np.float32)
        weights[f"{prefix}.self_attn.in_proj_bias"] = np.zeros(3 * H, dtype=np.float32)
        weights[f"{prefix}.self_attn.out_proj.weight"] = np.random.randn(H, H).astype(np.float32)
        weights[f"{prefix}.self_attn.out_proj.bias"] = np.zeros(H, dtype=np.float32)
        # FFN
        weights[f"{prefix}.linear1.weight"] = np.random.randn(H * 2, H).astype(np.float32)
        weights[f"{prefix}.linear1.bias"] = np.zeros(H * 2, dtype=np.float32)
        weights[f"{prefix}.linear2.weight"] = np.random.randn(H, H * 2).astype(np.float32)
        weights[f"{prefix}.linear2.bias"] = np.zeros(H, dtype=np.float32)
        # LayerNorms
        weights[f"{prefix}.norm1.weight"] = np.ones(H, dtype=np.float32)
        weights[f"{prefix}.norm1.bias"] = np.zeros(H, dtype=np.float32)
        weights[f"{prefix}.norm2.weight"] = np.ones(H, dtype=np.float32)
        weights[f"{prefix}.norm2.bias"] = np.zeros(H, dtype=np.float32)
    # out_projector MLP: Linear(H+D, D), ReLU, Linear(D, D), ReLU, Linear(D, D)
    weights["count_embed.transformer.out_projector.0.weight"] = np.random.randn(D, H + D).astype(np.float32)
    weights["count_embed.transformer.out_projector.0.bias"] = np.zeros(D, dtype=np.float32)
    weights["count_embed.transformer.out_projector.2.weight"] = np.random.randn(D, D).astype(np.float32)
    weights["count_embed.transformer.out_projector.2.bias"] = np.zeros(D, dtype=np.float32)
    weights["count_embed.transformer.out_projector.4.weight"] = np.random.randn(D, D).astype(np.float32)
    weights["count_embed.transformer.out_projector.4.bias"] = np.zeros(D, dtype=np.float32)

    return weights


def _convert_mock_weights(pt_weights):
    """Apply key remapping and QKV splitting to mock weights."""
    mlx_weights = {}
    for pt_key, value in pt_weights.items():
        mlx_key = _remap_key(pt_key)
        mlx_weights[mlx_key] = value
    mlx_weights = _split_fused_qkv(mlx_weights)
    return mlx_weights


class TestWeightLoading:
    def test_all_keys_mapped(self):
        """Every PyTorch key should produce a valid MLX key."""
        pt_weights = _make_mock_pt_weights()
        mlx_weights = _convert_mock_weights(pt_weights)
        # Basic sanity: no PT-style keys should remain
        for key in mlx_weights:
            assert "encoder.encoder.layer." not in key, f"Unmapped PT key: {key}"
            assert ".attention.self." not in key, f"Unmapped PT key: {key}"

    def test_weights_load_into_model(self):
        """Converted weights should load into the MLX Extractor without errors."""
        config = DebertaV2Config(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            relative_attention=True,
            max_relative_positions=128,
            position_biased_input=False,
            pos_att_type=["c2p", "p2c"],
            position_buckets=256,
            norm_rel_ebd="layer_norm",
        )
        model = Extractor(encoder_config=config, max_width=4, counting_layer="count_lstm")

        pt_weights = _make_mock_pt_weights()
        mlx_weights = _convert_mock_weights(pt_weights)

        # Convert numpy to mx.array
        mx_weights = {k: mx.array(v) for k, v in mlx_weights.items()}
        model.load_weights(list(mx_weights.items()))
        mx.eval(model.parameters())

    def test_model_produces_output_after_loading(self):
        """Model should produce valid output after weight loading."""
        config = DebertaV2Config(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            relative_attention=True,
            max_relative_positions=128,
            position_biased_input=False,
            pos_att_type=["c2p", "p2c"],
            position_buckets=256,
            norm_rel_ebd="layer_norm",
        )
        model = Extractor(encoder_config=config, max_width=4, counting_layer="count_lstm")

        pt_weights = _make_mock_pt_weights()
        mlx_weights = _convert_mock_weights(pt_weights)
        mx_weights = {k: mx.array(v) for k, v in mlx_weights.items()}
        model.load_weights(list(mx_weights.items()))

        # Forward pass
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        attention_mask = mx.ones((1, 5))
        out = model.encode(input_ids, attention_mask)
        mx.eval(out)
        assert out.shape == (1, 5, 64)
        assert not mx.any(mx.isnan(out)).item()

    def test_qkv_split(self):
        """Fused in_proj_weight should be split into query/key/value projections."""
        weights = {
            "foo.self_attn.in_proj_weight": np.random.randn(192, 64).astype(np.float32),
            "foo.self_attn.in_proj_bias": np.random.randn(192).astype(np.float32),
            "foo.self_attn.out_proj.weight": np.random.randn(64, 64).astype(np.float32),
            "foo.self_attn.out_proj.bias": np.random.randn(64).astype(np.float32),
        }
        result = _split_fused_qkv(weights)
        assert "foo.self_attn.query_proj.weight" in result
        assert "foo.self_attn.key_proj.weight" in result
        assert "foo.self_attn.value_proj.weight" in result
        assert "foo.self_attn.query_proj.bias" in result
        assert result["foo.self_attn.query_proj.weight"].shape == (64, 64)
        # Fused key should be removed
        assert "foo.self_attn.in_proj_weight" not in result
        assert "foo.self_attn.in_proj_bias" not in result
        # out_proj should pass through unchanged
        assert "foo.self_attn.out_proj.weight" in result

    def test_count_lstm_v2_weights_load(self):
        """count_lstm_v2 weights (with DownscaledTransformer) should load."""
        config = DebertaV2Config(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            relative_attention=True,
            max_relative_positions=128,
            position_biased_input=False,
            pos_att_type=["c2p", "p2c"],
            position_buckets=256,
            norm_rel_ebd="layer_norm",
        )
        model = Extractor(encoder_config=config, max_width=4, counting_layer="count_lstm_v2")

        pt_weights = _make_mock_pt_weights_v2()
        mlx_weights = _convert_mock_weights(pt_weights)
        mx_weights = {k: mx.array(v) for k, v in mlx_weights.items()}
        model.load_weights(list(mx_weights.items()))

        # Verify forward pass works
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        attention_mask = mx.ones((1, 5))
        out = model.encode(input_ids, attention_mask)
        mx.eval(out)
        assert out.shape == (1, 5, 64)
