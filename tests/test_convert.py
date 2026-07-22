"""Tests for weight conversion logic."""

import numpy as np
from safetensors import safe_open as _real_safe_open
from safetensors.numpy import save_file

import gliner2_mlx.convert as convert
from gliner2_mlx.convert import _remap_key


class TestKeyRemapping:
    def test_encoder_layer_renaming(self):
        key = "encoder.encoder.layer.0.attention.self.query_proj.weight"
        expected = "encoder.encoder.layers.0.attention.self_attn.query_proj.weight"
        assert _remap_key(key) == expected

    def test_encoder_layer_number(self):
        key = "encoder.encoder.layer.11.intermediate.dense.weight"
        expected = "encoder.encoder.layers.11.intermediate.dense.weight"
        assert _remap_key(key) == expected

    def test_span_rep_projection(self):
        key = "span_rep.span_rep_layer.project_start.0.weight"
        expected = "span_rep.span_rep_layer.project_start.layers.0.weight"
        assert _remap_key(key) == expected

    def test_span_rep_out_project(self):
        key = "span_rep.span_rep_layer.out_project.3.bias"
        expected = "span_rep.span_rep_layer.out_project.layers.3.bias"
        assert _remap_key(key) == expected

    def test_classifier_mlp(self):
        key = "classifier.0.weight"
        expected = "classifier.layers.0.weight"
        assert _remap_key(key) == expected

    def test_count_pred_mlp(self):
        key = "count_pred.2.weight"
        expected = "count_pred.layers.2.weight"
        assert _remap_key(key) == expected

    def test_count_embed_projector(self):
        key = "count_embed.projector.0.weight"
        expected = "count_embed.projector.layers.0.weight"
        assert _remap_key(key) == expected

    def test_count_embed_gru_weights(self):
        """GRU parameter names should pass through unchanged."""
        key = "count_embed.gru.weight_ih_l0"
        assert _remap_key(key) == key

    def test_encoder_embeddings(self):
        key = "encoder.embeddings.word_embeddings.weight"
        assert _remap_key(key) == key

    def test_encoder_rel_embeddings(self):
        key = "encoder.encoder.rel_embeddings.weight"
        assert _remap_key(key) == key

    def test_encoder_layernorm(self):
        key = "encoder.encoder.LayerNorm.weight"
        assert _remap_key(key) == key

    def test_attention_output(self):
        key = "encoder.encoder.layer.0.attention.output.dense.weight"
        expected = "encoder.encoder.layers.0.attention.output.dense.weight"
        assert _remap_key(key) == expected

    def test_count_embed_pos_embedding(self):
        key = "count_embed.pos_embedding.weight"
        assert _remap_key(key) == key

    def test_moe_router(self):
        key = "count_embed.router.0.weight"
        expected = "count_embed.router_linear1.weight"
        assert _remap_key(key) == expected

    def test_moe_router_linear2(self):
        key = "count_embed.router.2.weight"
        expected = "count_embed.router_linear2.weight"
        assert _remap_key(key) == expected


class _NonIterableSafeOpen:
    """Mimics a ``safetensors>=0.8`` handle: exposes ``keys()``/``get_tensor()``
    but is *not* directly iterable. Delegates to the real handle otherwise."""

    def __init__(self, *args, **kwargs):
        self._handle = _real_safe_open(*args, **kwargs)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)

    def __iter__(self):
        raise TypeError("'builtins.safe_open' object is not iterable")

    def keys(self):
        return self._handle.keys()

    def get_tensor(self, name):
        return self._handle.get_tensor(name)


class TestConvertWeightsSafeOpen:
    """Regression test for the safetensors-version conversion bug."""

    def test_conversion_handles_non_iterable_safe_open(self, tmp_path, monkeypatch):
        # A local model directory so convert_weights resolves files locally
        # (no HuggingFace download).
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        tensors = {
            "encoder.embeddings.word_embeddings.weight": np.random.rand(8, 4).astype(np.float32),
            "encoder.encoder.layer.0.intermediate.dense.weight": np.random.rand(4, 4).astype(np.float32),
            "encoder.encoder.layer.0.attention.output.dense.bias": np.random.rand(4).astype(np.float32),
        }
        save_file(tensors, str(model_dir / "model.safetensors"))

        monkeypatch.setattr("safetensors.safe_open", _NonIterableSafeOpen)

        out_dir = tmp_path / "converted"
        result = convert.convert_weights(str(model_dir), output_path=str(out_dir))

        converted = convert._load_mlx_weights(result)
        assert len(converted) == len(tensors)
        # Keys are enumerated (would be empty / error under the buggy iteration)
        # and remapped (layer -> layers).
        assert "encoder.embeddings.word_embeddings.weight" in converted
        assert "encoder.encoder.layers.0.intermediate.dense.weight" in converted
        assert "encoder.encoder.layers.0.attention.output.dense.bias" in converted
