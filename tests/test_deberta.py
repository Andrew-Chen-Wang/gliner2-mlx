"""Tests for the DeBERTaV2 encoder implementation."""

import mlx.core as mx

from gliner2_mlx.deberta_v2 import (
    DebertaV2Config,
    DebertaV2Embeddings,
    DebertaV2Encoder,
    DebertaV2Layer,
    DebertaV2Model,
    DisentangledSelfAttention,
    _build_relative_position,
    _make_log_bucket_position,
)


class TestRelativePosition:
    def test_build_relative_position_shape(self):
        rel_pos = _build_relative_position(8, 8)
        assert rel_pos.shape == (1, 8, 8)

    def test_build_relative_position_values(self):
        rel_pos = _build_relative_position(4, 4)
        # Diagonal should be 0
        for i in range(4):
            assert rel_pos[0, i, i].item() == 0
        # Position (0, 1) should be -1
        assert rel_pos[0, 0, 1].item() == -1
        # Position (1, 0) should be 1
        assert rel_pos[0, 1, 0].item() == 1

    def test_build_relative_position_with_buckets(self):
        rel_pos = _build_relative_position(16, 16, bucket_size=8, max_position=32)
        assert rel_pos.shape == (1, 16, 16)

    def test_log_bucket_position(self):
        relative_pos = mx.arange(-10, 11)
        result = _make_log_bucket_position(relative_pos, 8, 32)
        assert result.shape == (21,)


class TestEmbeddings:
    def test_output_shape_no_position(self, small_deberta_config):
        emb = DebertaV2Embeddings(small_deberta_config)
        input_ids = mx.array([[1, 2, 3, 4]])
        out = emb(input_ids)
        mx.eval(out)
        assert out.shape == (1, 4, small_deberta_config.hidden_size)

    def test_output_shape_with_position(self, small_config_with_position_bias):
        emb = DebertaV2Embeddings(small_config_with_position_bias)
        input_ids = mx.array([[1, 2, 3, 4]])
        out = emb(input_ids)
        mx.eval(out)
        assert out.shape == (1, 4, small_config_with_position_bias.hidden_size)

    def test_masking(self, small_deberta_config):
        emb = DebertaV2Embeddings(small_deberta_config)
        input_ids = mx.array([[1, 2, 3, 0]])
        mask = mx.array([[1, 1, 1, 0]])
        out = emb(input_ids, attention_mask=mask)
        mx.eval(out)
        # Masked position should be zeroed
        assert mx.allclose(out[0, 3, :], mx.zeros(small_deberta_config.hidden_size), atol=1e-6)


class TestDisentangledSelfAttention:
    def test_output_shape(self, small_deberta_config):
        attn = DisentangledSelfAttention(small_deberta_config)
        B, L, D = 2, 8, small_deberta_config.hidden_size
        hidden = mx.random.normal((B, L, D))
        mask = mx.ones((B, 1, 1, L))  # 4D mask

        rel_pos = _build_relative_position(
            L,
            L,
            bucket_size=small_deberta_config.position_buckets,
            max_position=small_deberta_config.max_relative_positions,
        )
        rel_emb = mx.random.normal((small_deberta_config.position_buckets * 2, D))

        out = attn(hidden, mask, relative_pos=rel_pos, rel_embeddings=rel_emb)
        mx.eval(out)
        assert out.shape == (B, L, D)


class TestDebertaV2Layer:
    def test_output_shape(self, small_deberta_config):
        layer = DebertaV2Layer(small_deberta_config)
        B, L, D = 2, 8, small_deberta_config.hidden_size
        hidden = mx.random.normal((B, L, D))
        mask = mx.ones((B, 1, 1, L))

        rel_pos = _build_relative_position(L, L, bucket_size=256, max_position=128)
        rel_emb = mx.random.normal((512, D))

        out = layer(hidden, mask, relative_pos=rel_pos, rel_embeddings=rel_emb)
        mx.eval(out)
        assert out.shape == (B, L, D)


class TestDebertaV2Encoder:
    def test_output_shape(self, small_deberta_config):
        encoder = DebertaV2Encoder(small_deberta_config)
        B, L, D = 2, 8, small_deberta_config.hidden_size
        hidden = mx.random.normal((B, L, D))
        mask = mx.ones((B, L))

        out = encoder(hidden, mask)
        mx.eval(out)
        assert out.shape == (B, L, D)


class TestDebertaV2Model:
    def test_output_shape(self, small_deberta_config, sample_input):
        model = DebertaV2Model(small_deberta_config)
        input_ids, attention_mask = sample_input
        out = model(input_ids, attention_mask)
        mx.eval(out)
        assert out.shape == (input_ids.shape[0], input_ids.shape[1], small_deberta_config.hidden_size)

    def test_deterministic_eval(self, small_deberta_config):
        """Without dropout, two passes should give the same result."""
        model = DebertaV2Model(small_deberta_config)
        input_ids = mx.array([[1, 2, 3]])
        mask = mx.ones((1, 3))

        out1 = model(input_ids, mask)
        mx.eval(out1)
        out2 = model(input_ids, mask)
        mx.eval(out2)
        assert mx.allclose(out1, out2, atol=1e-5)


class TestDebertaV2Config:
    def test_from_dict(self):
        d = {
            "vocab_size": 50000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "relative_attention": True,
            "pos_att_type": ["c2p", "p2c"],
        }
        config = DebertaV2Config.from_dict(d)
        assert config.vocab_size == 50000
        assert config.hidden_size == 768
        assert config.relative_attention is True

    def test_from_dict_ignores_extra_keys(self):
        d = {"vocab_size": 100, "unknown_key": "ignored"}
        config = DebertaV2Config.from_dict(d)
        assert config.vocab_size == 100
