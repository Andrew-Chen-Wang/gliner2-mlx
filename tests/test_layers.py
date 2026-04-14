"""Tests for custom layers (GRU, CountLSTM variants, MLP)."""

import mlx.core as mx

from gliner2_mlx.layers import (
    MLP,
    CompileSafeGRU,
    CountLSTM,
    CountLSTMoE,
    CountLSTMv2,
    DownscaledTransformer,
)


class TestCompileSafeGRU:
    def test_output_shape(self):
        gru = CompileSafeGRU(input_size=32, hidden_size=64)
        x = mx.random.normal((5, 3, 32))  # (seq_len, batch, input)
        h = mx.random.normal((3, 64))  # (batch, hidden)
        out = gru(x, h)
        mx.eval(out)
        assert out.shape == (5, 3, 64)

    def test_empty_sequence(self):
        gru = CompileSafeGRU(input_size=32, hidden_size=64)
        x = mx.zeros((0, 3, 32))
        h = mx.random.normal((3, 64))
        out = gru(x, h)
        mx.eval(out)
        assert out.shape == (0, 3, 64)

    def test_single_step(self):
        gru = CompileSafeGRU(input_size=16, hidden_size=16)
        x = mx.random.normal((1, 2, 16))
        h = mx.zeros((2, 16))
        out = gru(x, h)
        mx.eval(out)
        assert out.shape == (1, 2, 16)


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(
            input_dim=64,
            intermediate_dims=[128, 64],
            output_dim=1,
            dropout=0.0,
        )
        x = mx.random.normal((2, 64))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (2, 1)

    def test_single_layer(self):
        mlp = MLP(input_dim=32, intermediate_dims=[], output_dim=16, dropout=0.0)
        x = mx.random.normal((4, 32))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (4, 16)

    def test_with_layer_norm(self):
        mlp = MLP(
            input_dim=64,
            intermediate_dims=[128],
            output_dim=32,
            dropout=0.0,
            add_layer_norm=True,
        )
        x = mx.random.normal((2, 64))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (2, 32)


class TestDownscaledTransformer:
    def test_output_shape(self):
        dt = DownscaledTransformer(input_size=64, hidden_size=32, num_heads=2, num_layers=1, dropout=0.0)
        x = mx.random.normal((5, 3, 64))  # (L, M, input_size)
        out = dt(x)
        mx.eval(out)
        assert out.shape == (5, 3, 64)


class TestCountLSTM:
    def test_output_shape(self):
        clstm = CountLSTM(hidden_size=64, max_count=20)
        pc_emb = mx.random.normal((4, 64))  # (M, D)
        out = clstm(pc_emb, gold_count_val=3)
        mx.eval(out)
        assert out.shape == (3, 4, 64)

    def test_max_count_clamping(self):
        clstm = CountLSTM(hidden_size=32, max_count=5)
        pc_emb = mx.random.normal((2, 32))
        out = clstm(pc_emb, gold_count_val=100)
        mx.eval(out)
        assert out.shape == (5, 2, 32)  # Clamped to max_count

    def test_single_count(self):
        clstm = CountLSTM(hidden_size=32)
        pc_emb = mx.random.normal((3, 32))
        out = clstm(pc_emb, gold_count_val=1)
        mx.eval(out)
        assert out.shape == (1, 3, 32)


class TestCountLSTMv2:
    def test_output_shape(self):
        clstm = CountLSTMv2(hidden_size=64, max_count=20)
        pc_emb = mx.random.normal((4, 64))
        out = clstm(pc_emb, gold_count_val=3)
        mx.eval(out)
        assert out.shape == (3, 4, 64)


class TestCountLSTMoE:
    def test_output_shape(self):
        clstm = CountLSTMoE(hidden_size=64, max_count=20, n_experts=4, ffn_mult=2, dropout=0.0)
        pc_emb = mx.random.normal((4, 64))
        out = clstm(pc_emb, gold_count_val=3)
        mx.eval(out)
        assert out.shape == (3, 4, 64)

    def test_expert_gating(self):
        """Verify that the router produces valid probability distributions."""
        clstm = CountLSTMoE(hidden_size=32, n_experts=3, dropout=0.0)
        # Access router output
        h = mx.random.normal((3, 2, 32))
        gates = mx.softmax(
            clstm.router_linear2(clstm.router_act(clstm.router_linear1(h))),
            axis=-1,
        )
        mx.eval(gates)
        # Gates should sum to 1 along expert dimension
        sums = gates.sum(axis=-1)
        assert mx.allclose(sums, mx.ones_like(sums), atol=1e-5)
