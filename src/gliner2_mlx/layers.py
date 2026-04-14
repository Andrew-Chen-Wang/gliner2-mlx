"""Custom neural network layers ported to MLX.

Includes CompileSafeGRU, CountLSTM variants, MLP factory, and DownscaledTransformer.
"""

import mlx.core as mx
import mlx.nn as nn


class CompileSafeGRU(nn.Module):
    """GRU cell with explicit parameter names for checkpoint compatibility.

    Uses the same parameter layout (weight_ih_l0, weight_hh_l0, bias_ih_l0,
    bias_hh_l0) as PyTorch's nn.GRU / gliner2's CompileSafeGRU so pretrained
    weights load without key remapping.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale = 1.0 / (hidden_size**0.5)
        self.weight_ih_l0 = mx.random.uniform(-scale, scale, (3 * hidden_size, input_size))
        self.weight_hh_l0 = mx.random.uniform(-scale, scale, (3 * hidden_size, hidden_size))
        self.bias_ih_l0 = mx.random.uniform(-scale, scale, (3 * hidden_size,))
        self.bias_hh_l0 = mx.random.uniform(-scale, scale, (3 * hidden_size,))

    def __call__(self, x: mx.array, h: mx.array) -> mx.array:
        """Run GRU over a sequence.

        Args:
            x: (seq_len, batch, input_size)
            h: (batch, hidden_size) initial hidden state

        Returns:
            (seq_len, batch, hidden_size) hidden states at each step
        """
        seq_len = x.shape[0]
        if seq_len == 0:
            return mx.zeros((0, h.shape[0], self.hidden_size))

        outputs = []
        for t in range(seq_len):
            gi = x[t] @ self.weight_ih_l0.T + self.bias_ih_l0
            gh = h @ self.weight_hh_l0.T + self.bias_hh_l0

            i_r, i_z, i_n = mx.split(gi, 3, axis=-1)
            h_r, h_z, h_n = mx.split(gh, 3, axis=-1)

            r = mx.sigmoid(i_r + h_r)
            z = mx.sigmoid(i_z + h_z)
            n = mx.tanh(i_n + r * h_n)

            h = (1 - z) * n + z * h
            outputs.append(h)

        return mx.stack(outputs, axis=0)


_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": lambda: nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


def _get_activation(name: str):
    if name not in _ACTIVATION_MAP:
        raise ValueError(f"Unknown activation: {name}")
    return _ACTIVATION_MAP[name]()


class MLP(nn.Module):
    """Sequential MLP with configurable layers, activations, norm, dropout."""

    def __init__(
        self,
        input_dim: int,
        intermediate_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        add_layer_norm: bool = False,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for dim in intermediate_dims:
            layers.append(nn.Linear(in_dim, dim))
            if add_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(_get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer (self-attention + FFN)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(nn.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DownscaledTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_projector = nn.Linear(input_size, hidden_size)
        self.transformer_layers = [
            TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 2, dropout) for _ in range(num_layers)
        ]
        self.out_projector = MLP(
            input_dim=hidden_size + input_size,
            intermediate_dims=[input_size, input_size],
            output_dim=input_size,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (L, M, input_size)
        Returns:
            (L, M, input_size)
        """
        original_x = x
        x = self.in_projector(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = mx.concatenate([x, original_x], axis=-1)
        x = self.out_projector(x)
        return x


class CountLSTM(nn.Module):
    def __init__(self, hidden_size: int, max_count: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(input_size=hidden_size, hidden_size=hidden_size)
        self.projector = MLP(
            input_dim=hidden_size * 2,
            intermediate_dims=[hidden_size * 4],
            output_dim=hidden_size,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

    def __call__(self, pc_emb: mx.array, gold_count_val: int) -> mx.array:
        """
        Args:
            pc_emb: (M, hidden_size) field embeddings
            gold_count_val: number of count steps
        Returns:
            (gold_count_val, M, hidden_size)
        """
        M, D = pc_emb.shape
        gold_count_val = min(gold_count_val, self.max_count)
        count_indices = mx.arange(gold_count_val)
        pos_seq = self.pos_embedding(count_indices)  # (gold_count_val, D)
        # Expand over batch: (gold_count_val, M, D)
        pos_seq = mx.broadcast_to(pos_seq[:, None, :], (gold_count_val, M, D))
        output = self.gru(pos_seq, pc_emb)  # (gold_count_val, M, D)
        pc_broadcast = mx.broadcast_to(pc_emb[None, :, :], output.shape)
        return self.projector(mx.concatenate([output, pc_broadcast], axis=-1))


class CountLSTMv2(nn.Module):
    def __init__(self, hidden_size: int, max_count: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)
        self.transformer = DownscaledTransformer(
            hidden_size,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

    def __call__(self, pc_emb: mx.array, gold_count_val: int) -> mx.array:
        M, D = pc_emb.shape
        gold_count_val = min(gold_count_val, self.max_count)
        count_idx = mx.arange(gold_count_val)
        pos_seq = self.pos_embedding(count_idx)  # (gold_count_val, D)
        pos_seq = mx.broadcast_to(pos_seq[:, None, :], (gold_count_val, M, D))
        output = self.gru(pos_seq, pc_emb)
        pc_broadcast = mx.broadcast_to(pc_emb[None, :, :], output.shape)
        return self.transformer(output + pc_broadcast)


class CountLSTMoE(nn.Module):
    """Count-aware module with Mixture-of-Experts projector."""

    def __init__(
        self,
        hidden_size: int,
        max_count: int = 20,
        n_experts: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.n_experts = n_experts

        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = CompileSafeGRU(hidden_size, hidden_size)

        inner = hidden_size * ffn_mult
        # Expert parameters (packed)
        scale = (2.0 / (hidden_size + inner)) ** 0.5
        self.w1 = mx.random.uniform(-scale, scale, (n_experts, hidden_size, inner))
        self.b1 = mx.zeros((n_experts, inner))
        scale = (2.0 / (inner + hidden_size)) ** 0.5
        self.w2 = mx.random.uniform(-scale, scale, (n_experts, inner, hidden_size))
        self.b2 = mx.zeros((n_experts, hidden_size))

        self.dropout_layer = nn.Dropout(dropout)

        # Router
        self.router_linear1 = nn.Linear(hidden_size, hidden_size)
        self.router_act = nn.GELU()
        self.router_linear2 = nn.Linear(hidden_size, n_experts)

    def __call__(self, pc_emb: mx.array, gold_count_val: int) -> mx.array:
        """
        Args:
            pc_emb: (M, D) field embeddings
            gold_count_val: count steps
        Returns:
            (L, M, D) count-aware embeddings
        """
        M, _D = pc_emb.shape
        L = min(gold_count_val, self.max_count)

        idx = mx.arange(L)
        pos_seq = self.pos_embedding(idx)[:, None, :] * mx.ones((1, M, 1))  # (L, M, D)

        h = self.gru(pos_seq, pc_emb)  # (L, M, D)

        # Routing
        gates = mx.softmax(
            self.router_linear2(self.router_act(self.router_linear1(h))),
            axis=-1,
        )  # (L, M, E)

        # Expert FFN: run all experts in parallel via einsum
        x = mx.einsum("lmd,edh->lmeh", h, self.w1) + self.b1  # (L, M, E, inner)
        x = nn.gelu(x)
        x = self.dropout_layer(x)
        x = mx.einsum("lmeh,ehd->lmed", x, self.w2) + self.b2  # (L, M, E, D)

        # Mixture weighted by gates
        out = (gates[:, :, :, None] * x).sum(axis=2)  # (L, M, D)
        return out
