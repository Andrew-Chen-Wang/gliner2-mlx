"""DeBERTaV2 encoder ported to MLX.

Implements disentangled self-attention with relative position encoding,
matching HuggingFace's transformers DebertaV2Model architecture.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DebertaV2Config:
    """Configuration for the DeBERTaV2 model."""

    vocab_size: int = 128100
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    intermediate_size: int = 6144
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 0
    layer_norm_eps: float = 1e-7
    relative_attention: bool = False
    max_relative_positions: int = -1
    pad_token_id: int = 0
    position_biased_input: bool = True
    pos_att_type: list | None = None
    position_buckets: int = -1
    norm_rel_ebd: str = "none"
    conv_kernel_size: int = 0
    conv_groups: int = 1
    conv_act: str = "tanh"
    share_att_key: bool = False
    embedding_size: int | None = None
    attention_head_size: int | None = None

    @classmethod
    def from_hf_config(cls, hf_config) -> "DebertaV2Config":
        """Create from a HuggingFace DebertaV2Config."""
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=getattr(hf_config, "hidden_act", "gelu"),
            hidden_dropout_prob=hf_config.hidden_dropout_prob,
            attention_probs_dropout_prob=hf_config.attention_probs_dropout_prob,
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=getattr(hf_config, "type_vocab_size", 0),
            layer_norm_eps=hf_config.layer_norm_eps,
            relative_attention=getattr(hf_config, "relative_attention", False),
            max_relative_positions=getattr(hf_config, "max_relative_positions", -1),
            pad_token_id=getattr(hf_config, "pad_token_id", 0),
            position_biased_input=getattr(hf_config, "position_biased_input", True),
            pos_att_type=hf_config.pos_att_type,
            position_buckets=getattr(hf_config, "position_buckets", -1),
            norm_rel_ebd=getattr(hf_config, "norm_rel_ebd", "none"),
            conv_kernel_size=getattr(hf_config, "conv_kernel_size", 0),
            conv_groups=getattr(hf_config, "conv_groups", 1),
            conv_act=getattr(hf_config, "conv_act", "tanh"),
            share_att_key=getattr(hf_config, "share_att_key", False),
            embedding_size=getattr(hf_config, "embedding_size", None),
            attention_head_size=getattr(hf_config, "attention_head_size", None),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "DebertaV2Config":
        """Create from a dictionary (e.g. loaded from config.json)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_ACT_FN = {
    "gelu": nn.gelu,
    "relu": nn.relu,
    "tanh": nn.tanh,
    "silu": nn.silu,
}


def _get_act_fn(name: str):
    if name in _ACT_FN:
        return _ACT_FN[name]
    raise ValueError(f"Unknown activation: {name}")


def _make_log_bucket_position(relative_pos: mx.array, bucket_size: int, max_position: int) -> mx.array:
    sign = mx.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = mx.where(
        (relative_pos < mid) & (relative_pos > -mid),
        mx.array(mid - 1),
        mx.abs(relative_pos),
    )
    log_pos = (
        mx.ceil(
            mx.log(abs_pos.astype(mx.float32) / mid)
            / mx.log(mx.array((max_position - 1) / mid, dtype=mx.float32))
            * (mid - 1)
        )
        + mid
    )
    bucket_pos = mx.where(
        abs_pos <= mid,
        relative_pos.astype(mx.float32),
        log_pos * sign.astype(mx.float32),
    )
    return bucket_pos


def _build_relative_position(query_size: int, key_size: int, bucket_size: int = -1, max_position: int = -1) -> mx.array:
    q_ids = mx.arange(query_size)
    k_ids = mx.arange(key_size)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = _make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.astype(mx.int32)
    return rel_pos_ids[None, :, :]  # (1, Q, K)


class DebertaV2Embeddings(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        embedding_size = config.embedding_size or config.hidden_size
        self.embedding_size = embedding_size
        self.word_embeddings = nn.Embedding(config.vocab_size, embedding_size)
        self.position_biased_input = config.position_biased_input

        if config.position_biased_input:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, embedding_size)
        else:
            self.position_embeddings = None

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, embedding_size)
        else:
            self.token_type_embeddings = None

        if embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(embedding_size, config.hidden_size, bias=False)
        else:
            self.embed_proj = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        embeddings = self.word_embeddings(input_ids)

        if self.position_biased_input and self.position_embeddings is not None:
            seq_length = input_ids.shape[1]
            position_ids = mx.arange(seq_length)[None, :]
            embeddings = embeddings + self.position_embeddings(position_ids)

        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if attention_mask is not None:
            mask = attention_mask[:, :, None].astype(embeddings.dtype)
            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DisentangledSelfAttention(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = config.attention_head_size or head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.share_att_key = config.share_att_key
        self.pos_att_type = config.pos_att_type or []
        self.relative_attention = config.relative_attention

        if self.relative_attention:
            self.position_buckets = config.position_buckets
            self.max_relative_positions = config.max_relative_positions
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _transpose_for_scores(self, x: mx.array) -> mx.array:
        """(B, L, all_head_size) -> (B*H, L, head_size)"""
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_attention_heads, self.attention_head_size)
        x = x.transpose(0, 2, 1, 3)  # (B, H, L, head_size)
        return x.reshape(B * self.num_attention_heads, L, self.attention_head_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_pos: mx.array | None = None,
        rel_embeddings: mx.array | None = None,
    ) -> mx.array:
        query_layer = self._transpose_for_scores(self.query_proj(hidden_states))
        key_layer = self._transpose_for_scores(self.key_proj(hidden_states))
        value_layer = self._transpose_for_scores(self.value_proj(hidden_states))

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = mx.sqrt(mx.array(query_layer.shape[-1] * scale_factor, dtype=mx.float32))

        # Content-to-content attention
        attention_scores = (query_layer @ key_layer.transpose(0, 2, 1)) / scale

        # Disentangled relative position bias
        if self.relative_attention and rel_embeddings is not None:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self._disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )
            if rel_att is not None:
                attention_scores = attention_scores + rel_att

        # Reshape to (B, H, L, L)
        BH, Q, K = attention_scores.shape
        B = BH // self.num_attention_heads
        attention_scores = attention_scores.reshape(B, self.num_attention_heads, Q, K)

        # Apply attention mask
        attention_scores = mx.where(
            attention_mask.astype(mx.bool_),
            attention_scores,
            mx.array(mx.finfo(attention_scores.dtype).min),
        )

        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        # (B, H, Q, K) @ (B*H, K, head_size) -> need to reshape
        attention_probs_flat = attention_probs.reshape(BH, Q, K)
        context_layer = attention_probs_flat @ value_layer  # (B*H, Q, head_size)

        # Reshape back to (B, Q, all_head_size)
        context_layer = context_layer.reshape(B, self.num_attention_heads, Q, self.attention_head_size)
        context_layer = context_layer.transpose(0, 2, 1, 3)  # (B, Q, H, head_size)
        context_layer = context_layer.reshape(B, Q, -1)

        return context_layer

    def _disentangled_attention_bias(
        self,
        query_layer: mx.array,
        key_layer: mx.array,
        relative_pos: mx.array | None,
        rel_embeddings: mx.array,
        scale_factor: int,
    ) -> mx.array | None:
        if relative_pos is None:
            Q = query_layer.shape[1]
            K = key_layer.shape[1]
            relative_pos = _build_relative_position(
                Q,
                K,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )

        # Ensure relative_pos is 4D: (1, 1, Q, K)
        if relative_pos.ndim == 2:
            relative_pos = relative_pos[None, None, :, :]
        elif relative_pos.ndim == 3:
            relative_pos = relative_pos[:, None, :, :]

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.astype(mx.int32)

        # Slice rel_embeddings to [2*att_span, hidden_size] and add batch dim
        rel_embeddings = rel_embeddings[: att_span * 2, :][None, :, :]

        if self.share_att_key:
            pos_query_layer = self._transpose_for_scores(self.query_proj(rel_embeddings))
            pos_key_layer = self._transpose_for_scores(self.key_proj(rel_embeddings))
            repeat_count = query_layer.shape[0] // self.num_attention_heads
            pos_query_layer = mx.repeat(pos_query_layer, repeat_count, axis=0)
            pos_key_layer = mx.repeat(pos_key_layer, repeat_count, axis=0)
        else:
            pos_key_layer = None
            pos_query_layer = None
            if "c2p" in self.pos_att_type:
                pos_key_layer = self._transpose_for_scores(self.pos_key_proj(rel_embeddings))
                repeat_count = query_layer.shape[0] // self.num_attention_heads
                pos_key_layer = mx.repeat(pos_key_layer, repeat_count, axis=0)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self._transpose_for_scores(self.pos_query_proj(rel_embeddings))
                repeat_count = query_layer.shape[0] // self.num_attention_heads
                pos_query_layer = mx.repeat(pos_query_layer, repeat_count, axis=0)

        score = 0

        # Content-to-position
        if "c2p" in self.pos_att_type and pos_key_layer is not None:
            scale = mx.sqrt(mx.array(pos_key_layer.shape[-1] * scale_factor, dtype=mx.float32))
            c2p_att = query_layer @ pos_key_layer.transpose(0, 2, 1)  # (B*H, Q, 2*att_span)
            c2p_pos = mx.clip(relative_pos + att_span, 0, att_span * 2 - 1)  # (1, 1, Q, K)
            # Gather: for each (b*h, q, k), pick c2p_att[b*h, q, c2p_pos[0, 0, q, k]]
            c2p_pos_expanded = mx.broadcast_to(
                c2p_pos.squeeze(0),  # (1, Q, K)
                (query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]),
            )
            c2p_att = mx.take_along_axis(c2p_att, c2p_pos_expanded.astype(mx.int32), axis=-1)
            score = score + c2p_att / scale

        # Position-to-content
        if "p2c" in self.pos_att_type and pos_query_layer is not None:
            scale = mx.sqrt(mx.array(pos_query_layer.shape[-1] * scale_factor, dtype=mx.float32))
            # Build r_pos for p2c
            Q = query_layer.shape[1]
            K = key_layer.shape[1]
            if K != Q:
                r_pos = _build_relative_position(
                    K,
                    K,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
            else:
                r_pos = relative_pos
            p2c_pos = mx.clip(-r_pos + att_span, 0, att_span * 2 - 1)

            p2c_att = key_layer @ pos_query_layer.transpose(0, 2, 1)  # (B*H, K, 2*att_span)
            p2c_pos_expanded = mx.broadcast_to(
                p2c_pos.squeeze(0),  # (1, K, K)
                (query_layer.shape[0], key_layer.shape[1], key_layer.shape[1]),
            )
            p2c_att = mx.take_along_axis(p2c_att, p2c_pos_expanded.astype(mx.int32), axis=-1)
            p2c_att = p2c_att.transpose(0, 2, 1)  # (B*H, K, K) -> (B*H, K, K)
            score = score + p2c_att / scale

        return score if not isinstance(score, int) else None


class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Attention(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.self_attn = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_pos: mx.array | None = None,
        rel_embeddings: mx.array | None = None,
    ) -> mx.array:
        self_output = self.self_attn(
            hidden_states,
            attention_mask,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        return self.output(self_output, hidden_states)


class DebertaV2Intermediate(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_fn = _get_act_fn(config.hidden_act)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.act_fn(self.dense(hidden_states))


class DebertaV2Output(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_pos: mx.array | None = None,
        rel_embeddings: mx.array | None = None,
    ) -> mx.array:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class ConvLayer(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        kernel_size = config.conv_kernel_size
        self.conv_act = _get_act_fn(config.conv_act)
        padding = (kernel_size - 1) // 2
        # MLX Conv1d: (N, L, C_in) -> (N, L_out, C_out)
        # groups not directly supported in the same way; for groups=1 this is fine
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array, residual_states: mx.array, input_mask: mx.array) -> mx.array:
        # hidden_states: (B, L, C) - MLX Conv1d expects (B, L, C)
        out = self.conv(hidden_states)
        # Mask invalid positions
        rmask = (1 - input_mask).astype(mx.bool_)
        out = mx.where(rmask[:, :, None], mx.array(0.0), out)
        out = self.conv_act(self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)

        if input_mask is not None:
            mask = input_mask
            if mask.ndim != layer_norm_input.ndim:
                mask = mask[:, :, None]
            output = output * mask.astype(output.dtype)

        return output


class DebertaV2Encoder(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.layers = [DebertaV2Layer(config) for _ in range(config.num_hidden_layers)]
        self.relative_attention = config.relative_attention

        if self.relative_attention:
            self.max_relative_positions = config.max_relative_positions
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = config.position_buckets
            pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in config.norm_rel_ebd.lower().split("|")]
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.conv = ConvLayer(config) if config.conv_kernel_size > 0 else None

    def get_rel_embedding(self) -> mx.array | None:
        if not self.relative_attention:
            return None
        rel_embeddings = self.rel_embeddings.weight
        if "layer_norm" in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask: mx.array) -> mx.array:
        if attention_mask.ndim <= 2:
            extended = attention_mask[:, None, None, :]  # (B, 1, 1, L)
            attention_mask = extended * attention_mask[:, None, :, None]  # (B, 1, L, L)
        elif attention_mask.ndim == 3:
            attention_mask = attention_mask[:, None, :, :]
        return attention_mask

    def get_rel_pos(self, hidden_states: mx.array) -> mx.array | None:
        if not self.relative_attention:
            return None
        L = hidden_states.shape[1]
        return _build_relative_position(
            L,
            L,
            bucket_size=self.position_buckets,
            max_position=self.max_relative_positions,
        )

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        if attention_mask.ndim <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).astype(attention_mask.dtype)

        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states)
        rel_embeddings = self.get_rel_embedding()

        next_kv = hidden_states
        for i, layer_module in enumerate(self.layers):
            output_states = layer_module(
                next_kv,
                attention_mask,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )
            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
            next_kv = output_states

        return next_kv


class DebertaV2Model(nn.Module):
    """DeBERTaV2 encoder model in MLX.

    Returns the last hidden state of shape (batch, seq_len, hidden_size).
    """

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.config = config
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        return self.encoder(embedding_output, attention_mask)
