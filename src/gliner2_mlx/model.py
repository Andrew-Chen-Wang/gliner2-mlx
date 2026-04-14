"""MLX Extractor model matching GLiNER2's architecture.

Assembles DeBERTaV2 encoder + SpanRepLayer + classification/count heads.
Inference only (no training/loss computation).
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .deberta_v2 import DebertaV2Config, DebertaV2Model
from .layers import MLP, CountLSTM, CountLSTMoE, CountLSTMv2
from .span_rep import SpanRepLayer


class Extractor(nn.Module):
    """MLX port of the GLiNER2 Extractor model.

    Assembles the encoder, span representation, and task-specific heads.
    """

    def __init__(
        self,
        encoder_config: DebertaV2Config,
        max_width: int = 8,
        counting_layer: str = "count_lstm",
    ):
        super().__init__()
        self.max_width = max_width
        self.encoder = DebertaV2Model(encoder_config)
        hidden_size = encoder_config.hidden_size

        self.span_rep = SpanRepLayer(
            hidden_size=hidden_size,
            max_width=max_width,
            span_mode="markerV0",
            dropout=0.1,
        )

        self.classifier = MLP(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size * 2],
            output_dim=1,
            dropout=0.0,
            activation="relu",
        )

        self.count_pred = MLP(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size * 2],
            output_dim=20,
            dropout=0.0,
            activation="relu",
        )

        if counting_layer == "count_lstm":
            self.count_embed = CountLSTM(hidden_size)
        elif counting_layer == "count_lstm_moe":
            self.count_embed = CountLSTMoE(
                hidden_size=hidden_size,
                n_experts=4,
                ffn_mult=2,
                dropout=0.1,
            )
        elif counting_layer == "count_lstm_v2":
            self.count_embed = CountLSTMv2(hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown counting_layer: {counting_layer}")

    def encode(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        """Run the encoder and return hidden states.

        Args:
            input_ids: (batch, seq_len) input token IDs
            attention_mask: (batch, seq_len) attention mask

        Returns:
            (batch, seq_len, hidden_size)
        """
        return self.encoder(input_ids, attention_mask)

    def compute_span_rep(self, token_embeddings: mx.array) -> dict[str, Any]:
        """Compute span representations for a single sample.

        Args:
            token_embeddings: (text_len, hidden) token embeddings

        Returns:
            Dict with span_rep, span_mask
        """
        text_length = token_embeddings.shape[0]

        # Vectorized span index generation
        starts = mx.repeat(mx.arange(text_length)[:, None], self.max_width, axis=1)
        offsets = mx.arange(self.max_width)[None, :]
        ends = starts + offsets
        valid = ends < text_length

        starts_flat = starts.reshape(-1)
        ends_flat = ends.reshape(-1)
        invalid = ~valid.reshape(-1)
        starts_flat = mx.where(invalid, mx.array(0), starts_flat)
        ends_flat = mx.where(invalid, mx.array(0), ends_flat)
        spans_idx = mx.stack([starts_flat, ends_flat], axis=-1)[None, :]  # (1, N, 2)
        span_mask = invalid[None, :]  # (1, N)

        # Compute span representations
        span_rep = self.span_rep(
            token_embeddings[None, :, :],  # (1, text_len, hidden)
            spans_idx,
        ).squeeze(0)  # (text_len, max_width, hidden)

        return {
            "span_rep": span_rep,
            "span_mask": span_mask,
        }

    def compute_span_rep_batched(self, token_embs_list: list[mx.array]) -> list[dict[str, Any]]:
        """Batch span rep computation across multiple samples.

        Pads token embeddings to the max text length for a single batched
        forward pass through SpanMarkerV0.

        Args:
            token_embs_list: List of (text_len_i, hidden) arrays

        Returns:
            List of dicts with span_rep and span_mask per sample
        """
        if not token_embs_list:
            return []

        text_lengths = [t.shape[0] for t in token_embs_list]
        max_text_len = max(text_lengths)
        batch_size = len(token_embs_list)
        hidden = token_embs_list[0].shape[-1]

        # Pad to uniform length
        padded = mx.zeros((batch_size, max_text_len, hidden))
        for i, emb in enumerate(token_embs_list):
            padded = padded.at[i, : text_lengths[i]].add(emb)

        # Build span indices for max_text_len
        starts = mx.repeat(mx.arange(max_text_len)[:, None], self.max_width, axis=1)
        offsets = mx.arange(self.max_width)[None, :]
        ends = starts + offsets

        # Per-sample validity
        text_len_arr = mx.array(text_lengths)
        N = max_text_len * self.max_width
        ends_expanded = mx.broadcast_to(ends[None, :, :], (batch_size, max_text_len, self.max_width))
        valid = ends_expanded < text_len_arr[:, None, None]

        starts_flat = mx.broadcast_to(starts.reshape(-1)[None, :], (batch_size, N))
        ends_flat = mx.broadcast_to(ends.reshape(-1)[None, :], (batch_size, N))
        valid_flat = valid.reshape(batch_size, N)

        safe_starts = mx.where(valid_flat, starts_flat, mx.array(0))
        safe_ends = mx.where(valid_flat, ends_flat, mx.array(0))
        safe_spans = mx.stack([safe_starts, safe_ends], axis=-1)  # (batch, N, 2)
        span_mask = ~valid_flat  # (batch, N)

        # Batched forward pass
        span_rep = self.span_rep(padded, safe_spans)  # (batch, max_text_len, max_width, hidden)

        # Unpack per-sample
        results = []
        for i in range(batch_size):
            tl = text_lengths[i]
            n_spans = tl * self.max_width
            results.append(
                {
                    "span_rep": span_rep[i, :tl, :, :],
                    "span_mask": span_mask[i : i + 1, :n_spans],
                }
            )
        return results
