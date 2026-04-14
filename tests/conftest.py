"""Shared fixtures for gliner2-mlx tests."""

import mlx.core as mx
import pytest

from gliner2_mlx.deberta_v2 import DebertaV2Config


@pytest.fixture
def small_deberta_config():
    """A tiny DeBERTaV2 config for fast unit tests."""
    return DebertaV2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=128,
        type_vocab_size=0,
        layer_norm_eps=1e-7,
        relative_attention=True,
        max_relative_positions=128,
        position_biased_input=False,
        pos_att_type=["c2p", "p2c"],
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        conv_kernel_size=0,
    )


@pytest.fixture
def small_config_with_position_bias():
    """Config with position_biased_input=True (absolute positions)."""
    return DebertaV2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=128,
        type_vocab_size=0,
        layer_norm_eps=1e-7,
        relative_attention=True,
        max_relative_positions=128,
        position_biased_input=True,
        pos_att_type=["c2p", "p2c"],
        position_buckets=256,
        norm_rel_ebd="layer_norm",
    )


@pytest.fixture
def sample_input():
    """Sample input_ids and attention_mask for testing."""
    batch_size = 2
    seq_len = 16
    input_ids = mx.random.randint(0, 100, (batch_size, seq_len))
    attention_mask = mx.ones((batch_size, seq_len))
    return input_ids, attention_mask


@pytest.fixture
def sample_texts():
    """Sample texts for end-to-end testing."""
    return [
        "Apple released iPhone 15 in September 2023.",
        "Elon Musk founded SpaceX in 2002.",
    ]


@pytest.fixture
def sample_entity_types():
    """Sample entity types for testing."""
    return ["company", "product", "person", "date"]
