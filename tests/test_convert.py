"""Tests for weight conversion logic."""

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
