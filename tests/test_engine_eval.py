from gliner2_mlx.engine import GLiNER2MLX
from gliner2_mlx.model import Extractor


class _StubGliner2:
    """Minimal stand-in: GLiNER2MLX.__init__ only reads ``.processor``."""

    processor = None


def test_init_puts_mlx_model_in_eval_mode(small_deberta_config):
    """
    Regression test for the inference-mode dropout bug -- ``GLiNER2MLX.__init__``
    must put the MLX ``Extractor`` into eval mode so dropout is inactive during
    inference.
    """
    extractor = Extractor(
        encoder_config=small_deberta_config,
        max_width=4,
        counting_layer="count_lstm",
    )
    extractor.train()
    assert extractor.training is True

    engine = GLiNER2MLX(extractor, _StubGliner2())

    assert engine.mlx_model.training is False
    assert engine.mlx_model.encoder.training is False
