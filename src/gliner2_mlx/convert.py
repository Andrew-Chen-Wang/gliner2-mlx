"""Weight conversion from PyTorch GLiNER2 safetensors to MLX format.

Handles key remapping between HuggingFace/gliner2 parameter names
and the MLX module hierarchy.
"""

import logging
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def _build_key_map() -> dict[str, str]:
    """Build the PyTorch -> MLX key mapping rules.

    Returns a dict of prefix replacements applied in order.
    """
    return {
        # DeBERTa encoder
        "encoder.embeddings.word_embeddings.": "encoder.embeddings.word_embeddings.",
        "encoder.embeddings.position_embeddings.": "encoder.embeddings.position_embeddings.",
        "encoder.embeddings.token_type_embeddings.": "encoder.embeddings.token_type_embeddings.",
        "encoder.embeddings.embed_proj.": "encoder.embeddings.embed_proj.",
        "encoder.embeddings.LayerNorm.": "encoder.embeddings.LayerNorm.",
        # Encoder layers: encoder.encoder.layer.N -> encoder.encoder.layers.N
        "encoder.encoder.layer.": "encoder.encoder.layers.",
        # Attention submodules: .attention.self. -> .attention.self_attn.
        ".attention.self.": ".attention.self_attn.",
        # Intermediate/output remain the same name-wise
        # Encoder-level rel_embeddings and LayerNorm
        "encoder.encoder.rel_embeddings.": "encoder.encoder.rel_embeddings.",
        "encoder.encoder.LayerNorm.": "encoder.encoder.LayerNorm.",
        # Conv layer
        "encoder.encoder.conv.": "encoder.encoder.conv.",
        # Span representation: span_rep.span_rep_layer. -> span_rep.span_rep_layer.
        "span_rep.span_rep_layer.": "span_rep.span_rep_layer.",
        # Classifier, count_pred, count_embed stay the same
    }


def _remap_key(key: str) -> str:
    """Remap a single PyTorch state_dict key to the MLX module path."""
    result = key

    # encoder.encoder.layer.N -> encoder.encoder.layers.N
    result = result.replace("encoder.encoder.layer.", "encoder.encoder.layers.")

    # attention.self. -> attention.self_attn.
    result = result.replace(".attention.self.", ".attention.self_attn.")

    # Span rep projection layers: project_start.0.weight -> project_start.layers.0.weight
    # The gliner SpanMarkerV0 uses nn.Sequential internally, so keys are like:
    # span_rep.span_rep_layer.project_start.0.weight
    # Our MLX ProjectionLayer stores them as layers.0.weight
    for proj_name in ("project_start", "project_end", "out_project"):
        old_prefix = f"span_rep.span_rep_layer.{proj_name}."
        if old_prefix in result:
            idx = result.index(old_prefix)
            suffix = result[idx + len(old_prefix) :]
            result = f"span_rep.span_rep_layer.{proj_name}.layers.{suffix}"
            break

    # MLP layers in classifier, count_pred, count_embed.projector:
    # These use nn.Sequential in PyTorch: classifier.0.weight -> classifier.layers.0.weight
    for mlp_prefix in ("classifier.", "count_pred."):
        if result.startswith(mlp_prefix):
            suffix = result[len(mlp_prefix) :]
            result = f"{mlp_prefix}layers.{suffix}"
            break

    # count_embed.projector.N.weight -> count_embed.projector.layers.N.weight
    if "count_embed.projector." in result and "count_embed.projector.layers." not in result:
        result = result.replace("count_embed.projector.", "count_embed.projector.layers.")

    # count_embed.gru weights stay the same (weight_ih_l0, etc.)

    # CountLSTMoE router: router.0. -> router_linear1., router.2. -> router_linear2.
    if "count_embed.router." in result:
        result = result.replace("count_embed.router.0.", "count_embed.router_linear1.")
        result = result.replace("count_embed.router.2.", "count_embed.router_linear2.")

    # DownscaledTransformer: out_projector -> out_projector.layers
    if "count_embed.transformer.out_projector." in result and ".layers." not in result:
        result = result.replace(
            "count_embed.transformer.out_projector.",
            "count_embed.transformer.out_projector.layers.",
        )

    # DownscaledTransformer: transformer.layers.N -> transformer.transformer_layers.N
    if "count_embed.transformer.transformer.layers." in result:
        result = result.replace(
            "count_embed.transformer.transformer.layers.",
            "count_embed.transformer.transformer_layers.",
        )
    # Also handle TransformerEncoder wrapper: transformer.transformer. -> transformer.transformer_layers.
    # PyTorch TransformerEncoder stores layers as .layers.N
    # But we might also see transformer.transformer.0. etc depending on version

    return result


def _remap_value(key: str, value: np.ndarray, mlx_key: str) -> np.ndarray:
    """Remap weight values (e.g. transpose Conv1d weights)."""
    # PyTorch Conv1d weight: (out_channels, in_channels, kernel_size)
    # MLX Conv1d weight: (out_channels, kernel_size, in_channels)
    if "conv." in mlx_key and "weight" in mlx_key and value.ndim == 3:
        return np.transpose(value, (0, 2, 1))
    return value


def _split_fused_qkv(weights: dict) -> dict:
    """Split PyTorch's fused in_proj_weight/bias into separate Q/K/V projections.

    PyTorch MultiHeadAttention stores:
      self_attn.in_proj_weight  (3*d, d)
      self_attn.in_proj_bias    (3*d,)
      self_attn.out_proj.weight (d, d)
      self_attn.out_proj.bias   (d,)

    MLX MultiHeadAttention expects:
      self_attn.query_proj.weight (d, d)
      self_attn.key_proj.weight   (d, d)
      self_attn.value_proj.weight (d, d)
      self_attn.out_proj.weight   (d, d)
      self_attn.out_proj.bias     (d,)
    """
    keys_to_remove = []
    keys_to_add = {}

    for key, value in weights.items():
        if "self_attn.in_proj_weight" in key:
            prefix = key.replace("self_attn.in_proj_weight", "self_attn")
            d = value.shape[0] // 3
            q, k, v = value[:d], value[d : 2 * d], value[2 * d :]
            keys_to_add[f"{prefix}.query_proj.weight"] = q
            keys_to_add[f"{prefix}.key_proj.weight"] = k
            keys_to_add[f"{prefix}.value_proj.weight"] = v
            keys_to_remove.append(key)
        elif "self_attn.in_proj_bias" in key:
            prefix = key.replace("self_attn.in_proj_bias", "self_attn")
            d = value.shape[0] // 3
            q, k, v = value[:d], value[d : 2 * d], value[2 * d :]
            keys_to_add[f"{prefix}.query_proj.bias"] = q
            keys_to_add[f"{prefix}.key_proj.bias"] = k
            keys_to_add[f"{prefix}.value_proj.bias"] = v
            keys_to_remove.append(key)

    for k in keys_to_remove:
        del weights[k]
    weights.update(keys_to_add)
    return weights


def convert_weights(
    model_path: str,
    output_path: str | None = None,
) -> str:
    """Convert PyTorch GLiNER2 weights to MLX format.

    Args:
        model_path: HuggingFace repo ID or local directory with model files.
        output_path: Where to save converted weights. If None, uses
            ~/.cache/gliner2-mlx/<model_name>/

    Returns:
        Path to the directory containing converted weights.
    """
    from huggingface_hub import hf_hub_download

    def download_or_local(repo, filename):
        if os.path.isdir(repo):
            path = os.path.join(repo, filename)
            if os.path.exists(path):
                return path
            return None
        try:
            return hf_hub_download(repo, filename)
        except Exception:
            return None

    # Determine output path
    if output_path is None:
        cache_dir = Path.home() / ".cache" / "gliner2-mlx"
        model_name = model_path.replace("/", "--")
        output_path = str(cache_dir / model_name)

    os.makedirs(output_path, exist_ok=True)

    # Check if already converted
    converted_weights_path = os.path.join(output_path, "weights.safetensors")
    if os.path.exists(converted_weights_path):
        logger.info(f"Using cached converted weights at {output_path}")
        return output_path

    # Load PyTorch weights via safetensors (numpy, no torch needed)
    from safetensors import safe_open

    model_file = download_or_local(model_path, "model.safetensors")
    if model_file is None:
        raise FileNotFoundError(f"Could not find model.safetensors in {model_path}")

    logger.info(f"Loading weights from {model_file}")
    pt_weights = {}
    with safe_open(model_file, framework="numpy") as f:
        for key in f:
            pt_weights[key] = f.get_tensor(key)

    # Remap keys and values
    mlx_weights = {}
    for pt_key, value in pt_weights.items():
        mlx_key = _remap_key(pt_key)
        value = _remap_value(pt_key, value, mlx_key)
        mlx_weights[mlx_key] = value
        if pt_key != mlx_key:
            logger.debug(f"  {pt_key} -> {mlx_key}")

    # Split fused QKV projections (PyTorch MultiHeadAttention -> MLX separate projections)
    mlx_weights = _split_fused_qkv(mlx_weights)

    # Save as MLX safetensors
    from safetensors.numpy import save_file

    save_file(mlx_weights, converted_weights_path)
    logger.info(f"Saved converted weights to {converted_weights_path}")

    # Copy config files
    for config_file in ("config.json", "encoder_config/config.json"):
        src = download_or_local(model_path, config_file)
        if src is not None:
            dst_dir = os.path.join(output_path, os.path.dirname(config_file))
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(output_path, config_file)
            if not os.path.exists(dst):
                import shutil

                shutil.copy2(src, dst)

    return output_path


def _load_mlx_weights(weights_dir: str) -> dict:
    """Load converted MLX weights from a directory."""
    weights_path = os.path.join(weights_dir, "weights.safetensors")
    weights = mx.load(weights_path)
    return weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert GLiNER2 weights to MLX format")
    parser.add_argument("--model", required=True, help="HuggingFace repo ID or local path")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    output = convert_weights(args.model, args.output)
    print(f"Converted weights saved to: {output}")
