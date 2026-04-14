"""GLiNER2MLX inference engine.

Provides the user-facing API for entity extraction, classification,
relation extraction, and structured data extraction using MLX.
Reuses gliner2's preprocessing and postprocessing logic.
"""

import json
import logging
import os
from typing import Any

import mlx.core as mx
import numpy as np
import torch

from .convert import _load_mlx_weights, convert_weights
from .deberta_v2 import DebertaV2Config
from .model import Extractor

logger = logging.getLogger(__name__)


def _torch_to_mlx(t: torch.Tensor) -> mx.array:
    """Convert a PyTorch tensor to an MLX array via numpy."""
    return mx.array(t.detach().cpu().numpy())


def _mlx_to_torch(a: mx.array) -> torch.Tensor:
    """Convert an MLX array to a PyTorch tensor via numpy."""
    return torch.from_numpy(np.array(a))


class GLiNER2MLX:
    """GLiNER2 information extraction using MLX for inference.

    Uses gliner2's preprocessing (SchemaTransformer, tokenization) and
    postprocessing (result formatting, overlap removal), but runs the
    neural network forward pass on MLX for Apple Silicon acceleration.

    Example:
        >>> from gliner2_mlx import GLiNER2MLX
        >>> model = GLiNER2MLX.from_pretrained("fastino/gliner2-base-v1")
        >>> result = model.extract_entities(
        ...     "Apple released iPhone 15.",
        ...     ["company", "product"]
        ... )
    """

    def __init__(self, mlx_model: Extractor, gliner2_model):
        """Initialize with an MLX model and a gliner2 model for preprocessing.

        Args:
            mlx_model: The MLX Extractor model with loaded weights.
            gliner2_model: A gliner2 GLiNER2 instance (used for preprocessing
                and postprocessing only - its encoder weights are NOT used).
        """
        self.mlx_model = mlx_model
        self._gliner2 = gliner2_model
        self.processor = gliner2_model.processor

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, **kwargs) -> "GLiNER2MLX":
        """Load a pretrained GLiNER2 model with MLX inference.

        Downloads the model from HuggingFace Hub (or loads from local dir),
        converts weights to MLX format, and sets up the inference pipeline.

        Args:
            repo_or_dir: HuggingFace repo ID or local directory path.

        Returns:
            GLiNER2MLX instance ready for inference.
        """
        # Step 1: Load the gliner2 model for preprocessing
        from gliner2.inference.engine import GLiNER2

        logger.info(f"Loading gliner2 model from {repo_or_dir} for preprocessing...")
        gliner2_model = GLiNER2.from_pretrained(repo_or_dir, map_location="cpu")
        gliner2_model.eval()

        # Step 2: Convert weights to MLX format
        logger.info("Converting weights to MLX format...")
        weights_dir = convert_weights(repo_or_dir)

        # Step 3: Build MLX model from encoder config
        encoder_config_path = os.path.join(weights_dir, "encoder_config", "config.json")
        if not os.path.exists(encoder_config_path):
            # Fall back to reading from the gliner2 model's encoder config
            encoder_config = DebertaV2Config.from_hf_config(gliner2_model.encoder.config)
        else:
            with open(encoder_config_path) as f:
                encoder_config = DebertaV2Config.from_dict(json.load(f))

        extractor_config = gliner2_model.config
        mlx_model = Extractor(
            encoder_config=encoder_config,
            max_width=extractor_config.max_width,
            counting_layer=extractor_config.counting_layer,
        )

        # Step 4: Load converted weights
        logger.info("Loading MLX weights...")
        weights = _load_mlx_weights(weights_dir)
        mlx_model.load_weights(list(weights.items()))
        mx.eval(mlx_model.parameters())

        logger.info("GLiNER2MLX ready for inference.")
        return cls(mlx_model, gliner2_model)

    # =========================================================================
    # Core inference
    # =========================================================================

    def _encode_batch(self, batch) -> tuple[list[mx.array], list[list[mx.array]]]:
        """Encode a batch through the MLX encoder and extract embeddings.

        Returns token embeddings and schema embeddings as MLX arrays.
        """
        # Convert torch tensors to MLX
        input_ids = _torch_to_mlx(batch.input_ids)
        attention_mask = _torch_to_mlx(batch.attention_mask)

        # Run MLX encoder
        token_embeddings_mlx = self.mlx_model.encode(input_ids, attention_mask)
        mx.eval(token_embeddings_mlx)

        # Convert back to torch for gliner2's extract_embeddings_from_batch
        token_embeddings_torch = _mlx_to_torch(token_embeddings_mlx)

        # Use gliner2's embedding extraction (handles pooling, schema extraction)
        all_token_embs_torch, all_schema_embs_torch = self.processor.extract_embeddings_from_batch(
            token_embeddings_torch,
            batch.input_ids,
            batch,
        )

        # Convert results back to MLX
        all_token_embs = [_torch_to_mlx(t) for t in all_token_embs_torch]
        all_schema_embs = [
            [[_torch_to_mlx(e) for e in schema] for schema in sample] for sample in all_schema_embs_torch
        ]

        return all_token_embs, all_schema_embs

    def _extract_from_batch(
        self,
        batch,
        threshold: float,
        metadata_list: list[dict],
        include_confidence: bool,
        include_spans: bool,
    ) -> list[dict[str, Any]]:
        """Extract results from a preprocessed batch using MLX model."""
        all_token_embs, all_schema_embs = self._encode_batch(batch)

        # Compute span representations for samples that need them
        span_samples = []
        for i in range(len(batch)):
            has_span = any(t != "classifications" for t in batch.task_types[i])
            if has_span and all_token_embs[i].size > 0:
                span_samples.append(i)

        all_span_info = [None] * len(batch)
        if span_samples:
            span_embs = [all_token_embs[i] for i in span_samples]
            span_results = self.mlx_model.compute_span_rep_batched(span_embs)
            for idx, si in zip(span_samples, span_results, strict=True):
                all_span_info[idx] = si

        results = []
        for i in range(len(batch)):
            try:
                sample_result = self._extract_sample(
                    token_embs=all_token_embs[i],
                    schema_embs=all_schema_embs[i],
                    schema_tokens_list=batch.schema_tokens_list[i],
                    task_types=batch.task_types[i],
                    text_tokens=batch.text_tokens[i],
                    original_text=batch.original_texts[i],
                    schema=batch.original_schemas[i],
                    start_mapping=batch.start_mappings[i],
                    end_mapping=batch.end_mappings[i],
                    threshold=threshold,
                    metadata=metadata_list[i],
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                    span_info=all_span_info[i],
                )
                results.append(sample_result)
            except Exception as e:
                logger.error(f"Error extracting sample {i}: {e}")
                results.append({})

        return results

    def _extract_sample(
        self,
        token_embs: mx.array,
        schema_embs: list[list[mx.array]],
        schema_tokens_list: list[list[str]],
        task_types: list[str],
        text_tokens: list[str],
        original_text: str,
        schema: dict,
        start_mapping: list[int],
        end_mapping: list[int],
        threshold: float,
        metadata: dict,
        include_confidence: bool,
        include_spans: bool,
        span_info: dict | None = None,
    ) -> dict[str, Any]:
        """Extract from a single sample using MLX model components."""
        results = {}

        if span_info is None:
            has_span_task = any(t != "classifications" for t in task_types)
            if has_span_task and token_embs.size > 0:
                span_info = self.mlx_model.compute_span_rep(token_embs)

        cls_fields = {}
        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                for fname, fval in fields.items():
                    if isinstance(fval, dict) and "choices" in fval:
                        cls_fields[f"{parent}.{fname}"] = fval["choices"]

        text_len = len(start_mapping)

        for i, (schema_tokens, task_type) in enumerate(zip(schema_tokens_list, task_types, strict=True)):
            if len(schema_tokens) < 4 or not schema_embs[i]:
                continue

            schema_name = schema_tokens[2].split(" [DESCRIPTION] ")[0]
            embs = mx.stack(schema_embs[i])

            if task_type == "classifications":
                self._extract_classification_result(results, schema_name, schema, embs, schema_tokens)
            else:
                self._extract_span_result(
                    results,
                    schema_name,
                    task_type,
                    embs,
                    span_info,
                    schema_tokens,
                    text_tokens,
                    text_len,
                    original_text,
                    start_mapping,
                    end_mapping,
                    threshold,
                    metadata,
                    cls_fields,
                    include_confidence,
                    include_spans,
                )

        return results

    def _extract_classification_result(
        self,
        results: dict,
        schema_name: str,
        schema: dict,
        embs: mx.array,
        schema_tokens: list[str],
    ):
        """Extract classification result using MLX classifier."""
        cls_config = next(c for c in schema["classifications"] if schema_tokens[2].startswith(c["task"]))

        cls_embeds = embs[1:]
        logits = self.mlx_model.classifier(cls_embeds).squeeze(-1)

        is_multi = cls_config.get("multi_label", False)
        activation = cls_config.get("class_act", "auto")

        if activation == "sigmoid" or (activation == "auto" and is_multi):
            probs = mx.sigmoid(logits)
        elif activation == "softmax" or (activation == "auto" and not is_multi):
            probs = mx.softmax(logits, axis=-1)
        else:
            probs = mx.sigmoid(logits)

        labels = cls_config["labels"]
        cls_threshold = cls_config.get("cls_threshold", 0.5)
        probs_list = probs.tolist()

        if is_multi:
            chosen = [(labels[j], probs_list[j]) for j in range(len(labels)) if probs_list[j] >= cls_threshold]
            if not chosen:
                best = int(mx.argmax(probs).item())
                chosen = [(labels[best], probs_list[best])]
            results[schema_name] = chosen
        else:
            best = int(mx.argmax(probs).item())
            results[schema_name] = (labels[best], probs_list[best])

    def _extract_span_result(
        self,
        results: dict,
        schema_name: str,
        task_type: str,
        embs: mx.array,
        span_info: dict | None,
        schema_tokens: list[str],
        text_tokens: list[str],
        text_len: int,
        original_text: str,
        start_mapping: list[int],
        end_mapping: list[int],
        threshold: float,
        metadata: dict,
        cls_fields: dict,
        include_confidence: bool,
        include_spans: bool,
    ):
        """Extract span-based results using MLX model."""
        field_names = []
        for j in range(len(schema_tokens) - 1):
            if schema_tokens[j] in ("[E]", "[C]", "[R]"):
                field_names.append(schema_tokens[j + 1])

        if not field_names:
            results[schema_name] = [] if schema_name == "entities" else {}
            return

        # Predict count
        count_logits = self.mlx_model.count_pred(embs[0:1])
        pred_count = int(mx.argmax(count_logits, axis=1).item())

        if pred_count <= 0 or span_info is None:
            if schema_name == "entities" or task_type == "relations":
                results[schema_name] = []
            else:
                results[schema_name] = {}
            return

        # Get span scores
        struct_proj = self.mlx_model.count_embed(embs[1:], pred_count)
        # span_info["span_rep"]: (text_len, max_width, D)
        # struct_proj: (L, K, D)  where L=pred_count, K=num_fields
        span_scores = mx.sigmoid(mx.einsum("lkd,bpd->bplk", span_info["span_rep"], struct_proj))

        # Convert to torch for gliner2's extraction logic
        span_scores_torch = _mlx_to_torch(span_scores)

        # Delegate to gliner2's extraction methods
        if schema_name == "entities":
            results[schema_name] = self._gliner2._extract_entities(
                field_names,
                span_scores_torch,
                text_len,
                text_tokens,
                original_text,
                start_mapping,
                end_mapping,
                threshold,
                metadata,
                include_confidence,
                include_spans,
            )
        elif task_type == "relations":
            results[schema_name] = self._gliner2._extract_relations(
                schema_name,
                field_names,
                span_scores_torch,
                pred_count,
                text_len,
                text_tokens,
                original_text,
                start_mapping,
                end_mapping,
                threshold,
                metadata,
                include_confidence,
                include_spans,
            )
        else:
            results[schema_name] = self._gliner2._extract_structures(
                schema_name,
                field_names,
                span_scores_torch,
                pred_count,
                text_len,
                text_tokens,
                original_text,
                start_mapping,
                end_mapping,
                threshold,
                metadata,
                cls_fields,
                include_confidence,
                include_spans,
            )

    # =========================================================================
    # Public API (mirrors GLiNER2)
    # =========================================================================

    def batch_extract(
        self,
        texts: list[str],
        schemas,
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract from multiple texts.

        Args:
            texts: List of input texts.
            schemas: Single schema or list of schemas (Schema objects or dicts).
            batch_size: Batch size for processing.
            threshold: Confidence threshold.
            format_results: Whether to format output.
            include_confidence: Include confidence scores.
            include_spans: Include character-level positions.
            max_len: Maximum word tokens per text.

        Returns:
            List of extraction results.
        """
        if not texts:
            return []

        from gliner2.training.trainer import ExtractorCollator

        self._gliner2.eval()
        self.processor.change_mode(is_training=False)

        # Normalize schemas
        if isinstance(schemas, list):
            if len(schemas) != len(texts):
                raise ValueError(f"Schema count ({len(schemas)}) != text count ({len(texts)})")
            schema_list = schemas
        else:
            schema_list = [schemas] * len(texts)

        schema_dicts = []
        metadata_list = []

        for schema in schema_list:
            if hasattr(schema, "build"):
                schema_dict = schema.build()
                classification_tasks = [c["task"] for c in schema_dict.get("classifications", [])]
                meta = {
                    "field_metadata": schema._field_metadata,
                    "entity_metadata": schema._entity_metadata,
                    "relation_metadata": getattr(schema, "_relation_metadata", {}),
                    "field_orders": schema._field_orders,
                    "entity_order": schema._entity_order,
                    "relation_order": getattr(schema, "_relation_order", []),
                    "classification_tasks": classification_tasks,
                }
            else:
                schema_dict = schema
                entities = schema_dict.get("entities")
                if isinstance(entities, list):
                    schema_dict = {**schema_dict, "entities": {e: "" for e in entities}}
                classification_tasks = [c["task"] for c in schema_dict.get("classifications", [])]
                entity_order = (
                    list(schema_dict["entities"].keys()) if isinstance(schema_dict.get("entities"), dict) else []
                )
                meta = {
                    "field_metadata": {},
                    "entity_metadata": {},
                    "relation_metadata": {},
                    "field_orders": {},
                    "entity_order": entity_order,
                    "relation_order": [],
                    "classification_tasks": classification_tasks,
                }

            for cls_config in schema_dict.get("classifications", []):
                cls_config.setdefault("true_label", ["N/A"])

            schema_dicts.append(schema_dict)
            metadata_list.append(meta)

        dataset = list(zip(texts, schema_dicts, strict=True))
        collator = ExtractorCollator(self.processor, is_training=False, max_len=max_len)

        # Process in batches
        all_results = []
        sample_idx = 0

        for batch_start in range(0, len(dataset), batch_size):
            batch_data = dataset[batch_start : batch_start + batch_size]
            batch = collator(batch_data)

            batch_results = self._extract_from_batch(
                batch,
                threshold,
                metadata_list[sample_idx : sample_idx + len(batch)],
                include_confidence,
                include_spans,
            )

            if format_results:
                for i, result in enumerate(batch_results):
                    meta = metadata_list[sample_idx + i]
                    requested_relations = meta.get("relation_order", [])
                    classification_tasks = meta.get("classification_tasks", [])
                    batch_results[i] = self._gliner2.format_results(
                        result,
                        include_confidence,
                        requested_relations,
                        classification_tasks,
                    )

            all_results.extend(batch_results)
            sample_idx += len(batch)

        return all_results

    def extract(
        self,
        text: str,
        schema,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict:
        """Extract from a single text."""
        return self.batch_extract(
            [text],
            schema,
            1,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len=max_len,
        )[0]

    def extract_entities(
        self,
        text: str,
        entity_types,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict:
        """Extract entities from text."""
        from gliner2.inference.engine import Schema

        schema = Schema().entities(entity_types)
        return self.extract(
            text,
            schema,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len=max_len,
        )

    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types,
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict]:
        """Batch extract entities."""
        from gliner2.inference.engine import Schema

        schema = Schema().entities(entity_types)
        return self.batch_extract(
            texts,
            schema,
            batch_size,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len=max_len,
        )

    def classify_text(
        self,
        text: str,
        tasks: dict,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        max_len: int | None = None,
    ) -> dict:
        """Classify text."""
        from gliner2.inference.engine import Schema

        schema = Schema()
        for name, config in tasks.items():
            if isinstance(config, dict) and "labels" in config:
                cfg = config.copy()
                labels = cfg.pop("labels")
                schema.classification(name, labels, **cfg)
            else:
                schema.classification(name, config)
        return self.extract(
            text,
            schema,
            threshold,
            format_results,
            include_confidence,
            max_len=max_len,
        )

    def extract_json(
        self,
        text: str,
        structures: dict,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict:
        """Extract structured data."""
        from gliner2.inference.engine import Schema

        schema = Schema()
        for parent, fields in structures.items():
            builder = schema.structure(parent)
            for spec in fields:
                name, dtype, choices, desc = self._gliner2._parse_field_spec(spec)
                builder.field(name, dtype=dtype, choices=choices, description=desc)
        return self.extract(
            text,
            schema,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len=max_len,
        )

    def extract_relations(
        self,
        text: str,
        relation_types,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict:
        """Extract relations."""
        from gliner2.inference.engine import Schema

        schema = Schema().relations(relation_types)
        return self.extract(
            text,
            schema,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len=max_len,
        )
