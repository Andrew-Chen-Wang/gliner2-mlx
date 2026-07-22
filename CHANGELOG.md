# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-07-22

### Fixed

- Disabled dropout during inference: `GLiNER2MLX.__init__` now puts the MLX `Extractor` in eval mode. Previously every dropout layer stayed active, making inference non-deterministic and lowering span scores ([#2](https://github.com/Andrew-Chen-Wang/gliner2-mlx/pull/2))
- Fixed batched relative-position bias in DeBERTa disentangled attention: `mx.repeat` interleaved head blocks, pairing rows with the wrong head's position bias for `batch_size > 1`; switched to `mx.tile` to match the batch-major layout ([#2](https://github.com/Andrew-Chen-Wang/gliner2-mlx/pull/2))
- Weight conversion now iterates `safe_open` handles via `.keys()`, fixing `TypeError` on safetensors >= 0.8 ([#2](https://github.com/Andrew-Chen-Wang/gliner2-mlx/pull/2))

### Added

- Optional affine weight quantization: `from_pretrained(..., quantize=True, q_bits=8, q_group_size=64, quantize_embeddings=True, quantize_heads=False)`. Task heads and `rel_embeddings` stay full precision by default ([#2](https://github.com/Andrew-Chen-Wang/gliner2-mlx/pull/2))
- `py.typed` marker so type checkers pick up this package's annotations (mirrors gliner2 1.3.0)

### Changed

- Require `gliner2[local]>=1.3.2`. As of gliner2 1.3.1, the base package no longer installs `torch`/`transformers` (they moved behind the `local` extra), which this package relies on transitively for preprocessing. Without the extra, a fresh install would fail at import time.
- Upgrading to gliner2 1.3.2 drops the transitive `gliner`, `onnxruntime`, `sentencepiece`, and `protobuf` dependencies (gliner2 vendored the span-rep layer) and adds a bounded LRU cache for tokenization in gliner2's processor, speeding up repeated inference.

## [0.1.1] - 2026-04-14

### Changed

- Eliminated torch round-trips in inference pipeline by extracting embeddings and span scores directly in MLX
- Ported `_extract_entities`, `_extract_relations`, `_extract_structures`, and `_find_spans` to operate on MLX arrays natively
- Removed `_mlx_to_torch` helper (no longer needed in the main inference path)

### Performance

- Structure extraction: 25.12 ms → 14.24 ms (43% faster)
- Relation extraction: 29.87 ms → 17.06 ms (43% faster)
- Single entity extraction: 18.51 ms → 14.35 ms (22% faster)
- Long text entity extraction: 30.45 ms → 19.55 ms (36% faster)
- Overall speedup vs PyTorch CPU: 2.45x → 3.07x

## [0.1.0] - 2025-04-13

### Added

- DeBERTaV2 encoder ported to MLX with disentangled attention and relative position encoding
- Span representation layer (SpanMarkerV0) ported to MLX
- Custom layers: CompileSafeGRU, CountLSTM, CountLSTMv2, CountLSTMoE, DownscaledTransformer
- MLX Extractor model assembling encoder, span rep, and task-specific heads
- GLiNER2MLX inference engine with full API: `extract_entities`, `batch_extract_entities`, `classify_text`, `extract_json`, `extract_relations`
- Automatic weight conversion from PyTorch safetensors to MLX format with caching
- Unit tests for all modules (61 tests)
- Integration tests for end-to-end extraction (requires model download)
- Statistical benchmark comparing PyTorch CPU vs MLX GPU

[0.1.2]: https://github.com/Andrew-Chen-Wang/gliner2-mlx/releases/tag/v0.1.2
[0.1.1]: https://github.com/Andrew-Chen-Wang/gliner2-mlx/releases/tag/v0.1.1
[0.1.0]: https://github.com/Andrew-Chen-Wang/gliner2-mlx/releases/tag/v0.1.0
