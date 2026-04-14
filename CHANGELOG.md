# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/Andrew-Chen-Wang/gliner2-mlx/releases/tag/v0.1.0
