# gliner2-mlx

[![PyPI version](https://img.shields.io/pypi/v/gliner2-mlx)](https://pypi.org/project/gliner2-mlx/)

An [MLX](https://github.com/ml-explore/mlx) port of [GLiNER2](https://github.com/fastino-ai/GLiNER2) for fast information extraction on Apple Silicon.

GLiNER2 is a multi-task model for named entity recognition, text classification, relation extraction, and structured data extraction. This package runs the compute-heavy transformer encoder on MLX (Apple's GPU framework) while reusing GLiNER2's preprocessing and postprocessing, delivering significant speedups on Mac.

## Installation

```bash
pip install gliner2-mlx
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add gliner2-mlx
```

## Quick Start

```python
from gliner2_mlx import GLiNER2MLX

model = GLiNER2MLX.from_pretrained("fastino/gliner2-base-v1")

# Named entity recognition
result = model.extract_entities(
    "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino.",
    ["company", "person", "product", "location"],
)
print(result)
# {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
```

## Features

The API mirrors [GLiNER2](https://github.com/fastino-ai/GLiNER2) — all extraction methods work the same way:

```python
# Batch entity extraction
results = model.batch_extract_entities(
    ["Apple released iPhone 15.", "Google announced Pixel 8."],
    ["company", "product"],
    batch_size=8,
)

# Text classification
result = model.classify_text(
    "I love this product, it's amazing!",
    {"sentiment": ["positive", "negative", "neutral"]},
)

# Relation extraction
result = model.extract_relations(
    "Elon Musk founded SpaceX in 2002.",
    ["founded_by"],
)

# Structured data extraction
result = model.extract_json(
    "John Smith, aged 35, is a software engineer at Google.",
    {"person": ["name::str", "age::str", "company::str"]},
)

# Schema builder (same as GLiNER2)
from gliner2.inference.engine import Schema

schema = (
    Schema()
    .entities(["company", "person"])
    .relations(["works_for"])
)
result = model.extract("Tim Cook works at Apple.", schema)
```

## Benchmark

Measured on Apple M3 Max, `fastino/gliner2-base-v1`, 1000 iterations per scenario with 10 warmup iterations. Interleaved execution order to eliminate systematic bias. All results statistically significant (p < 0.001, paired t-test).

| Scenario | PyTorch CPU | MLX GPU | Speedup |
|---|---|---|---|
| Single entity extraction | 41.79 ms | 14.35 ms | **2.91x** |
| Batch entity extraction (8 texts) | 106.19 ms | 35.41 ms | **3.00x** |
| Long text entity extraction | 68.78 ms | 19.55 ms | **3.52x** |
| Structure extraction | 41.60 ms | 14.24 ms | **2.92x** |
| Relation extraction | 50.43 ms | 17.06 ms | **2.96x** |
| **Overall** | **61.76 ms** | **20.12 ms** | **3.07x** |

## How It Works

gliner2-mlx ports the neural network inference to MLX while keeping GLiNER2 as a dependency for everything else:

- **MLX (GPU):** DeBERTaV2 encoder, span representation, classification/count heads
- **GLiNER2 (CPU):** Tokenization, schema processing, result formatting, overlap removal

On first use, `from_pretrained` automatically converts PyTorch weights to MLX format and caches them in `~/.cache/gliner2-mlx/`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, running tests, linting, benchmarking, and publishing instructions.

## License

MIT
