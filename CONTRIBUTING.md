# Contributing to gliner2-mlx

## Development Setup

```bash
git clone https://github.com/Andrew-Chen-Wang/gliner2-mlx.git
cd gliner2-mlx
uv sync --group dev --group bench
```

## Running Tests

```bash
uv run pytest                          # unit tests (fast, no model download)
uv run pytest -m slow                  # integration tests (downloads model)
uv run pytest --cov=gliner2_mlx        # with coverage
```

## Linting and Formatting

```bash
uv run ruff format src/ tests/         # format
uv run ruff check src/ tests/          # lint
uv run ruff check --fix src/ tests/    # lint with auto-fix
uv run ty check src/                   # type check
```

## Running Benchmarks

```bash
uv run python benchmark_statistical.py --n 1000
```

## Publishing to PyPI

1. Update the version:

   ```bash
   uv version --bump patch   # 0.1.0 -> 0.1.1
   # or
   uv version --bump minor   # 0.1.0 -> 0.2.0
   # or
   uv version 1.0.0          # set exact version
   ```

2. Make sure everything passes:

   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run ty check src/
   uv run pytest
   ```

3. Build the distributions:

   ```bash
   rm -rf dist/
   uv build
   ```

4. Verify the build looks correct:

   ```bash
   # Check wheel contents
   python -m zipfile -l dist/gliner2_mlx-*.whl

   # Check sdist contents
   tar tzf dist/gliner2_mlx-*.tar.gz
   ```

5. Publish to PyPI:

   ```bash
   uv publish --token <YOUR_PYPI_TOKEN>
   ```

   Or set the token as an environment variable:

   ```bash
   export UV_PUBLISH_TOKEN=pypi-...
   uv publish
   ```

6. Tag the release:

   ```bash
   git tag v$(uv version)
   git push origin v$(uv version)
   ```
