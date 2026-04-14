"""Integration tests for the GLiNER2MLX inference engine.

These tests require downloading the model from HuggingFace Hub.
Run with: pytest -m slow
"""

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    """Load the GLiNER2MLX model (cached across tests in this module)."""
    from gliner2_mlx import GLiNER2MLX

    return GLiNER2MLX.from_pretrained("fastino/gliner2-base-v1")


class TestEntityExtraction:
    def test_basic_extraction(self, model):
        result = model.extract_entities(
            "Apple released iPhone 15 in September 2023.",
            ["company", "product", "date"],
        )
        assert "entities" in result
        entities = result["entities"]
        assert isinstance(entities, dict)

    def test_with_confidence(self, model):
        result = model.extract_entities(
            "Elon Musk founded SpaceX.",
            ["person", "company"],
            include_confidence=True,
        )
        assert "entities" in result

    def test_batch_extraction(self, model):
        texts = [
            "Apple released iPhone 15.",
            "Google announced Pixel 8.",
        ]
        results = model.batch_extract_entities(texts, ["company", "product"], batch_size=2)
        assert len(results) == 2

    def test_empty_text(self, model):
        result = model.extract_entities("", ["person"])
        assert isinstance(result, dict)


class TestClassification:
    def test_basic_classification(self, model):
        result = model.classify_text(
            "I love this product, it's amazing!",
            {"sentiment": ["positive", "negative", "neutral"]},
        )
        assert "sentiment" in result


class TestRelationExtraction:
    def test_basic_relations(self, model):
        result = model.extract_relations(
            "Elon Musk founded SpaceX in 2002.",
            ["founded_by"],
        )
        assert isinstance(result, dict)


class TestStructuredExtraction:
    def test_basic_structure(self, model):
        result = model.extract_json(
            "John Smith works at Google as a software engineer.",
            {
                "employment": [
                    "person::str",
                    "company::str",
                    "role::str",
                ],
            },
        )
        assert isinstance(result, dict)
