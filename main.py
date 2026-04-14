"""Example usage of gliner2-mlx."""

from gliner2_mlx import GLiNER2MLX


def main():
    model = GLiNER2MLX.from_pretrained("fastino/gliner2-base-v1")

    result = model.extract_entities(
        "Elon Musk founded SpaceX in 2002.",
        ["person", "company", "date"],
    )
    print(result)


if __name__ == "__main__":
    main()
