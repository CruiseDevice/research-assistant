"""Command-line entrypoint.

Run via the ``research`` console script (see ``pyproject.toml``),
via ``python -m src.main``, or via the root ``main.py`` shim.
"""

from src.graph import graph


def main() -> None:
    query = input("Enter a research query or factual claim:")
    result = graph.invoke({"query": query})
    print(result.get("report", result))


if __name__ == "__main__":
    main()
