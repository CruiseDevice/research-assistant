"""Thin shim so ``python main.py`` still works from the repo root.

The real entrypoint lives in ``src.main:main`` and is also exposed as the
``research`` console script (see ``pyproject.toml``).
"""

from src.main import main

if __name__ == "__main__":
    main()
