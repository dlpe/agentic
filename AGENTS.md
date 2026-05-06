# AGENTS.md

## Cursor Cloud specific instructions

**pygentix** is a pure Python library (no web app, no Docker, no external services needed for dev/test).

### Quick reference

- **Install**: `pip install -e ".[dev]"` (installs all backends + pytest + pymupdf)
- **Test**: `pytest` (runs 200+ unit tests using `MockAgent` and in-memory SQLite; no API keys needed)
- **Lint**: No linter is configured in the repo. Standard `pyright` or `mypy` can be used if desired.
- **Build**: `python -m build` (pure-Python wheel, setuptools backend)

### Key notes

- The test suite is fully self-contained: it uses `MockAgent` (from `pygentix.testing`) and `sqlite:///:memory:`, so no Ollama server or cloud API keys are required.
- 17 tests are skipped by design — they are integration tests requiring a live Ollama server (`test_integration.py`, `test_vision.py`). This is expected and not a failure.
- Source code lives in `src/pygentix/`; tests live in `tests/`.
- The package is installed in editable mode (`-e`), so code changes are picked up immediately without reinstalling.
