# Dev README

## Install Dev Requirements

```{bash}
pip install -e .[dev]
```

## Linting

To check code quality and formatting, run the following commands:

```bash
uv run ruff format .
uvx flake8 .
uvx mypy .
uv run ruff check . --fix
```

## Tests

```bash
uv add pytest pytest-cov
pytest [--cov=modern_bert_score] [--cov-report=term-missing] [-s]
``
