# Contributing to Raggy

We love your input! We want to make contributing to Raggy as easy and transparent as possible.

## Development Setup

We recommend using [uv](https://github.com/astral-sh/uv) for Python environment management and package installation:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/zzstoatzz/raggy.git
cd raggy

# Create and activate a virtual environment
uv venv

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

## Running Tests

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run tests
pytest
```

## Building Documentation

```bash
# Install docs dependencies
uv pip install -e ".[docs]"

# Serve docs locally
mkdocs serve
```

## Code Style

We use:

- `ruff` for linting and formatting
- `mypy` for type checking
- `pytest` for testing

You can install and run these tools using uv:

```bash
# Install tools
uv pip install ruff mypy pytest

# Run checks
ruff check .
ruff format .
mypy src
pytest
```

## Running Examples

All examples can be run using uv:

```bash
# Install example dependencies
uv pip install -r <(uv pip parse-script examples/chat_with_X/website.py)

# Run example
uv run examples/chat_with_X/website.py
```

See our [example gallery](examples/index.md) for more details.

## Versioning

We use [Semantic Versioning](http://semver.org/). For the versions available, see the [tags on this repository](https://github.com/zzstoatzz/raggy/tags).

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
