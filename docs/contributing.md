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

```
pre-commit install
pre-commit run --all-files # happens automatically on commit
```

## Running Examples

All examples can be run using uv:

!!! question "where are the dependencies?"
    `uv` will run the example in an isolated environment using [inline script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).

```bash
# Run example
uv run examples/chat_with_X/website.py
```

See our [example gallery](examples/index.md) for more details.

## Versioning

We use [Semantic Versioning](http://semver.org/). For the versions available, see the [tags on this repository](https://github.com/zzstoatzz/raggy/tags).
