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
uv venv --python 3.12 && source .venv/bin/activate

# Install in editable mode with dev dependencies
uv sync -U
```

## Running Tests

```bash
uv run pytest
```

## Building Documentation

```bash
uv run mkdocs serve
```

## Code Style

```bash
uv run pre-commit install
uv run pre-commit run --all-files # happens automatically on commit
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
