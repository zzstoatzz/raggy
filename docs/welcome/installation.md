# Installation

To install the package from PyPI:

```bash
# using pip
pip install raggy

# using uv
uv pip install raggy
```

!!! question "What's `uv`?"
    [Well I'm glad you asked 🙂](https://github.com/astral-sh/uv?tab=readme-ov-file#uv)

## Requirements
`raggy` unapolagetically requires python 3.10+.

## Optional dependencies
`raggy` offers a few optional dependencies that can be installed as extras:

- `chroma` - for using the `Chroma` vectorstore
- `tpuf` - for using the (managed) `Turbopuffer` vectorstore
- `pdf` - for parsing PDFs

## Development

Clone the repo:
    
```bash
git clone https://github.com/zzstoatzz/raggy.git
cd raggy
```

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```