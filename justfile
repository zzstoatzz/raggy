# Check for uv installation
check-uv:
    #!/usr/bin/env sh
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv is not installed or not found in expected locations."
        case "$(uname)" in
            "Darwin")
                echo "To install uv on macOS, run one of:"
                echo "• brew install uv"
                echo "• curl -LsSf https://astral.sh/uv/install.sh | sh"
                ;;
            "Linux")
                echo "To install uv, run:"
                echo "• curl -LsSf https://astral.sh/uv/install.sh | sh"
                ;;
            *)
                echo "To install uv, visit: https://github.com/astral-sh/uv"
                ;;
        esac
        exit 1
    fi

# Build and serve documentation
serve-docs: check-uv
    uv run mkdocs serve

# Install development dependencies
install: check-uv
    uv sync

# Run linting and type checking
lint: check-uv install
    uv run ruff check .
    uv run ruff format .

# Run type checking
typecheck: check-uv
    uv run ty check

# Clean up environment
clean: check-uv
    deactivate || true
    rm -rf .venv

run-pre-commits: check-uv
    uv run pre-commit run --all-files

refresh-prefect-docs-mcp *args:
    uv run examples/prefect_docs_mcp/refresh_namespace.py {{args}}