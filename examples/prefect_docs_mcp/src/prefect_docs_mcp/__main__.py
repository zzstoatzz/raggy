"""CLI entry point for the Prefect docs MCP server."""

from prefect_docs_mcp.server import prefect_docs_mcp


def main() -> None:
    prefect_docs_mcp.run()


if __name__ == "__main__":
    main()
