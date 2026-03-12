code-check:
	uv run ruff check src/
	uv run pyright src/

code-clean:
	uv run ruff check --fix-only src/
	uv run ruff check --select I --fix src/
	uv run ruff format src/

help:
	@echo "Available commands:"
	@echo "  code-check              - Run linting and type checking"
	@echo "  code-clean              - Clean code"

.PHONY: code-check code-clean
