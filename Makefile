.PHONY: install sync lock fmt lint typecheck test test-fast clean cli help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install all deps incl. dev + notebooks
	uv sync --all-extras

sync: ## Sync runtime deps only
	uv sync

lock: ## Re-lock dependencies
	uv lock

fmt: ## Format code with ruff
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

lint: ## Lint without auto-fix
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck: ## Run mypy
	uv run mypy src

test: ## Run all tests
	uv run pytest

test-fast: ## Run only fast tests (skip slow + live markers)
	uv run pytest -m "not slow and not live"

cli: ## Show CLI help
	uv run trading --help

clean: ## Remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
