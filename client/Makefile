.PHONY: test integration lint typecheck all

PYTEST ?= python -m pytest
RUFF ?= python -m ruff
MYPY ?= python -m mypy

test:
	$(PYTEST) tests/

integration:
	MEMORY_MCP_INTEGRATION_URL=$${MEMORY_MCP_INTEGRATION_URL:-http://127.0.0.1:8080/mcp} \
		$(PYTEST) tests/integration/ -m integration

lint:
	$(RUFF) check src/ tests/

typecheck:
	$(MYPY) src/

all: lint typecheck test
