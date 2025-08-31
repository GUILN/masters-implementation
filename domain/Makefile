
REQUIREMENTS_DEV=requirements-dev.txt

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  freeze-dev   - Freeze the current Python environment requirements"

.PHONY: freeze-dev
freeze-dev:
	pip freeze > $(REQUIREMENTS_DEV)
