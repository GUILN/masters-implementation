
REQUIREMENTS_DEV=requirements-dev.txt


.PHONY: help
help:
	@echo "Available targets:"
	@echo "  freeze-dev   - Freeze the current Python environment requirements"
	@echo "  test         - Run all tests using pytest"


.PHONY: freeze-dev
freeze-dev:
	pip freeze > $(REQUIREMENTS_DEV)

.PHONY: test
test:
	pytest tests

.PHONY: example
example:
	python example_dump_video.py
