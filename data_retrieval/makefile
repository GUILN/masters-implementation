# Data Retrieval Makefile

# Variables
PYTHON = python
CLI_SCRIPT = cli_extract.py
SKELETON_CLI_SCRIPT = cli_extract_skeleton.py
IMAGE_TO_VISUALIZE = "/home/guilherme/Mestrado/masters-implementation/data_retrieval/data/output/ntu_rgb_d_60/nturgbd_rgb_s013/nturgb+d_rgb/S013C003P037R001A033_rgb_frames/S013C003P037R001A033_rgb_frame_000004.jpg"
CONFIG_FILE = config.ini

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  test-cli-skeleton    - Test CLI with skeleton extraction (dry run)"
	@echo "  test-cli-objects     - Test CLI with object extraction (dry run)"
	@echo "  test-cli-frame       - Test CLI with frame extraction (dry run)"
	@echo "  test-cli-help        - Show CLI help"
	@echo "  test-config          - Test configuration loading"
	@echo "  test-sentry          - Test Sentry integration by triggering an exception"
	@echo "  run-cli-frame        - Run CLI with frame extraction"
	@echo "  run-cli-objects      - Run CLI with object extraction"
	@echo "  run-cli-skeleton     - Run CLI with skeleton extraction"
	@echo "  visualize-skeleton    - Visualize skeleton extraction results"
	@echo "  setup-secrets        - Create secrets.ini from template"
	@echo "  clean                - Clean temporary files"
	@echo "  install              - Install dependencies"

# CLI Testing targets
.PHONY: test-cli-skeleton
test-cli-skeleton:
	@echo "Testing CLI with skeleton extraction..."
	$(PYTHON) $(CLI_SCRIPT) skeleton --dry-run --verbose

.PHONY: test-cli-objects
test-cli-objects:
	@echo "Testing CLI with object extraction..."
	$(PYTHON) $(CLI_SCRIPT) objects --dry-run --verbose

.PHONY: test-cli-frame
test-cli-frame:
	@echo "Testing CLI with frame extraction..."
	$(PYTHON) $(CLI_SCRIPT) frame --dry-run --verbose

.PHONY: test-cli-help
test-cli-help:
	@echo "Showing CLI help..."
	$(PYTHON) $(CLI_SCRIPT) --help

.PHONY: test-config
test-config:
	@echo "Testing configuration loading..."
	$(PYTHON) config_manager.py

.PHONY: test-sentry
test-sentry:
	@echo "Testing Sentry integration..."
	$(PYTHON) $(CLI_SCRIPT) --test-sentry


# cli run targets

.PHONY: run-cli-frame
run-cli-frame:
	@echo "Running CLI with frame extraction..."
	$(PYTHON) $(CLI_SCRIPT) frame --verbose

.PHONY: run-cli-objects
run-cli-objects:
	@echo "Running CLI with object extraction..."
	$(PYTHON) $(CLI_SCRIPT) objects --verbose

.PHONY: run-cli-skeleton
run-cli-skeleton:
	@echo "Running CLI with skeleton extraction..."
	$(PYTHON) $(SKELETON_CLI_SCRIPT) --verbose

.PHONY: visualize-skeleton
visualize-skeleton:
	@echo "Visualizing skeleton"
	$(PYTHON) $(SKELETON_CLI_SCRIPT) --visualize $(IMAGE_TO_VISUALIZE) --verbose



# Setup
.PHONY: setup-secrets
setup-secrets:
	@if [ ! -f secrets.ini ]; then \
		echo "Creating secrets.ini from template..."; \
		cp secrets.ini.template secrets.ini; \
		echo "✅ secrets.ini created! Please edit it with your actual secrets."; \
		echo "📖 See SECRETS_README.md for instructions."; \
	else \
		echo "⚠️  secrets.ini already exists."; \
	fi

# Installation
.PHONY: install
install:
	pip install -r requirements.txt

# Cleanup
.PHONY: clean
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f *.log