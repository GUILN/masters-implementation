
# Variables
PYTHON = python
CLI_SCRIPT = cli/model_cli.py
CONFIG_FILE = config.ini
SECRETS_FILE = secrets.ini

# Versioning Variables
MODEL_VERSION = v0.0.0
VERSION_MSG = "Release $(MODEL_VERSION): Multimodal HAR Model\n\
First Implementation\n\
Features:\n\
- GAT-based object and skeleton graph processing\n\
- Video dataset loader with caching\n\
- Training loop with proper batching\n\
- Support for NW-UCLA dataset"


# Default target
.PHONY: help
help:
	@echo "Available Targets:"
	@echo "	test-sentry	-> Cause an error to test Sentry functionality"
	@echo "	test-video-dataloader	-> Tests that videos are being loaded"
	@echo "	version-model-git	-> Versions the model"
	@echo "	help-cli	-> Shows cli help"


# Help
.PHONY: help-cli
help-cli:
	$(PYTHON) $(CLI_SCRIPT) --help

# Tests
.PHONY: test-sentry
test-sentry:
	@echo "Testing Sentry..."
	$(PYTHON) $(CLI_SCRIPT) test-sentry --verbose

.PHONY: test-video-dataloader
test-video-dataloader:
	@echo "Video data loader..."
	$(PYTHON) $(CLI_SCRIPT) test-video-dataloader --verbose

# Versioning
.PHONY: version-model-git
	@echo "Versioning model..."
	git tag -a $(MODEL_VERSION) -m $(VERSION_MSG)
	git push origin $(MODEL_VERSION)
	git push origin --tags
	@echo "Model versioned"
