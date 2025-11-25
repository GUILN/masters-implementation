
# Variables
PYTHON = python
CLI_SCRIPT = cli/model_cli.py
CONFIG_FILE = config.ini
SECRETS_FILE = secrets.ini

# Versioning Variables
MODEL_VERSION = v1.0.1
VERSION_MSG = "Release $(MODEL_VERSION): Regularization Strategies\n\
Features:\n\
- temporal modeling - final implementation"


# Default target
.PHONY: help
help:
	@echo "Available Targets:"
	@echo "	test-sentry	-> Cause an error to test Sentry functionality"
	@echo "	test-video-dataloader	-> Tests that videos are being loaded"
	@echo "	version-model-git	-> Versions the model"
	@echo "	split-nwucla-dataset	-> Splits nwucla into train and test datasets"
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
version-model-git:
	@echo "Versioning model..."
	git add makefile
	git commit -m "new version - $(MODEL_VERSION)"
	git push
	git tag -a $(MODEL_VERSION) -m $(VERSION_MSG)
	git push origin $(MODEL_VERSION)
	git push origin --tags
	@echo "Model versioned"

# Helpers
.PHONY: split-nwucla-dataset
split-nwucla-dataset:
	@echo "Testing Sentry..."
	$(PYTHON) $(CLI_SCRIPT) split-nwucla-dataset --verbose

