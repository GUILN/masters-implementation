
# Variables
PYTHON = python
CLI_SCRIPT = cli/model_cli.py
CONFIG_FILE = config.ini
SECRETS_FILE = secrets.ini

# Default target
.PHONY: help
help:
	@echo "Available Targets:"
	@echo "	test-sentry	-> Cause an error to test Sentry functionality"
	@echo "	test-video-dataloader	-> Tests that videos are being loaded"
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