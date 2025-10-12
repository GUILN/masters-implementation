import sys
from pathlib import Path

# Add the parent directory to the Python path first
sys.path.append(str(Path(__file__).parent.parent))

# Now import the modules
try:
    from cli.cli_args import ModelArgs, Command
except ImportError:
    from cli_args import ModelArgs, Command

try:
    from config.config_manager import ModelConfig
except ImportError:
    from config_manager import ModelConfig


# Loading CLI arguments and configuration
print("Loading CLI arguments and configuration...")  # Debug statement
args = ModelArgs.parse_args()
print("Loading config...")
model_config = ModelConfig(args.config, args.secrets)
model_config.load_config()
print("Loading secrets...")
model_config.load_secrets()


def show_settings():
    print("Current Configuration Settings:")
    print("Logging settings:")
    print(model_config.logging_settings)
    print("Model settings:")
    print(model_config.model_settings)
    print("Sentry settings:")
    print(model_config.sentry_settings)


def main():
    """Main CLI entry point."""
    print(args)
    if args.verbose:
        print("Verbose mode enabled")
        show_settings()
    if args.command == Command.TEST_SENTRY:
        run_sentry_test()


def run_sentry_test():
    """Execute sentry mode operations."""
    print("Sentry mode specific operations:")
    print("- Monitoring and logging enabled")
    print("- Error tracking active")


if __name__ == "__main__":
    main()
