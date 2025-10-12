import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from cli.cli_args import ModelArgs, Command
except ImportError:
    from cli_args import ModelArgs, Command


def main():
    """Main CLI entry point."""
    args = ModelArgs.parse_args()
    print(args)
    if args.command == Command.TEST_SENTRY:
        run_sentry_test()


def run_sentry_test():
    """Execute sentry mode operations."""
    print("Sentry mode specific operations:")
    print("- Monitoring and logging enabled")
    print("- Error tracking active")


if __name__ == "__main__":
    main()
