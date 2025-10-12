
import argparse
from enum import Enum
import sys
from typing import List, NamedTuple, Optional
from pathlib import Path


def create_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    config_path = Path(path_str)
    if not config_path.exists():
        print(
            f"Error: Config file '{path_str}' does not exist.",
            file=sys.stderr
        )
        raise FileNotFoundError(f"Config file '{path_str}' does not exist.")


class Command(str, Enum):
    TEST_SENTRY = "test-sentry"

    @classmethod
    def create_from_str(cls, value: str) -> "Command":
        try:
            return cls(value.lower())
        except ValueError as e:
            raise ValueError(f"Command '{value}' is not valid", e)

    @classmethod
    def get_all_options(cls) -> List[str]:
        return [command.value for command in cls]


class ModelArgs(NamedTuple):
    command: Command
    verbose: bool

    @classmethod
    def parse_args(cls) -> "ModelArgs":
        parser = argparse.ArgumentParser(
            description="Model CLI - A command-line interface for model ops",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=""
        )
        parser.add_argument(
            "command",
            choices=Command.get_all_options(),
            help="Command to execute test sentry"
        )

        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )

        args = parser.parse_args()
        command = Command.create_from_str(args.command)
        verbose = bool(args.verbose)

        return cls(
            command=command,
            verbose=verbose,
        )
