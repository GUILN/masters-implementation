# Utils Module for Masters Implementation

This directory contains utility modules for the monorepo, including a configurable logging module with Sentry support.

## Installation

You can install this package directly from GitHub using pip:

```
pip install git+https://github.com/GUILN/masters-implementation.git#subdirectory=utils
```

This will make all utilities under `utils` available for import in your projects.

## Example Usage

```
from logging.application_logger import ApplicationLogger

logger = ApplicationLogger(loglevel=20, logfile="example.log", log_to_stdout=True, sentry_dsn=None)
logger.info("This is an info message.")
```

See `logging/example_usage.py` for a complete example.
