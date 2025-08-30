import logging
from logging import StreamHandler, FileHandler, Formatter
from typing import Optional

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

class ApplicationLogger:
    def __init__(
        self,
        name: str = "SimpleLogger",
        loglevel: int = logging.INFO,
        logfile: Optional[str] = None,
        log_to_stdout: bool = True,
        sentry_dsn: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(loglevel)
        formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if logfile:
            file_handler = FileHandler(logfile)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if log_to_stdout:
            stream_handler = StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        if sentry_dsn and SENTRY_AVAILABLE:
            sentry_sdk.init(sentry_dsn)
            self.sentry_enabled = True
        else:
            self.sentry_enabled = False

    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, **kwargs)
        if self.sentry_enabled:
            sentry_sdk.capture_message(msg, level="error")

    def exception(self, msg: str, **kwargs):
        self.logger.exception(msg, **kwargs)
        if self.sentry_enabled:
            sentry_sdk.capture_exception()
