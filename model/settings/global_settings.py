
class GlobalSettings:
    pass

import os
from app_logging.application_logger import ApplicationLogger
from config.config_manager import ModelConfig

LOGGER_NAME = "main_model"

_logger_initialized = False
_logger_instance = None

_config_initialized = False
_config_instance = None


class GlobalSettings:
    @staticmethod
    def get_config(
            config_file: str = "config.ini",
            secrets_file: str = "secrets.ini"
    ) -> ModelConfig:
        """Retrieve the global configuration instance."""
        global _config_initialized, _config_instance
        if _config_initialized and _config_instance is not None:
            return _config_instance
        _config_instance = ModelConfig(
            config_file=config_file,
            secrets_file=secrets_file
        )
        print("Loading config...")
        _config_instance.load_config()
        print("Loading secrets...")
        _config_instance.load_secrets()
        _config_initialized = True
        return _config_instance

    @staticmethod
    def get_logger() -> ApplicationLogger:
        """Setup logging based on configuration. Only executes once."""
        global _logger_initialized, _logger_instance

        # Return existing logger if already initialized
        if _logger_initialized and _logger_instance is not None:
            return _logger_instance

        process_id = os.getpid()
        config = GlobalSettings.get_config()
        log_settings = config.logging_settings
        sentry_settings = config.sentry_settings

        # Create logger using app_logging with Sentry DSN from secrets
        logger = ApplicationLogger(
            name=LOGGER_NAME,
            loglevel=log_settings.level,
            logfile=f"{log_settings.log_file}.{process_id}.log",
            log_to_stdout=True,
            sentry_dsn=sentry_settings.dsn
        )
        logger.info(f"Sentry DSN set to: {sentry_settings.dsn}")

        if sentry_settings.dsn:
            logger.info(
                f"Sentry initialized with environment: {sentry_settings.environment}"
            )
        else:
            logger.warning("Sentry DSN not configured - errors will not be sent to Sentry")

        _logger_initialized = True
        _logger_instance = logger

        return logger
