
from config_manager import ExtractionConfig
from app_logging.application_logger import ApplicationLogger

# Module-level flag and logger instance
_logger_initialized = False
_logger_instance = None

class CommonSetup:
    @staticmethod
    def get_logger() -> ApplicationLogger:
        """Setup logging based on configuration. Only executes once."""
        global _logger_initialized, _logger_instance
        
        # Return existing logger if already initialized
        if _logger_initialized and _logger_instance is not None:
            return _logger_instance
       
        config = ExtractionConfig() 
        log_settings = config.logging_settings
        sentry_settings = config.sentry_settings

        # Create logger using app_logging with Sentry DSN from secrets
        logger = ApplicationLogger(
            name="data_extraction",
            loglevel=log_settings['level'],
            logfile=log_settings['log_file'],
            log_to_stdout=True,
            sentry_dsn=sentry_settings['dsn']
        )
        logger.info(f"Sentry DSN set to: {sentry_settings['dsn']}")

        if sentry_settings['dsn']:
            logger.info(f"Sentry initialized with environment: {sentry_settings['environment']}")
        else:
            logger.warning("Sentry DSN not configured - errors will not be sent to Sentry")

        # Mark as initialized and store instance
        _logger_initialized = True
        _logger_instance = logger
        
        return logger
