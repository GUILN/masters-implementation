import configparser
import os
from pathlib import Path
from typing import Optional, NamedTuple


class ModelSettings(NamedTuple):
    model_save_dir: Path
    video_data_dir: Path


class LoggingSettings(NamedTuple):
    level: str
    log_file: Path


class SentrySettings(NamedTuple):
    dsn: Optional[str]
    environment: str


class ModelConfig:
    """Configuration manager for video extraction scripts."""

    def __init__(
        self,
        config_file: str = 'config.ini',
        secrets_file: str = 'secrets.ini'
    ) -> None:
        self.config = configparser.ConfigParser()
        self.secrets = configparser.ConfigParser()
        self.config_file = config_file if config_file else 'config.ini'
        self.secrets_file = secrets_file if secrets_file else 'secrets.ini'
        self.load_config()
        self.load_secrets()
        self._model_settings = ModelSettings(
            model_save_dir=Path(self.get('model', 'model_save_dir')),
            video_data_dir=Path(self.get('model', 'video_data_dir')),
        )
        self._logging_settings = LoggingSettings(
            level=self.get('logging', 'level'),
            log_file=Path(self.get('logging', 'log_file'))
        )
        self._sentry_settings = SentrySettings(
            dsn=self.get_secret('sentry', 'dsn'),
            environment=self.get_secret('sentry', 'environment', fallback='development')
        )

    def load_config(self) -> None:
        """Load configuration from INI file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(
                f"Configuration file '{self.config_file}' not found"
            )

        self.config.read(self.config_file)

    def load_secrets(self) -> None:
        """Load secrets from INI file (optional)."""
        if os.path.exists(self.secrets_file):
            self.secrets.read(self.secrets_file)
        else:
            print(
                f"Warning: Secrets file '{self.secrets_file}' not found. Create it from secrets.ini.template"
            )

    def get(self, section: str, key: str, fallback: Optional[str] = None) -> str:
        """Get a configuration value with optional fallback."""
        return self.config.get(section, key, fallback=fallback)

    def getint(self, section: str, key: str, fallback: Optional[int] = None) -> int:
        """Get an integer configuration value."""
        return self.config.getint(section, key, fallback=fallback)

    def getfloat(self, section: str, key: str, fallback: Optional[float] = None) -> float:
        """Get a float configuration value."""
        return self.config.getfloat(section, key, fallback=fallback)

    def getboolean(self, section: str, key: str, fallback: Optional[bool] = None) -> bool:
        """Get a boolean configuration value."""
        return self.config.getboolean(section, key, fallback=fallback)

    def get_secret(self, section: str, key: str, fallback: Optional[str] = None) -> Optional[str]:
        """Get a secret value with optional fallback."""
        if self.secrets.has_section(section) and self.secrets.has_option(section, key):
            return self.secrets.get(section, key, fallback=fallback)
        return fallback

    def has_secret(self, section: str, key: str) -> bool:
        """Check if a secret exists."""
        return self.secrets.has_section(section) and self.secrets.has_option(section, key)

    @property
    def model_settings(self) -> ModelSettings:
        """Get model settings."""
        return self._model_settings

    @property
    def logging_settings(self) -> LoggingSettings:
        """Get logging settings."""
        return self._logging_settings

    @property
    def sentry_settings(self) -> SentrySettings:
        """Get Sentry settings from secrets."""
        return self._sentry_settings
