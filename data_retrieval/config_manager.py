import configparser
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union

class ExtractionConfig:
    """Configuration manager for video extraction scripts."""
    
    def __init__(self, config_file: str = 'config.ini', secrets_file: str = 'secrets.ini') -> None:
        self.config = configparser.ConfigParser()
        self.secrets = configparser.ConfigParser()
        self.config_file = config_file
        self.secrets_file = secrets_file
        self.load_config()
        self.load_secrets()
    
    def load_config(self) -> None:
        """Load configuration from INI file."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found")
        
        self.config.read(self.config_file)
    
    def load_secrets(self) -> None:
        """Load secrets from INI file (optional)."""
        if os.path.exists(self.secrets_file):
            self.secrets.read(self.secrets_file)
        else:
            print(f"Warning: Secrets file '{self.secrets_file}' not found. Create it from secrets.ini.template")
    
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
    def frame_extraction_settings(self) -> Dict[str, Union[int, str, Path]]:
        """Get frame extraction settings."""
        return {
            'frame_rate_per_second': self.getint('frame_extraction', 'frame_rate_per_second'),
            'resolution': self.get('frame_extraction', 'resolution'),
            'batch_size': self.getint('frame_extraction', 'batch_size'),
            'input_dir': Path(self.get('frame_extraction', 'input_dir')),
            'output_dir': Path(self.get('frame_extraction', 'output_dir')),
            'temp_dir': Path(self.get('frame_extraction', 'temp_dir')),
            'input_format': self.get('DEFAULT', 'input_format'),
            'output_format': self.get('DEFAULT', 'output_format')
        }
    
    @property
    def skeleton_extraction_settings(self) -> Dict[str, Union[Path, float, int]]:
        """Get skeleton extraction settings."""
        return {
            'model_path': Path(self.get('skeleton_extraction', 'model_path')),
            'confidence_threshold': self.getfloat('skeleton_extraction', 'confidence_threshold'),
            'max_persons': self.getint('skeleton_extraction', 'max_persons')
        }
    
    @property
    def object_extraction_settings(self) -> Dict[str, Union[Path, float, int]]:
        """Get object extraction settings."""
        return {
            'model_path': Path(self.get('object_extraction', 'model_path')),
            'confidence_threshold': self.getfloat('object_extraction', 'confidence_threshold'),
            'max_objects': self.getint('object_extraction', 'max_objects')
        }
    
    @property
    def paths(self) -> Dict[str, Path]:
        """Get path settings."""
        return {
            'input_dir': Path(self.get('paths', 'input_dir')),
            'output_dir': Path(self.get('paths', 'output_dir')),
            'temp_dir': Path(self.get('paths', 'temp_dir'))
        }
    
    @property
    def logging_settings(self) -> Dict[str, str]:
        """Get logging settings."""
        return {
            'level': self.get('logging', 'level'),
            'log_file': self.get('logging', 'log_file')
        }
    
    @property
    def sentry_settings(self) -> Dict[str, Optional[str]]:
        """Get Sentry settings from secrets."""
        return {
            'dsn': self.get_secret('sentry', 'dsn'),
            'environment': self.get_secret('sentry', 'environment', fallback='development')
        }


# Example usage in an extraction script
if __name__ == "__main__":
    # Load configuration
    config: ExtractionConfig = ExtractionConfig()
    
    # Access configuration values
    print("Frame Extraction Settings:")
    frame_extraction: Dict[str, Union[int, str, Path]] = config.frame_extraction_settings
    print(f"  Frame rate per second: {frame_extraction['frame_rate_per_second']}")
    print(f"  Resolution: {frame_extraction['resolution']}")
    print(f"  Batch size: {frame_extraction['batch_size']}")
    print(f"  Input directory: {frame_extraction['input_dir']}")
    print(f"  Output directory: {frame_extraction['output_dir']}")
    print(f"  Temp directory: {frame_extraction['temp_dir']}")
    
    print("\nSkeleton Extraction Settings:")
    skeleton_extraction: Dict[str, Union[Path, float, int]] = config.skeleton_extraction_settings
    print(f"  Model path: {skeleton_extraction['model_path']}")
    print(f"  Confidence threshold: {skeleton_extraction['confidence_threshold']}")
    print(f"  Max persons: {skeleton_extraction['max_persons']}")
    
    print("\nObject Extraction Settings:")
    object_extraction: Dict[str, Union[Path, float, int]] = config.object_extraction_settings
    print(f"  Model path: {object_extraction['model_path']}")
    print(f"  Confidence threshold: {object_extraction['confidence_threshold']}")
    print(f"  Max objects: {object_extraction['max_objects']}")
    
    print("\nPaths:")
    paths: Dict[str, Path] = config.paths
    print(f"  Input directory: {paths['input_dir']}")
    print(f"  Output directory: {paths['output_dir']}")
    print(f"  Temp directory: {paths['temp_dir']}")
    
    print("\nSentry Settings:")
    sentry_settings: Dict[str, Optional[str]] = config.sentry_settings
    if sentry_settings['dsn']:
        print(f"  DSN: {sentry_settings['dsn'][:50]}..." if len(sentry_settings['dsn']) > 50 else f"  DSN: {sentry_settings['dsn']}")
        print(f"  Environment: {sentry_settings['environment']}")
    else:
        print("  DSN: Not configured")
    
    # Create directories if they don't exist
    paths['output_dir'].mkdir(parents=True, exist_ok=True)
    paths['temp_dir'].mkdir(parents=True, exist_ok=True)
