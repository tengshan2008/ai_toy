"""
Configuration settings for the Code Standardizer package.

This module provides configuration options for the code standardization process.
"""

import os
from typing import Dict, Any
import json
import logging

# Default configuration
DEFAULT_CONFIG = {
    # Similarity threshold for matching
    'similarity_threshold': 0.7,
    
    # Maximum number of worker threads for batch processing
    'max_workers': 4,
    
    # Default paths
    'standard_library_path': 'data/standard_library.json',
    
    # Logging configuration
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': None  # Set to a file path to enable file logging
    }
}


class Config:
    """
    Configuration manager for the Code Standardizer package.
    
    This class provides functionality to load, save, and access configuration settings.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default configuration.
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config_path = config_path
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        
        # Configure logging
        self._configure_logging()
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Update the configuration with loaded values
            self.config.update(loaded_config)
            self.config_path = file_path
            
            # Reconfigure logging with new settings
            self._configure_logging()
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Failed to load configuration: {e}")
    
    def save_to_file(self, file_path: str = None) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path to save the JSON file. If None, uses the path from initialization.
        """
        save_path = file_path or self.config_path
        if not save_path:
            logging.warning("No file path specified for saving configuration")
            return
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: The configuration key
            default: Default value if the key is not found
            
        Returns:
            The configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: The configuration key
            value: The value to set
        """
        self.config[key] = value
        
        # Reconfigure logging if logging settings changed
        if key == 'logging':
            self._configure_logging()
    
    def _configure_logging(self) -> None:
        """Configure logging based on the current configuration."""
        log_config = self.config.get('logging', {})
        
        # Set log level
        level_name = log_config.get('level', 'INFO')
        level = getattr(logging, level_name, logging.INFO)
        
        # Set log format
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Set log file
        log_file = log_config.get('file')
        
        # Configure root logger
        handlers = []
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add new handlers
        for handler in handlers:
            root_logger.addHandler(handler)
