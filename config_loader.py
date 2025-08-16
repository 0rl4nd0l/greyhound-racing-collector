#!/usr/bin/env python3
"""
Configuration Loader for Greyhound Racing Prediction System
==========================================================

This module provides a unified configuration loader that reads from:
1. YAML configuration files
2. Environment variables 
3. Command-line interface (CLI) flags

Priority (highest to lowest):
1. CLI flags
2. Environment variables
3. YAML configuration files

Required configuration parameters:
- min_confidence: Minimum prediction confidence threshold
- max_calibration_error: Maximum acceptable calibration error
- form_guide_required_fields: Number of required fields in form guide
- drift_window_days: Number of days for model drift detection window
- imbalance_ratio_warn: Warning threshold for data imbalance ratio

Author: Configuration Management System
Date: January 2025
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Try to import yaml, with fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("Warning: PyYAML not available. YAML config loading will be limited.")
    YAML_AVAILABLE = False
    yaml = None

# Import existing logging configuration
try:
    from config.logging_config import log_config, get_component_logger
except ImportError:
    # Fallback logging if config module not available
    logging.basicConfig(level=logging.INFO)
    def log_config(msg, **kwargs):
        logging.info(f"CONFIG: {msg}")
    def get_component_logger():
        return logging.getLogger("config")


class ConfigurationError(Exception):
    """Raised when configuration loading fails"""
    pass


class ConfigLoader:
    """
    Unified configuration loader supporting YAML, environment variables, and CLI flags.
    
    The loader follows a priority system where CLI flags override environment variables,
    which in turn override YAML configuration values.
    """
    
    # Configuration parameter definitions with types and defaults
    CONFIG_PARAMS = {
        'min_confidence': {
            'type': float,
            'default': 0.20,
            'description': 'Minimum prediction confidence threshold',
            'env_key': 'MIN_CONFIDENCE',
            'cli_flag': '--min-confidence'
        },
        'max_calibration_error': {
            'type': float,
            'default': 0.15,
            'description': 'Maximum acceptable calibration error',
            'env_key': 'MAX_CALIBRATION_ERROR',
            'cli_flag': '--max-calibration-error'
        },
        'form_guide_required_fields': {
            'type': int,
            'default': 10,
            'description': 'Number of required fields in form guide',
            'env_key': 'FORM_GUIDE_REQUIRED_FIELDS',
            'cli_flag': '--form-guide-required-fields'
        },
        'drift_window_days': {
            'type': int,
            'default': 30,
            'description': 'Number of days for model drift detection window',
            'env_key': 'DRIFT_WINDOW_DAYS',
            'cli_flag': '--drift-window-days'
        },
        'imbalance_ratio_warn': {
            'type': float,
            'default': 0.3,
            'description': 'Warning threshold for data imbalance ratio',
            'env_key': 'IMBALANCE_RATIO_WARN',
            'cli_flag': '--imbalance-ratio-warn'
        }
    }

    def __init__(self, config_file: str = 'config/defaults.yaml', 
                 parse_cli_args: bool = True, 
                 cli_args: Optional[list] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_file: Path to YAML configuration file
            parse_cli_args: Whether to parse CLI arguments
            cli_args: List of CLI arguments to parse (for testing)
        """
        self.config_file = Path(config_file)
        self.config = {}
        self.parse_cli_args = parse_cli_args
        self.cli_args = cli_args
        self.logger = get_component_logger()
        
        log_config(f"Initializing ConfigLoader with file: {self.config_file}")
        
        # Load configuration in priority order
        self._load_from_yaml()
        self._override_with_env()
        if self.parse_cli_args:
            self._override_with_cli_args()
        
        # Validate final configuration
        self._validate_config()
        
        log_config("Configuration loaded successfully", 
                  details=self._get_config_summary())

    def _load_from_yaml(self) -> None:
        """
        Load configuration from YAML file.
        
        Raises:
            ConfigurationError: If YAML file cannot be loaded
        """
        try:
            if not YAML_AVAILABLE:
                log_config("PyYAML not available, using default configuration")
                self.config = self._get_default_config()
                return
                
            if not self.config_file.exists():
                log_config(f"YAML config file not found: {self.config_file}, using defaults")
                self.config = self._get_default_config()
                return
                
            with open(self.config_file, 'r', encoding='utf-8') as file:
                yaml_config = yaml.safe_load(file)
                
            if yaml_config is None:
                yaml_config = {}
                
            # Extract specific configuration parameters we need
            self.config = self._extract_config_params(yaml_config)
            
            log_config(f"Loaded YAML configuration from {self.config_file}",
                      details={"loaded_params": list(self.config.keys())})
                      
        except Exception as e:
            if YAML_AVAILABLE and hasattr(yaml, 'YAMLError') and isinstance(e, yaml.YAMLError):
                raise ConfigurationError(f"Invalid YAML in config file {self.config_file}: {e}")
            else:
                raise ConfigurationError(f"Error loading YAML config file {self.config_file}: {e}")

    def _extract_config_params(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant configuration parameters from YAML structure.
        
        The YAML file has nested structure, so we need to extract values from
        appropriate sections like confidence_thresholds, drift_thresholds, etc.
        """
        config = {}
        
        # Map YAML structure to our flat config parameters
        yaml_mappings = {
            'min_confidence': ['confidence_thresholds', 'minimum_confidence'],
            'max_calibration_error': ['model_registry', 'maximum_test_brier_score'],  # Close analog
            'form_guide_required_fields': ['data_quality', 'critical_fields_completeness'],  # Analog
            'drift_window_days': ['drift_thresholds', 'performance_window_days'],
            'imbalance_ratio_warn': ['data_quality', 'maximum_duplicate_records_pct']  # Analog
        }
        
        for param_name, param_config in self.CONFIG_PARAMS.items():
            value = param_config['default']  # Start with default
            
            # Try to get value from YAML structure
            if param_name in yaml_mappings:
                yaml_path = yaml_mappings[param_name]
                try:
                    temp_value = yaml_config
                    for key in yaml_path:
                        temp_value = temp_value[key]
                    value = param_config['type'](temp_value)
                except (KeyError, TypeError, ValueError):
                    # Use default if path doesn't exist or conversion fails
                    pass
                    
            config[param_name] = value
            
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        """
        return {param: config['default'] 
                for param, config in self.CONFIG_PARAMS.items()}

    def _override_with_env(self) -> None:
        """
        Override configuration with environment variables.
        """
        overridden = []
        
        for param_name, param_config in self.CONFIG_PARAMS.items():
            env_key = param_config['env_key']
            if env_key in os.environ:
                try:
                    env_value = param_config['type'](os.environ[env_key])
                    self.config[param_name] = env_value
                    overridden.append(f"{param_name}={env_value} (from {env_key})")
                except (ValueError, TypeError) as e:
                    log_config(f"Invalid environment variable {env_key}: {e}", level="WARNING")
        
        if overridden:
            log_config(f"Environment variable overrides applied", 
                      details={"overridden": overridden})

    def _override_with_cli_args(self) -> None:
        """
        Override configuration with CLI arguments.
        """
        parser = argparse.ArgumentParser(
            description='Greyhound Racing Prediction System Configuration',
            add_help=False  # Don't interfere with main application's help
        )
        
        # Add arguments for each configuration parameter
        for param_name, param_config in self.CONFIG_PARAMS.items():
            parser.add_argument(
                param_config['cli_flag'],
                type=param_config['type'],
                help=param_config['description'],
                dest=param_name
            )
        
        try:
            # Parse known args only to avoid conflicts with other CLI parsers
            if self.cli_args is not None:
                args, _ = parser.parse_known_args(self.cli_args)
            else:
                args, _ = parser.parse_known_args()
            
            overridden = []
            
            # Apply CLI overrides
            for param_name in self.CONFIG_PARAMS.keys():
                cli_value = getattr(args, param_name, None)
                if cli_value is not None:
                    self.config[param_name] = cli_value
                    overridden.append(f"{param_name}={cli_value}")
            
            if overridden:
                log_config(f"CLI argument overrides applied", 
                          details={"overridden": overridden})
                          
        except SystemExit:
            # argparse calls sys.exit on --help, ignore this
            pass
        except Exception as e:
            log_config(f"Error parsing CLI arguments: {e}", level="WARNING")

    def _validate_config(self) -> None:
        """
        Validate the final configuration values.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors = []
        
        # Validate min_confidence
        if not (0.0 <= self.config['min_confidence'] <= 1.0):
            errors.append("min_confidence must be between 0.0 and 1.0")
            
        # Validate max_calibration_error
        if not (0.0 <= self.config['max_calibration_error'] <= 1.0):
            errors.append("max_calibration_error must be between 0.0 and 1.0")
            
        # Validate form_guide_required_fields
        if self.config['form_guide_required_fields'] < 1:
            errors.append("form_guide_required_fields must be at least 1")
            
        # Validate drift_window_days
        if self.config['drift_window_days'] < 1:
            errors.append("drift_window_days must be at least 1")
            
        # Validate imbalance_ratio_warn
        if not (0.0 <= self.config['imbalance_ratio_warn'] <= 1.0):
            errors.append("imbalance_ratio_warn must be between 0.0 and 1.0")
        
        if errors:
            error_msg = "Configuration validation failed: " + "; ".join(errors)
            log_config(error_msg, level="ERROR")
            raise ConfigurationError(error_msg)

    def _get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration for logging.
        """
        return {
            "config_file": str(self.config_file),
            "config_values": self.config.copy()
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            key: Configuration parameter name
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self.config['min_confidence']
    
    def get_max_calibration_error(self) -> float:
        """Get maximum calibration error threshold."""
        return self.config['max_calibration_error']
    
    def get_form_guide_required_fields(self) -> int:
        """Get number of required form guide fields."""
        return self.config['form_guide_required_fields']
    
    def get_drift_window_days(self) -> int:
        """Get drift detection window in days."""
        return self.config['drift_window_days']
    
    def get_imbalance_ratio_warn(self) -> float:
        """Get imbalance ratio warning threshold."""
        return self.config['imbalance_ratio_warn']
    
    def reload(self) -> None:
        """
        Reload configuration from all sources.
        """
        log_config("Reloading configuration")
        self.__init__(str(self.config_file), self.parse_cli_args, self.cli_args)


# Global configuration instance
_global_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_file: str = 'config/defaults.yaml') -> ConfigLoader:
    """
    Get the global configuration loader instance.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _global_config_loader
    
    if _global_config_loader is None:
        _global_config_loader = ConfigLoader(config_file)
    
    return _global_config_loader


def get_config(config_file: str = 'config/defaults.yaml') -> Dict[str, Any]:
    """
    Convenience function to get configuration dictionary.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    return get_config_loader(config_file).get_config()


if __name__ == "__main__":
    import json
    
    print("🔧 Greyhound Racing Configuration Loader")
    print("=" * 50)
    
    try:
        # Test with default configuration file
        config_loader = ConfigLoader('config/defaults.yaml')
        config = config_loader.get_config()
        
        print("\n📋 Loaded Configuration:")
        print(json.dumps(config, indent=2))
        
        print("\n🎯 Specific Parameter Access:")
        print(f"Min Confidence: {config_loader.get_min_confidence()}")
        print(f"Max Calibration Error: {config_loader.get_max_calibration_error()}")
        print(f"Form Guide Required Fields: {config_loader.get_form_guide_required_fields()}")
        print(f"Drift Window Days: {config_loader.get_drift_window_days()}")
        print(f"Imbalance Ratio Warning: {config_loader.get_imbalance_ratio_warn()}")
        
        print("\n💡 Environment Variable Examples:")
        print("export MIN_CONFIDENCE=0.25")
        print("export DRIFT_WINDOW_DAYS=45")
        
        print("\n🚀 CLI Flag Examples:")
        print("python config_loader.py --min-confidence 0.3")
        print("python your_script.py --drift-window-days 21 --imbalance-ratio-warn 0.4")
        
        print("\n✅ Configuration loader test completed successfully!")
        
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)
