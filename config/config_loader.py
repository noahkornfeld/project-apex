"""
Configuration Loader for Project Apex
=====================================

Provides utilities for loading and validating configuration files.
Implements Gate 1 requirement: "Loading invalid config raises ValidationError"

Usage:
    from config.config_loader import load_config
    
    config = load_config("config/master_config.yaml")
"""

from pathlib import Path
from typing import Union
import yaml

from config.config_schema import ProjectConfig, ValidationError


def load_config(config_path: Union[str, Path]) -> ProjectConfig:
    """
    Load and validate configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated ProjectConfig instance
        
    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
        
    Examples:
        >>> config = load_config("config/master_config.yaml")
        >>> print(config.sac.gamma)
        0.975
    """
    try:
        config = ProjectConfig.from_yaml(str(config_path))
        return config
    except AssertionError as e:
        raise ValidationError(f"Configuration validation failed: {str(e)}") from e
    except Exception as e:
        raise ValidationError(f"Failed to load configuration: {str(e)}") from e


def validate_config_dict(config_dict: dict) -> ProjectConfig:
    """
    Validate configuration from dictionary
    
    Args:
        config_dict: Configuration as nested dictionary
        
    Returns:
        Validated ProjectConfig instance
        
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        return ProjectConfig.from_dict(config_dict)
    except AssertionError as e:
        raise ValidationError(f"Configuration validation failed: {str(e)}") from e
    except Exception as e:
        raise ValidationError(f"Failed to validate configuration: {str(e)}") from e


def create_default_config(output_path: Union[str, Path]) -> ProjectConfig:
    """
    Create default configuration file
    
    Args:
        output_path: Path where to save the default config
        
    Returns:
        Default ProjectConfig instance
    """
    config = ProjectConfig()
    config.save_yaml(str(output_path))
    return config


def merge_configs(base_config: ProjectConfig, override_dict: dict) -> ProjectConfig:
    """
    Merge override values into base configuration
    
    Args:
        base_config: Base configuration
        override_dict: Dictionary of values to override
        
    Returns:
        Merged ProjectConfig instance
        
    Raises:
        ValidationError: If merged config is invalid
    """
    base_dict = base_config.to_dict()
    
    # Deep merge
    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(base_dict, override_dict)
    
    return validate_config_dict(base_dict)


if __name__ == "__main__":
    # Test loading default config
    print("Testing config loader...")
    
    config_path = Path(__file__).parent / "master_config.yaml"
    
    try:
        config = load_config(config_path)
        print(f"✓ Successfully loaded config from {config_path}")
        print(f"  SAC gamma: {config.sac.gamma}")
        print(f"  K_max: {config.architecture.K_max}")
        print(f"  Reward lambda_slow: {config.reward.lambda_slow}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
