"""Configuration management for Quantopia."""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    'llm': {
        'provider': 'openai',
        'model': 'gpt-4',
        'temperature': 0.7
    },
    'database': {
        'path': 'data/strategies.db'
    },
    'ml': {
        'registry_dir': 'models/registry',
        'enable_background_improvement': True
    },
    'agents': {
        'exploration_rate': 0.3,
        'max_refinement_attempts': 2,
        'strategy_family_size': 3
    },
    'daemon': {
        'pid_file': 'data/quantopia.pid',
        'log_file': 'logs/quantopia.log'
    },
    'research': {
        'default_symbol': 'BTC-USD',
        'default_days': 365,
        'batch_size': 10
    }
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)

    if not path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()

    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # Merge with defaults
        merged = _merge_configs(DEFAULT_CONFIG, config)

        # Expand environment variables
        merged = _expand_env_vars(merged)

        logger.info(f"Loaded configuration from: {config_path}")
        return merged

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG.copy()


def init_config(config_path: str, force: bool = False):
    """
    Initialize configuration file.

    Args:
        config_path: Path to create configuration file
        force: Overwrite if exists
    """
    path = Path(config_path)

    if path.exists() and not force:
        logger.error(f"Config file already exists: {config_path}")
        logger.info("Use --force to overwrite")
        return

    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write default config
    with open(path, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created configuration file: {config_path}")
    logger.info("Edit this file to customize Quantopia settings")


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config."""
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def _expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in config values."""
    expanded = {}

    for key, value in config.items():
        if isinstance(value, dict):
            expanded[key] = _expand_env_vars(value)
        elif isinstance(value, str):
            expanded[key] = os.path.expandvars(value)
        else:
            expanded[key] = value

    return expanded
