"""Configuration management for SimulatedExchange."""

from typing import Dict, Optional, Any
from datetime import datetime
import yaml
import json
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for SimulatedExchange."""

    DEFAULT_CONFIG = {
        'exchange': {
            'initial_capital': 10000.0,
            'mode': 'backtest'
        },
        'slippage': {
            'model': 'fixed',
            'fixed_bps': 5
        },
        'fees': {
            'model': 'tiered',
            'maker_fee': 0.0000,  # Hyperliquid: 0%
            'taker_fee': 0.00025  # Hyperliquid: 0.025%
        },
        'backtest': {
            'symbols': ['BTC-USD', 'ETH-USD'],
            'timeframe': '1h'
        },
        'live': {
            'exchange': 'binance',
            'testnet': True,
            'symbols': ['BTC-USD']
        }
    }

    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize configuration.

        Args:
            config_dict: Configuration dictionary (uses defaults if None)
        """
        self._config = self.DEFAULT_CONFIG.copy()
        if config_dict:
            self._deep_update(self._config, config_dict)

    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from file.

        Args:
            filepath: Path to config file (.yaml, .yml, or .json)

        Returns:
            Config instance

        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If file not found
        """
        if filepath.endswith(('.yaml', '.yml')):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {filepath}")

        logger.info(f"Loaded configuration from {filepath}")
        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (supports dot notation, e.g., 'exchange.initial_capital')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def save(self, filepath: str):
        """Save configuration to file.

        Args:
            filepath: Path to save config file (.yaml, .yml, or .json)

        Raises:
            ValueError: If file format not supported
        """
        if filepath.endswith(('.yaml', '.yml')):
            with open(filepath, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {filepath}")

        logger.info(f"Saved configuration to {filepath}")

    def _deep_update(self, base: Dict, update: Dict):
        """Deep update base dict with update dict."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value


# Predefined configurations

HYPERLIQUID_CONFIG = {
    'exchange': {
        'initial_capital': 10000.0,
        'mode': 'backtest'
    },
    'slippage': {
        'model': 'fixed',
        'fixed_bps': 3
    },
    'fees': {
        'model': 'tiered',
        'maker_fee': 0.0000,
        'taker_fee': 0.00025
    },
    'live': {
        'exchange': 'hyperliquid',
        'testnet': True
    }
}

BINANCE_CONFIG = {
    'exchange': {
        'initial_capital': 10000.0,
        'mode': 'backtest'
    },
    'slippage': {
        'model': 'fixed',
        'fixed_bps': 5
    },
    'fees': {
        'model': 'tiered',
        'maker_fee': 0.0002,
        'taker_fee': 0.0004
    },
    'live': {
        'exchange': 'binance',
        'testnet': True
    }
}


def get_config(preset: Optional[str] = None) -> Config:
    """Get a configuration instance.

    Args:
        preset: Preset name ('hyperliquid', 'binance', or None for default)

    Returns:
        Config instance
    """
    if preset == 'hyperliquid':
        return Config(HYPERLIQUID_CONFIG)
    elif preset == 'binance':
        return Config(BINANCE_CONFIG)
    else:
        return Config()
