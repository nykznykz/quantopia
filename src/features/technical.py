"""Technical features derived from indicators."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from .base import BaseFeature, FeatureCategory, FeatureSet
from src.indicators import INDICATOR_REGISTRY

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Generate technical features from indicator library."""

    def __init__(self, indicator_configs: Optional[Dict] = None):
        """
        Initialize technical feature generator.

        Args:
            indicator_configs: Dict mapping indicator name to parameters
                Example: {'RSI': {'period': 14}, 'EMA': {'period': 20}}
                If None, uses default configurations
        """
        self.indicator_configs = indicator_configs or self._get_default_configs()

    def _get_default_configs(self) -> Dict:
        """Get default indicator configurations."""
        return {
            # Trend indicators
            'EMA': [{'period': 10}, {'period': 20}, {'period': 50}],
            'SMA': [{'period': 20}, {'period': 50}],
            'MACD': [{'fast_period': 12, 'slow_period': 26, 'signal_period': 9}],
            'ADX': [{'period': 14}],

            # Momentum indicators
            'RSI': [{'period': 14}],
            'Stochastic': [{'period': 14, 'smooth_k': 3, 'smooth_d': 3}],
            'MFI': [{'period': 14}],
            'ROC': [{'period': 12}],

            # Volatility indicators
            'ATR': [{'period': 14}],
            'BollingerBands': [{'period': 20, 'std_dev': 2}],
            'HistoricalVolatility': [{'period': 20}],
        }

    def compute_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with all technical features
        """
        logger.info("Computing technical features from indicators")

        features_df = pd.DataFrame(index=data.index)

        for indicator_name, config_list in self.indicator_configs.items():
            if indicator_name not in INDICATOR_REGISTRY:
                logger.warning(f"Indicator {indicator_name} not found in registry, skipping")
                continue

            indicator_class = INDICATOR_REGISTRY[indicator_name]

            for config in config_list:
                try:
                    # Create indicator instance
                    indicator = indicator_class(**config)

                    # Calculate indicator values
                    indicator_values = indicator.calculate(data)

                    # Add indicator values as features
                    if isinstance(indicator_values, dict):
                        # Multi-output indicator (e.g., MACD, Bollinger Bands)
                        for key, values in indicator_values.items():
                            feature_name = f"{indicator_name}_{key}_{self._config_to_str(config)}"
                            features_df[feature_name] = values
                    else:
                        # Single output indicator
                        feature_name = f"{indicator_name}_{self._config_to_str(config)}"
                        features_df[feature_name] = indicator_values

                    logger.debug(f"  ✓ Computed {indicator_name} with config {config}")

                except Exception as e:
                    logger.error(f"  ✗ Failed to compute {indicator_name} with config {config}: {e}")

        logger.info(f"Computed {len(features_df.columns)} technical features")
        return features_df

    def compute_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price-based features.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=data.index)

        # Returns
        features['return_1'] = data['close'].pct_change(1)
        features['return_5'] = data['close'].pct_change(5)
        features['return_10'] = data['close'].pct_change(10)
        features['return_20'] = data['close'].pct_change(20)

        # Log returns
        features['log_return_1'] = np.log(data['close'] / data['close'].shift(1))
        features['log_return_5'] = np.log(data['close'] / data['close'].shift(5))

        # Price momentum
        features['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1

        # High-Low range
        features['hl_range'] = (data['high'] - data['low']) / data['close']
        features['hl_range_ma5'] = features['hl_range'].rolling(5).mean()

        # Close position in range
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])

        # Gap features
        features['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)

        logger.info(f"Computed {len(features.columns)} price features")
        return features

    def compute_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume-based features.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)

        # Volume ratios
        features['volume_ratio_5'] = data['volume'] / data['volume'].rolling(5).mean()
        features['volume_ratio_20'] = data['volume'] / data['volume'].rolling(20).mean()

        # Volume momentum
        features['volume_momentum'] = data['volume'].pct_change(1)

        # Volume-price correlation
        features['volume_price_corr_10'] = (
            data['close'].rolling(10).corr(data['volume'])
        )

        # On-balance volume (OBV)
        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ma10'] = obv.rolling(10).mean()

        logger.info(f"Computed {len(features.columns)} volume features")
        return features

    def compute_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility-based features.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change()

        # Rolling volatility (different windows)
        features['volatility_5'] = returns.rolling(5).std()
        features['volatility_10'] = returns.rolling(10).std()
        features['volatility_20'] = returns.rolling(20).std()

        # Volatility of volatility
        features['vol_of_vol'] = features['volatility_20'].rolling(10).std()

        # Parkinson's volatility (high-low range)
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * ((np.log(data['high'] / data['low'])) ** 2)
        ).rolling(20).mean()

        # Garman-Klass volatility
        features['gk_vol'] = np.sqrt(
            0.5 * (np.log(data['high'] / data['low'])) ** 2 -
            (2 * np.log(2) - 1) * (np.log(data['close'] / data['open'])) ** 2
        ).rolling(20).mean()

        logger.info(f"Computed {len(features.columns)} volatility features")
        return features

    def compute_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time-based features.

        Args:
            data: OHLCV DataFrame with DatetimeIndex

        Returns:
            DataFrame with temporal features
        """
        features = pd.DataFrame(index=data.index)

        # Hour of day (for intraday data)
        if hasattr(data.index, 'hour'):
            features['hour'] = data.index.hour
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

        # Day of week
        if hasattr(data.index, 'dayofweek'):
            features['day_of_week'] = data.index.dayofweek
            features['is_weekend'] = (data.index.dayofweek >= 5).astype(int)

        # Day of month
        if hasattr(data.index, 'day'):
            features['day_of_month'] = data.index.day

        # Month
        if hasattr(data.index, 'month'):
            features['month'] = data.index.month

        logger.info(f"Computed {len(features.columns)} temporal features")
        return features

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all feature types.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with all features combined
        """
        logger.info("Computing all technical features")

        all_features = pd.DataFrame(index=data.index)

        # Compute each feature type
        feature_types = [
            ('technical', self.compute_all_features),
            ('price', self.compute_price_features),
            ('volume', self.compute_volume_features),
            ('volatility', self.compute_volatility_features),
            ('temporal', self.compute_temporal_features),
        ]

        for feature_type, compute_func in feature_types:
            try:
                features = compute_func(data)
                all_features = pd.concat([all_features, features], axis=1)
                logger.info(f"  ✓ Added {len(features.columns)} {feature_type} features")
            except Exception as e:
                logger.error(f"  ✗ Failed to compute {feature_type} features: {e}")

        # Drop rows with NaN values (from rolling windows)
        initial_rows = len(all_features)
        all_features = all_features.dropna()
        dropped_rows = initial_rows - len(all_features)

        logger.info(
            f"Total features computed: {len(all_features.columns)}, "
            f"Dropped {dropped_rows} rows with NaN"
        )

        return all_features

    def _config_to_str(self, config: Dict) -> str:
        """Convert config dict to string for feature naming."""
        return '_'.join([f"{k}{v}" for k, v in config.items()])
