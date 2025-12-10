"""Volatility indicators for OmniAlpha."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .base import BaseIndicator


class ATR(BaseIndicator):
    """Average True Range."""

    def __init__(self, period: int = 14):
        """Initialize ATR.

        Args:
            period: ATR period (default: 14)
        """
        super().__init__(name='ATR', category='volatility', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR.

        Args:
            data: OHLCV DataFrame

        Returns:
            ATR values
        """
        self.validate_data(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR using Wilder's smoothing
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()

        return atr


class BollingerBands(BaseIndicator):
    """Bollinger Bands."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """Initialize Bollinger Bands.

        Args:
            period: Period for moving average (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
        """
        super().__init__(
            name='BollingerBands',
            category='volatility',
            params={'period': period, 'std_dev': std_dev}
        )
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with columns: middle (SMA), upper, lower, bandwidth
        """
        self.validate_data(data)

        # Calculate middle band (SMA)
        middle = data['close'].rolling(window=self.period).mean()

        # Calculate standard deviation
        std = data['close'].rolling(window=self.period).std()

        # Calculate upper and lower bands
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        # Calculate bandwidth (normalized volatility measure)
        bandwidth = (upper - lower) / middle

        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'bandwidth': bandwidth
        })


class KeltnerChannels(BaseIndicator):
    """Keltner Channels."""

    def __init__(self, ema_period: int = 20, atr_period: int = 10, atr_multiplier: float = 2.0):
        """Initialize Keltner Channels.

        Args:
            ema_period: Period for EMA (default: 20)
            atr_period: Period for ATR (default: 10)
            atr_multiplier: Multiplier for ATR (default: 2.0)
        """
        super().__init__(
            name='KeltnerChannels',
            category='volatility',
            params={
                'ema_period': ema_period,
                'atr_period': atr_period,
                'atr_multiplier': atr_multiplier
            }
        )
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channels.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with columns: middle (EMA), upper, lower
        """
        self.validate_data(data)

        # Calculate middle line (EMA)
        middle = data['close'].ewm(span=self.ema_period, adjust=False).mean()

        # Calculate ATR
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/self.atr_period, adjust=False).mean()

        # Calculate upper and lower channels
        upper = middle + (atr * self.atr_multiplier)
        lower = middle - (atr * self.atr_multiplier)

        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower
        })


class HistoricalVolatility(BaseIndicator):
    """Historical Volatility (annualized)."""

    def __init__(self, period: int = 20):
        """Initialize Historical Volatility.

        Args:
            period: Lookback period (default: 20)
        """
        super().__init__(
            name='HistoricalVolatility',
            category='volatility',
            params={'period': period}
        )
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Historical Volatility.

        Args:
            data: OHLCV DataFrame

        Returns:
            Annualized volatility (%)
        """
        self.validate_data(data)

        # Calculate log returns
        log_returns = np.log(data['close'] / data['close'].shift(1))

        # Calculate rolling standard deviation
        rolling_std = log_returns.rolling(window=self.period).std()

        # Annualize volatility (assuming 365 days for crypto)
        annualized_vol = rolling_std * np.sqrt(365) * 100

        return annualized_vol


class DonchianChannels(BaseIndicator):
    """Donchian Channels."""

    def __init__(self, period: int = 20):
        """Initialize Donchian Channels.

        Args:
            period: Lookback period (default: 20)
        """
        super().__init__(
            name='DonchianChannels',
            category='volatility',
            params={'period': period}
        )
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channels.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with columns: upper, middle, lower
        """
        self.validate_data(data)

        # Calculate upper and lower channels
        upper = data['high'].rolling(window=self.period).max()
        lower = data['low'].rolling(window=self.period).min()

        # Calculate middle channel
        middle = (upper + lower) / 2

        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
