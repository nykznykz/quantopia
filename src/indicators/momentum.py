"""Momentum indicators for OmniAlpha."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .base import BaseIndicator


class RSI(BaseIndicator):
    """Relative Strength Index."""

    def __init__(self, period: int = 14):
        """Initialize RSI.

        Args:
            period: RSI period (default: 14)
        """
        super().__init__(name='RSI', category='momentum', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI.

        Args:
            data: OHLCV DataFrame

        Returns:
            RSI values (0-100)
        """
        self.validate_data(data)

        # Calculate price changes
        delta = data['close'].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/self.period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/self.period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi


class Stochastic(BaseIndicator):
    """Stochastic Oscillator."""

    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
        """Initialize Stochastic.

        Args:
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D (signal line) calculation (default: 3)
            smooth_k: Smoothing period for %K (default: 3)
        """
        super().__init__(
            name='Stochastic',
            category='momentum',
            params={
                'k_period': k_period,
                'd_period': d_period,
                'smooth_k': smooth_k
            }
        )
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with columns: k (fast), d (slow/signal)
        """
        self.validate_data(data)

        # Calculate %K (fast stochastic)
        low_min = data['low'].rolling(window=self.k_period).min()
        high_max = data['high'].rolling(window=self.k_period).max()

        k_fast = 100 * (data['close'] - low_min) / (high_max - low_min)

        # Smooth %K
        k = k_fast.rolling(window=self.smooth_k).mean()

        # Calculate %D (signal line)
        d = k.rolling(window=self.d_period).mean()

        return pd.DataFrame({
            'k': k,
            'd': d
        })


class MFI(BaseIndicator):
    """Money Flow Index."""

    def __init__(self, period: int = 14):
        """Initialize MFI.

        Args:
            period: MFI period (default: 14)
        """
        super().__init__(name='MFI', category='momentum', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate MFI.

        Args:
            data: OHLCV DataFrame

        Returns:
            MFI values (0-100)
        """
        self.validate_data(data)

        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        # Calculate raw money flow
        money_flow = typical_price * data['volume']

        # Determine positive and negative money flow
        typical_price_diff = typical_price.diff()
        positive_flow = money_flow.where(typical_price_diff > 0, 0)
        negative_flow = money_flow.where(typical_price_diff < 0, 0)

        # Calculate money flow ratio
        positive_mf_sum = positive_flow.rolling(window=self.period).sum()
        negative_mf_sum = negative_flow.rolling(window=self.period).sum()

        money_ratio = positive_mf_sum / negative_mf_sum

        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi


class ROC(BaseIndicator):
    """Rate of Change."""

    def __init__(self, period: int = 12):
        """Initialize ROC.

        Args:
            period: ROC period (default: 12)
        """
        super().__init__(name='ROC', category='momentum', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Rate of Change.

        Args:
            data: OHLCV DataFrame

        Returns:
            ROC values (percentage)
        """
        self.validate_data(data)

        # Calculate ROC
        roc = ((data['close'] - data['close'].shift(self.period)) /
               data['close'].shift(self.period)) * 100

        return roc


class WilliamsR(BaseIndicator):
    """Williams %R."""

    def __init__(self, period: int = 14):
        """Initialize Williams %R.

        Args:
            period: Lookback period (default: 14)
        """
        super().__init__(name='WilliamsR', category='momentum', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R.

        Args:
            data: OHLCV DataFrame

        Returns:
            Williams %R values (-100 to 0)
        """
        self.validate_data(data)

        # Calculate highest high and lowest low
        highest_high = data['high'].rolling(window=self.period).max()
        lowest_low = data['low'].rolling(window=self.period).min()

        # Calculate Williams %R
        williams_r = -100 * (highest_high - data['close']) / (highest_high - lowest_low)

        return williams_r
