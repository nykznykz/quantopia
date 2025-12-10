"""Trend indicators for OmniAlpha."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .base import BaseIndicator


class EMA(BaseIndicator):
    """Exponential Moving Average."""

    def __init__(self, period: int = 20):
        """Initialize EMA.

        Args:
            period: EMA period (default: 20)
        """
        super().__init__(name='EMA', category='trend', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate EMA.

        Args:
            data: OHLCV DataFrame

        Returns:
            EMA values
        """
        self.validate_data(data)
        return data['close'].ewm(span=self.period, adjust=False).mean()


class SMA(BaseIndicator):
    """Simple Moving Average."""

    def __init__(self, period: int = 20):
        """Initialize SMA.

        Args:
            period: SMA period (default: 20)
        """
        super().__init__(name='SMA', category='trend', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate SMA.

        Args:
            data: OHLCV DataFrame

        Returns:
            SMA values
        """
        self.validate_data(data)
        return data['close'].rolling(window=self.period).mean()


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence."""

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Initialize MACD.

        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
        """
        super().__init__(
            name='MACD',
            category='trend',
            params={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period
            }
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with columns: macd, signal, histogram
        """
        self.validate_data(data)

        # Calculate fast and slow EMAs
        fast_ema = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=self.slow_period, adjust=False).mean()

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })


class ADX(BaseIndicator):
    """Average Directional Index."""

    def __init__(self, period: int = 14):
        """Initialize ADX.

        Args:
            period: ADX period (default: 14)
        """
        super().__init__(name='ADX', category='trend', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with columns: adx, plus_di, minus_di
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

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth with Wilder's smoothing
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/self.period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/self.period, adjust=False).mean() / atr

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/self.period, adjust=False).mean()

        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })


class EMASlope(BaseIndicator):
    """EMA Slope - measures the rate of change of EMA."""

    def __init__(self, ema_period: int = 20, slope_period: int = 5):
        """Initialize EMA Slope.

        Args:
            ema_period: EMA period (default: 20)
            slope_period: Period to calculate slope over (default: 5)
        """
        super().__init__(
            name='EMASlope',
            category='trend',
            params={'ema_period': ema_period, 'slope_period': slope_period}
        )
        self.ema_period = ema_period
        self.slope_period = slope_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate EMA slope.

        Args:
            data: OHLCV DataFrame

        Returns:
            EMA slope values (percentage change)
        """
        self.validate_data(data)

        # Calculate EMA
        ema = data['close'].ewm(span=self.ema_period, adjust=False).mean()

        # Calculate slope as percentage change
        slope = ema.pct_change(periods=self.slope_period) * 100

        return slope
