"""Regime indicators for OmniAlpha."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .base import BaseIndicator


class VolumeZScore(BaseIndicator):
    """Volume Z-Score - measures volume relative to recent average."""

    def __init__(self, period: int = 20):
        """Initialize Volume Z-Score.

        Args:
            period: Lookback period (default: 20)
        """
        super().__init__(name='VolumeZScore', category='regime', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Z-Score.

        Args:
            data: OHLCV DataFrame

        Returns:
            Volume Z-Score values
        """
        self.validate_data(data)

        # Calculate rolling mean and std of volume
        volume_mean = data['volume'].rolling(window=self.period).mean()
        volume_std = data['volume'].rolling(window=self.period).std()

        # Calculate Z-Score
        z_score = (data['volume'] - volume_mean) / volume_std

        return z_score


class HurstExponent(BaseIndicator):
    """Hurst Exponent - measures trend persistence/mean reversion."""

    def __init__(self, period: int = 100, lags_range: tuple = (2, 20)):
        """Initialize Hurst Exponent.

        Args:
            period: Rolling window period (default: 100)
            lags_range: Min and max lags for R/S calculation (default: (2, 20))
        """
        super().__init__(
            name='HurstExponent',
            category='regime',
            params={'period': period, 'lags_range': lags_range}
        )
        self.period = period
        self.min_lag, self.max_lag = lags_range

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Hurst Exponent.

        Args:
            data: OHLCV DataFrame

        Returns:
            Hurst Exponent values (0-1, 0.5=random, <0.5=mean-reverting, >0.5=trending)
        """
        self.validate_data(data)

        prices = data['close'].values
        hurst_values = []

        for i in range(len(prices)):
            if i < self.period:
                hurst_values.append(np.nan)
                continue

            # Get window of prices
            window = prices[i-self.period:i]

            # Calculate log returns
            log_returns = np.log(window[1:] / window[:-1])

            # Calculate R/S statistic for different lags
            rs_values = []
            lags = range(self.min_lag, min(self.max_lag, len(log_returns) // 2))

            for lag in lags:
                # Split into subseries
                num_subseries = len(log_returns) // lag
                if num_subseries < 2:
                    continue

                rs_subseries = []
                for j in range(num_subseries):
                    subseries = log_returns[j*lag:(j+1)*lag]

                    # Calculate mean
                    mean = np.mean(subseries)

                    # Calculate cumulative deviate
                    cumdev = np.cumsum(subseries - mean)

                    # Calculate range
                    R = np.max(cumdev) - np.min(cumdev)

                    # Calculate standard deviation
                    S = np.std(subseries, ddof=1)

                    if S > 0:
                        rs_subseries.append(R / S)

                if rs_subseries:
                    rs_values.append((np.log(lag), np.log(np.mean(rs_subseries))))

            # Calculate Hurst exponent from slope of log(R/S) vs log(lag)
            if len(rs_values) > 1:
                x = np.array([val[0] for val in rs_values])
                y = np.array([val[1] for val in rs_values])

                # Linear regression
                slope, _ = np.polyfit(x, y, 1)
                hurst_values.append(slope)
            else:
                hurst_values.append(np.nan)

        return pd.Series(hurst_values, index=data.index)


class MarketRegime(BaseIndicator):
    """Market Regime Classifier - identifies trending, ranging, volatile regimes."""

    def __init__(self, period: int = 50):
        """Initialize Market Regime.

        Args:
            period: Lookback period (default: 50)
        """
        super().__init__(name='MarketRegime', category='regime', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Market Regime indicators.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with columns: trend_strength, volatility_regime, volume_regime
        """
        self.validate_data(data)

        # Calculate ADX for trend strength
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/14, adjust=False).mean()

        # Normalize trend strength (0-100)
        trend_strength = adx

        # Calculate volatility regime (high/low volatility)
        log_returns = np.log(close / close.shift(1))
        rolling_vol = log_returns.rolling(window=self.period).std() * np.sqrt(365) * 100

        # Normalize volatility (percentile rank)
        volatility_regime = rolling_vol.rolling(window=self.period).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        # Calculate volume regime (high/low volume)
        volume_ma = data['volume'].rolling(window=self.period).mean()
        volume_regime = (data['volume'] / volume_ma - 1) * 100

        return pd.DataFrame({
            'trend_strength': trend_strength,
            'volatility_regime': volatility_regime,
            'volume_regime': volume_regime
        })


class TrendStrength(BaseIndicator):
    """Trend Strength Indicator - combines multiple trend measures."""

    def __init__(self, period: int = 20):
        """Initialize Trend Strength.

        Args:
            period: Lookback period (default: 20)
        """
        super().__init__(name='TrendStrength', category='regime', params={'period': period})
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Trend Strength.

        Args:
            data: OHLCV DataFrame

        Returns:
            Trend strength score (-1 to 1, negative=downtrend, positive=uptrend)
        """
        self.validate_data(data)

        close = data['close']

        # Linear regression slope
        x = np.arange(self.period)

        def calculate_slope(window):
            if len(window) < self.period:
                return np.nan
            y = window.values
            slope, _ = np.polyfit(x, y, 1)
            # Normalize by price level
            return slope / np.mean(y)

        slope = close.rolling(window=self.period).apply(calculate_slope, raw=False)

        # R-squared (trend consistency)
        def calculate_r_squared(window):
            if len(window) < self.period:
                return np.nan
            y = window.values
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            return r_squared

        r_squared = close.rolling(window=self.period).apply(calculate_r_squared, raw=False)

        # Combine slope and R-squared
        trend_strength = slope * r_squared * 1000  # Scale for readability

        return trend_strength


class VolatilityRegime(BaseIndicator):
    """Volatility Regime Indicator - detects high/low volatility periods."""

    def __init__(self, short_period: int = 10, long_period: int = 50):
        """Initialize Volatility Regime.

        Args:
            short_period: Short-term volatility period (default: 10)
            long_period: Long-term volatility period (default: 50)
        """
        super().__init__(
            name='VolatilityRegime',
            category='regime',
            params={'short_period': short_period, 'long_period': long_period}
        )
        self.short_period = short_period
        self.long_period = long_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volatility Regime.

        Args:
            data: OHLCV DataFrame

        Returns:
            Volatility ratio (short-term vol / long-term vol)
            Values > 1 indicate rising volatility, < 1 indicate falling volatility
        """
        self.validate_data(data)

        # Calculate returns
        returns = data['close'].pct_change()

        # Calculate short-term and long-term volatility
        short_vol = returns.rolling(window=self.short_period).std()
        long_vol = returns.rolling(window=self.long_period).std()

        # Calculate ratio
        vol_ratio = short_vol / long_vol

        return vol_ratio
