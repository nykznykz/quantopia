"""Abstract PriceFeed interface and implementations."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Union, Tuple
import pandas as pd
import numpy as np
import logging

from .models import OHLCV
from .exceptions import DataFeedError

logger = logging.getLogger(__name__)


class PriceFeed(ABC):
    """Abstract base class for price feeds."""

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current/latest price for symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')

        Returns:
            Current price

        Raises:
            DataFeedError: If symbol not found or no data available
        """
        pass

    @abstractmethod
    def get_current_volume(self, symbol: str) -> float:
        """Get current/latest volume for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current volume

        Raises:
            DataFeedError: If symbol not found or no data available
        """
        pass

    @abstractmethod
    def get_current_timestamp(self) -> datetime:
        """Get current simulation/real time.

        Returns:
            Current timestamp
        """
        pass

    @abstractmethod
    def get_ohlcv(self, symbol: str) -> OHLCV:
        """Get current OHLCV bar.

        Args:
            symbol: Trading symbol

        Returns:
            OHLCV object with current bar data

        Raises:
            DataFeedError: If symbol not found or no data available
        """
        pass


class HistoricalPriceFeed(PriceFeed):
    """Price feed for historical backtesting."""

    def __init__(
        self,
        data_source: Union[str, pd.DataFrame],
        symbols: List[str],
        timeframe: str = '1h'
    ):
        """Initialize historical price feed.

        Args:
            data_source: Path to CSV/Parquet file or DataFrame
            symbols: List of trading symbols
            timeframe: Data timeframe ('1m', '5m', '15m', '1h', '4h', '1d')

        Raises:
            DataFeedError: If data cannot be loaded or is invalid
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self._current_index = 0
        self._data = self._load_data(data_source)
        self._validate_data()

        logger.info(f"Loaded historical data: {len(self._data)} bars for {symbols}")

    def _load_data(self, data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load data from file or DataFrame."""
        try:
            if isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
            elif isinstance(data_source, str):
                if data_source.endswith('.parquet'):
                    df = pd.read_parquet(data_source)
                else:
                    df = pd.read_csv(data_source)
            else:
                raise DataFeedError(f"Invalid data source type: {type(data_source)}")

            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            else:
                raise DataFeedError("Data must have 'timestamp' column")

            return df

        except Exception as e:
            raise DataFeedError(f"Failed to load data: {str(e)}")

    def _validate_data(self):
        """Validate data quality."""
        required_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in self._data.columns]
        if missing:
            raise DataFeedError(f"Missing required columns: {missing}")

        # Check for valid prices
        if (self._data[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.warning("Found non-positive prices in data")

        # Check high >= low
        if (self._data['high'] < self._data['low']).any():
            raise DataFeedError("Found bars where high < low")

        # Check for symbols
        available_symbols = self._data['symbol'].unique()
        missing_symbols = [s for s in self.symbols if s not in available_symbols]
        if missing_symbols:
            raise DataFeedError(f"Symbols not found in data: {missing_symbols}")

    def get_current_price(self, symbol: str) -> float:
        """Get current close price for symbol."""
        if self._current_index >= len(self._data):
            raise DataFeedError("No more data available")

        current_data = self._data.iloc[self._current_index]
        if isinstance(current_data, pd.DataFrame):
            # Multi-symbol data
            symbol_data = current_data[current_data['symbol'] == symbol]
            if symbol_data.empty:
                raise DataFeedError(f"Symbol {symbol} not found in current data")
            return float(symbol_data.iloc[0]['close'])
        else:
            # Single bar, check if it's for the requested symbol
            if current_data['symbol'] != symbol:
                # Look for the symbol in current timestamp
                current_time = current_data['timestamp']
                symbol_data = self._data[
                    (self._data['timestamp'] == current_time) &
                    (self._data['symbol'] == symbol)
                ]
                if symbol_data.empty:
                    raise DataFeedError(f"Symbol {symbol} not found at current timestamp")
                return float(symbol_data.iloc[0]['close'])
            return float(current_data['close'])

    def get_current_volume(self, symbol: str) -> float:
        """Get current volume for symbol."""
        if self._current_index >= len(self._data):
            raise DataFeedError("No more data available")

        current_data = self._data.iloc[self._current_index]
        if isinstance(current_data, pd.DataFrame):
            symbol_data = current_data[current_data['symbol'] == symbol]
            if symbol_data.empty:
                raise DataFeedError(f"Symbol {symbol} not found in current data")
            return float(symbol_data.iloc[0]['volume'])
        else:
            if current_data['symbol'] != symbol:
                current_time = current_data['timestamp']
                symbol_data = self._data[
                    (self._data['timestamp'] == current_time) &
                    (self._data['symbol'] == symbol)
                ]
                if symbol_data.empty:
                    raise DataFeedError(f"Symbol {symbol} not found at current timestamp")
                return float(symbol_data.iloc[0]['volume'])
            return float(current_data['volume'])

    def get_current_timestamp(self) -> datetime:
        """Get current timestamp."""
        if self._current_index >= len(self._data):
            raise DataFeedError("No more data available")
        return pd.to_datetime(self._data.iloc[self._current_index]['timestamp'])

    def get_ohlcv(self, symbol: str) -> OHLCV:
        """Get current OHLCV bar for symbol."""
        if self._current_index >= len(self._data):
            raise DataFeedError("No more data available")

        current_data = self._data.iloc[self._current_index]

        # Handle multi-symbol or single-symbol data
        if isinstance(current_data, pd.DataFrame):
            symbol_data = current_data[current_data['symbol'] == symbol]
            if symbol_data.empty:
                raise DataFeedError(f"Symbol {symbol} not found in current data")
            row = symbol_data.iloc[0]
        else:
            if current_data['symbol'] != symbol:
                current_time = current_data['timestamp']
                symbol_data = self._data[
                    (self._data['timestamp'] == current_time) &
                    (self._data['symbol'] == symbol)
                ]
                if symbol_data.empty:
                    raise DataFeedError(f"Symbol {symbol} not found at current timestamp")
                row = symbol_data.iloc[0]
            else:
                row = current_data

        return OHLCV(
            timestamp=pd.to_datetime(row['timestamp']),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume'])
        )

    def next_bar(self) -> bool:
        """Advance to next time bar.

        Returns:
            True if successful, False if no more data
        """
        if self._current_index < len(self._data) - 1:
            self._current_index += 1
            return True
        return False

    def has_next(self) -> bool:
        """Check if more data is available."""
        return self._current_index < len(self._data) - 1

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Get start and end dates of available data."""
        return (
            pd.to_datetime(self._data['timestamp'].min()),
            pd.to_datetime(self._data['timestamp'].max())
        )

    def reset(self):
        """Reset to beginning of data."""
        self._current_index = 0
        logger.info("Historical feed reset to beginning")
