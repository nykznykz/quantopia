"""Data downloader for fetching historical price data from exchanges."""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import logging
import time

from .exceptions import DataFeedError

logger = logging.getLogger(__name__)


class DataDownloader:
    """Download historical OHLCV data from cryptocurrency exchanges."""

    def __init__(self, exchange_name: str = 'binance'):
        """Initialize data downloader.

        Args:
            exchange_name: Name of exchange ('binance', 'hyperliquid', etc.)

        Raises:
            DataFeedError: If exchange is not supported
        """
        self.exchange_name = exchange_name.lower()
        self._init_exchange()

    def _init_exchange(self):
        """Initialize exchange connection."""
        try:
            if self.exchange_name == 'binance':
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
            elif self.exchange_name == 'hyperliquid':
                # Hyperliquid support through ccxt if available
                try:
                    self.exchange = ccxt.hyperliquid({
                        'enableRateLimit': True
                    })
                except Exception as e:
                    logger.warning(
                        f"Hyperliquid not available in ccxt: {e}. "
                        "Falling back to Binance for data."
                    )
                    self.exchange = ccxt.binance({'enableRateLimit': True})
                    self.exchange_name = 'binance'
            else:
                # Try to initialize any ccxt-supported exchange
                exchange_class = getattr(ccxt, self.exchange_name, None)
                if exchange_class is None:
                    raise DataFeedError(f"Exchange {self.exchange_name} not supported")
                self.exchange = exchange_class({'enableRateLimit': True})

            logger.info(f"Initialized {self.exchange_name} exchange for data download")

        except Exception as e:
            raise DataFeedError(f"Failed to initialize exchange: {str(e)}")

    def download(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Download historical OHLCV data.

        Args:
            symbols: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: now)
            limit: Maximum number of bars per symbol (default: None = all available)

        Returns:
            DataFrame with columns: timestamp, symbol, open, high, low, close, volume

        Raises:
            DataFeedError: If download fails
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        all_data = []

        for symbol in symbols:
            logger.info(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}")

            try:
                symbol_data = self._download_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit
                )

                if symbol_data:
                    all_data.extend(symbol_data)
                    logger.info(f"Downloaded {len(symbol_data)} bars for {symbol}")
                else:
                    logger.warning(f"No data downloaded for {symbol}")

            except Exception as e:
                logger.error(f"Failed to download {symbol}: {str(e)}")
                raise DataFeedError(f"Failed to download {symbol}: {str(e)}")

        if not all_data:
            raise DataFeedError("No data downloaded for any symbol")

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        logger.info(f"Downloaded total {len(df)} bars for {len(symbols)} symbols")
        return df

    def _download_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None
    ) -> List[dict]:
        """Download data for a single symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            limit: Max bars to fetch

        Returns:
            List of OHLCV dictionaries
        """
        all_candles = []
        since = int(start_date.timestamp() * 1000)  # Convert to milliseconds
        end_ts = int(end_date.timestamp() * 1000)

        # Normalize symbol format for different exchanges
        normalized_symbol = self._normalize_symbol(symbol)

        while True:
            try:
                # Fetch batch of candles
                fetch_limit = min(1000, limit - len(all_candles)) if limit else 1000
                candles = self.exchange.fetch_ohlcv(
                    normalized_symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=fetch_limit
                )

                if not candles:
                    break

                # Filter by end date
                candles = [c for c in candles if c[0] <= end_ts]

                if not candles:
                    break

                # Convert to dict format
                for candle in candles:
                    all_candles.append({
                        'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                        'symbol': self._format_symbol(symbol),  # Standardize to BTC-USD format
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })

                # Check if we should continue
                if limit and len(all_candles) >= limit:
                    break

                if candles[-1][0] >= end_ts:
                    break

                # Update 'since' to last candle timestamp + 1
                since = candles[-1][0] + 1

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                logger.error(f"Error fetching candles for {symbol}: {str(e)}")
                break

        return all_candles

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to exchange format.

        Args:
            symbol: Symbol in format 'BTC-USD' or 'BTC/USDT'

        Returns:
            Exchange-specific symbol format
        """
        # Convert BTC-USD to BTC/USDT for most exchanges
        if '-' in symbol:
            base, quote = symbol.split('-')
            # Convert USD to USDT for crypto exchanges (except some that use USD)
            if quote == 'USD':
                quote = 'USDT'
            return f"{base}/{quote}"
        return symbol

    def _format_symbol(self, symbol: str) -> str:
        """Format symbol to standard format (BTC-USD).

        Args:
            symbol: Symbol in any format

        Returns:
            Standardized symbol (e.g., 'BTC-USD')
        """
        if '/' in symbol:
            base, quote = symbol.split('/')
            # Standardize USDT back to USD
            if quote == 'USDT':
                quote = 'USD'
            return f"{base}-{quote}"
        return symbol

    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str
    ):
        """Save downloaded data to CSV file.

        Args:
            df: DataFrame with OHLCV data
            filename: Output filename
        """
        df.to_csv(filename, index=False)
        logger.info(f"Saved data to {filename}")

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        filename: str
    ):
        """Save downloaded data to Parquet file.

        Args:
            df: DataFrame with OHLCV data
            filename: Output filename
        """
        df.to_parquet(filename, index=False)
        logger.info(f"Saved data to {filename}")


def download_data(
    symbols: List[str],
    exchange: str = 'binance',
    timeframe: str = '1h',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """Convenience function to download data.

    Args:
        symbols: List of symbols (e.g., ['BTC-USD', 'ETH-USD'])
        exchange: Exchange name (default: 'binance')
        timeframe: Timeframe (default: '1h')
        start_date: Start date (default: 1 year ago)
        end_date: End date (default: now)
        output_file: Optional output file path (.csv or .parquet)

    Returns:
        DataFrame with downloaded data

    Example:
        >>> df = download_data(
        ...     symbols=['BTC-USD', 'ETH-USD'],
        ...     timeframe='1h',
        ...     start_date=datetime(2024, 1, 1),
        ...     output_file='data/crypto_data.csv'
        ... )
    """
    downloader = DataDownloader(exchange)
    df = downloader.download(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    if output_file:
        if output_file.endswith('.parquet'):
            downloader.save_to_parquet(df, output_file)
        else:
            downloader.save_to_csv(df, output_file)

    return df
