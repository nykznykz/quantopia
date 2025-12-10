"""SimulatedExchange - A simulated cryptocurrency exchange for backtesting and paper trading."""

__version__ = "0.1.0"

# Core classes
from .exchange import SimulatedExchange
from .price_feed import PriceFeed, HistoricalPriceFeed
from .live_feed import LivePriceFeed

# Models
from .models import (
    Order, Trade, Position, EquityCurvePoint, OHLCV,
    OrderSide, OrderType, OrderStatus
)

# Slippage and fees
from .slippage import (
    SlippageModel, FixedSlippageModel, VolumeBasedSlippageModel,
    HybridSlippageModel, NoSlippageModel, create_slippage_model
)
from .fees import (
    FeeModel, FlatFeeModel, TieredFeeModel, NoFeeModel, create_fee_model
)

# Data downloader
from .data_downloader import DataDownloader, download_data

# Configuration
from .config import Config, get_config, HYPERLIQUID_CONFIG, BINANCE_CONFIG

# Performance metrics
from .performance import calculate_performance_metrics, get_equity_curve_df, get_trade_history_df

# Exceptions
from .exceptions import (
    SimulatedExchangeError,
    InsufficientFundsError,
    InvalidOrderError,
    OrderNotFoundError,
    PositionNotFoundError,
    DataFeedError,
    ConnectionError
)

# Setup logging
import logging

def setup_logging(level=logging.INFO, log_file=None):
    """Setup logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

__all__ = [
    # Version
    '__version__',

    # Core
    'SimulatedExchange',
    'PriceFeed',
    'HistoricalPriceFeed',
    'LivePriceFeed',

    # Models
    'Order',
    'Trade',
    'Position',
    'EquityCurvePoint',
    'OHLCV',
    'OrderSide',
    'OrderType',
    'OrderStatus',

    # Slippage
    'SlippageModel',
    'FixedSlippageModel',
    'VolumeBasedSlippageModel',
    'HybridSlippageModel',
    'NoSlippageModel',
    'create_slippage_model',

    # Fees
    'FeeModel',
    'FlatFeeModel',
    'TieredFeeModel',
    'NoFeeModel',
    'create_fee_model',

    # Data
    'DataDownloader',
    'download_data',

    # Config
    'Config',
    'get_config',
    'HYPERLIQUID_CONFIG',
    'BINANCE_CONFIG',

    # Performance
    'calculate_performance_metrics',
    'get_equity_curve_df',
    'get_trade_history_df',

    # Exceptions
    'SimulatedExchangeError',
    'InsufficientFundsError',
    'InvalidOrderError',
    'OrderNotFoundError',
    'PositionNotFoundError',
    'DataFeedError',
    'ConnectionError',

    # Utilities
    'setup_logging',
]
