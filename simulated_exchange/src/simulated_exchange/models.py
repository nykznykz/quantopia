"""Data models for SimulatedExchange."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Optional[float] = None
    filled_size: float = 0.0
    avg_fill_price: float = 0.0

    def to_dict(self) -> dict:
        """Convert order to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'size': self.size,
            'limit_price': self.limit_price,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'filled_size': self.filled_size,
            'avg_fill_price': self.avg_fill_price
        }


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    size: float
    fill_price: float
    slippage: float
    fee: float
    timestamp: datetime
    pnl: float = 0.0  # Realized PnL for closing trades

    def to_dict(self) -> dict:
        """Convert trade to dictionary."""
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'size': self.size,
            'fill_price': self.fill_price,
            'slippage': self.slippage,
            'fee': self.fee,
            'timestamp': self.timestamp,
            'pnl': self.pnl
        }


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    size: float  # Positive for long, negative for short
    avg_entry_price: float
    entry_timestamp: datetime
    last_update: datetime

    def to_dict(self, current_price: float = None) -> dict:
        """Convert position to dictionary with optional current price for PnL."""
        result = {
            'symbol': self.symbol,
            'size': self.size,
            'avg_entry_price': self.avg_entry_price,
            'entry_timestamp': self.entry_timestamp,
            'last_update': self.last_update
        }

        if current_price is not None:
            unrealized_pnl = (current_price - self.avg_entry_price) * self.size
            unrealized_pnl_pct = ((current_price - self.avg_entry_price) / self.avg_entry_price) * 100
            result.update({
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct
            })

        return result


@dataclass
class EquityCurvePoint:
    """Represents a point in the equity curve."""
    timestamp: datetime
    portfolio_value: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


@dataclass
class OHLCV:
    """Represents an OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }
