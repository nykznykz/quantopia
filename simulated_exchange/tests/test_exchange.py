"""Tests for SimulatedExchange core functionality."""

import pytest
import pandas as pd
from datetime import datetime
import sys
sys.path.insert(0, '../src')

from simulated_exchange import (
    SimulatedExchange,
    HistoricalPriceFeed,
    InsufficientFundsError,
    InvalidOrderError
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'symbol': ['BTC-USD'] * 100,
        'open': [40000 + i * 10 for i in range(100)],
        'high': [40100 + i * 10 for i in range(100)],
        'low': [39900 + i * 10 for i in range(100)],
        'close': [40000 + i * 10 for i in range(100)],
        'volume': [100] * 100
    }
    return pd.DataFrame(data)


@pytest.fixture
def exchange(sample_data):
    """Create exchange with sample data."""
    feed = HistoricalPriceFeed(sample_data, symbols=['BTC-USD'])
    exchange = SimulatedExchange(
        price_feed=feed,
        initial_capital=10000.0,
        mode='backtest'
    )
    return exchange


def test_initialization(exchange):
    """Test exchange initialization."""
    assert exchange.cash == 10000.0
    assert exchange.initial_capital == 10000.0
    assert len(exchange.positions) == 0
    assert len(exchange.open_orders) == 0
    assert len(exchange.trade_history) == 0


def test_market_buy_order(exchange):
    """Test placing a market buy order."""
    result = exchange.place_order(
        symbol='BTC-USD',
        side='buy',
        size=0.1,
        order_type='market'
    )

    assert result['status'] == 'filled'
    assert result['filled_size'] == 0.1
    assert result['avg_fill_price'] > 0
    assert result['fee'] >= 0

    # Check position was created
    position = exchange.get_position('BTC-USD')
    assert position is not None
    assert position['size'] == 0.1

    # Check cash was deducted
    assert exchange.cash < 10000.0


def test_market_sell_order(exchange):
    """Test placing a market sell order."""
    # First buy
    exchange.place_order('BTC-USD', 'buy', 0.1, 'market')

    # Then sell
    result = exchange.place_order('BTC-USD', 'sell', 0.1, 'market')

    assert result['status'] == 'filled'
    assert result['filled_size'] == 0.1

    # Position should be closed
    position = exchange.get_position('BTC-USD')
    assert position is None


def test_insufficient_funds(exchange):
    """Test that insufficient funds raises error."""
    with pytest.raises(InsufficientFundsError):
        exchange.place_order('BTC-USD', 'buy', 10.0, 'market')  # Way too large


def test_invalid_order_size(exchange):
    """Test that invalid order size raises error."""
    with pytest.raises(InvalidOrderError):
        exchange.place_order('BTC-USD', 'buy', -1.0, 'market')


def test_limit_order(exchange):
    """Test limit order placement."""
    current_price = exchange.price_feed.get_current_price('BTC-USD')

    # Place limit buy below current price
    result = exchange.place_order(
        symbol='BTC-USD',
        side='buy',
        size=0.1,
        order_type='limit',
        limit_price=current_price - 100
    )

    # Should be pending
    assert result['status'] == 'pending'

    # Check open orders
    open_orders = exchange.get_open_orders('BTC-USD')
    assert len(open_orders) == 1


def test_pnl_calculation(exchange):
    """Test PnL calculation."""
    # Buy at initial price
    exchange.place_order('BTC-USD', 'buy', 0.1, 'market')

    # Advance price feed
    exchange.price_feed.next_bar()
    exchange.update()

    pnl = exchange.get_pnl()
    # Since prices are increasing, unrealized PnL should be positive
    assert pnl['unrealized_pnl'] >= 0

    # Sell
    exchange.place_order('BTC-USD', 'sell', 0.1, 'market')

    pnl = exchange.get_pnl()
    # Should have realized PnL now
    assert 'realized_pnl' in pnl


def test_portfolio_value(exchange):
    """Test portfolio value calculation."""
    initial_value = exchange.get_portfolio_value()
    assert initial_value == 10000.0

    # Buy
    exchange.place_order('BTC-USD', 'buy', 0.1, 'market')

    # Portfolio value should still be close to initial (minus fees)
    portfolio_value = exchange.get_portfolio_value()
    assert portfolio_value < 10000.0
    assert portfolio_value > 9900.0  # Should lose a bit to fees


def test_reset(exchange):
    """Test exchange reset."""
    # Make some trades
    exchange.place_order('BTC-USD', 'buy', 0.1, 'market')

    # Reset
    exchange.reset()

    assert exchange.cash == 10000.0
    assert len(exchange.positions) == 0
    assert len(exchange.trade_history) == 0


def test_get_performance_metrics(exchange):
    """Test performance metrics calculation."""
    # Make a trade
    exchange.place_order('BTC-USD', 'buy', 0.1, 'market')
    exchange.price_feed.next_bar()
    exchange.update()
    exchange.place_order('BTC-USD', 'sell', 0.1, 'market')

    metrics = exchange.get_performance_metrics()

    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'total_trades' in metrics
    assert metrics['total_trades'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
