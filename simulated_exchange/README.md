# SimulatedExchange

A unified simulation and paper trading system for testing LLM-based trading agents in cryptocurrency markets. Supports both historical backtesting and live paper trading through a single, consistent interface.

## Features

- 🔄 **Unified Interface**: Same API for both backtesting and live paper trading
- 📊 **Historical Backtesting**: Test strategies on historical data with realistic execution
- 🔴 **Live Paper Trading**: Real-time simulation with live market data
- 💱 **Multi-Exchange Support**: Binance and Hyperliquid (extensible to others)
- 📈 **Performance Metrics**: Comprehensive analytics (Sharpe ratio, drawdown, win rate, etc.)
- 💰 **Realistic Execution**: Slippage and fee models for accurate simulation
- 📉 **Order Types**: Market and limit orders with proper execution logic
- 🎯 **Position Management**: Full position tracking with PnL calculations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Simple Backtest

```python
from simulated_exchange import (
    SimulatedExchange,
    HistoricalPriceFeed,
    download_data
)
from datetime import datetime

# Download historical data
df = download_data(
    symbols=['BTC-USD'],
    exchange='binance',
    timeframe='1h',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31)
)

# Create price feed
feed = HistoricalPriceFeed(df, symbols=['BTC-USD'])

# Create exchange
exchange = SimulatedExchange(
    price_feed=feed,
    initial_capital=10000.0
)

# Run backtest
while feed.has_next():
    # Your trading logic here
    current_price = feed.get_current_price('BTC-USD')

    # Example: buy if no position
    if not exchange.get_position('BTC-USD'):
        size = 1000 / current_price  # Invest $1000
        exchange.place_order('BTC-USD', 'buy', size, 'market')

    exchange.update()
    feed.next_bar()

# Get results
metrics = exchange.get_performance_metrics()
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### 2. Paper Trading

```python
from simulated_exchange import SimulatedExchange, LivePriceFeed
import time

# Create live price feed
feed = LivePriceFeed(
    exchange='binance',
    symbols=['BTC-USD'],
    testnet=True
)
feed.connect()

# Create exchange
exchange = SimulatedExchange(
    price_feed=feed,
    initial_capital=10000.0,
    mode='live'
)

# Trading loop
while True:
    current_price = feed.get_current_price('BTC-USD')

    # Your trading logic here
    # ... (use your LLM agent to make decisions)

    exchange.update()
    time.sleep(30)  # Check every 30 seconds
```

## Examples

The `examples/` directory contains several complete examples:

- **simple_backtest.py**: Basic buy-and-hold strategy
- **advanced_backtest.py**: Moving average crossover strategy
- **paper_trading.py**: Live paper trading example

Run them with:
```bash
cd examples
python simple_backtest.py
```

## Configuration

You can configure the exchange using dict-based or file-based config:

```python
from simulated_exchange import Config, SimulatedExchange

# Using dict
config = Config.from_dict({
    'exchange': {
        'initial_capital': 10000.0
    },
    'slippage': {
        'model': 'fixed',
        'fixed_bps': 5
    },
    'fees': {
        'model': 'tiered',
        'maker_fee': 0.0000,
        'taker_fee': 0.00025
    }
})

# Or load from file
config = Config.from_file('config.yaml')

# Use with exchange
exchange = SimulatedExchange(
    price_feed=feed,
    initial_capital=config.get('exchange.initial_capital'),
    slippage_config=config.get('slippage'),
    fee_config=config.get('fees')
)
```

## Key Components

### SimulatedExchange

The core exchange class that handles:
- Order placement and execution
- Position management
- PnL tracking
- Performance metrics

### PriceFeed

Abstract interface with two implementations:
- **HistoricalPriceFeed**: For backtesting with historical data
- **LivePriceFeed**: For paper trading with real-time data

### Slippage Models

- **Fixed**: Constant percentage slippage
- **Volume-based**: Slippage based on order size vs volume
- **Hybrid**: Combined base + volume + volatility (most realistic)

### Fee Models

- **Flat**: Same fee for all orders
- **Tiered**: Different fees for makers vs takers (recommended)

## Data Downloader

Download historical data from exchanges:

```python
from simulated_exchange import download_data
from datetime import datetime

df = download_data(
    symbols=['BTC-USD', 'ETH-USD'],
    exchange='binance',
    timeframe='1h',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    output_file='data/crypto_data.csv'
)
```

## Performance Metrics

The system calculates comprehensive metrics:
- Total return (% and USD)
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- Average win/loss
- Total trades and fees
- Average holding period

Access with:
```python
metrics = exchange.get_performance_metrics()
equity_curve = exchange.get_equity_curve()
trades = exchange.get_trade_history()
```

## Testing

Run tests with pytest:
```bash
pytest tests/ -v
```

## Project Structure

```
simulated_exchange/
├── src/simulated_exchange/
│   ├── __init__.py           # Main exports
│   ├── exchange.py           # SimulatedExchange class
│   ├── price_feed.py         # Price feed implementations
│   ├── live_feed.py          # Live price feeds
│   ├── slippage.py           # Slippage models
│   ├── fees.py               # Fee models
│   ├── performance.py        # Performance metrics
│   ├── data_downloader.py    # Data download utilities
│   ├── config.py             # Configuration system
│   ├── models.py             # Data models
│   └── exceptions.py         # Custom exceptions
├── examples/                 # Example scripts
├── tests/                    # Test suite
├── data/                     # Data files
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.9+
- pandas >= 2.0.0
- numpy >= 1.24.0
- ccxt >= 4.0.0 (for exchange connectivity)
- websocket-client >= 1.6.0 (for live feeds)
- pyyaml >= 6.0 (for config files)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Roadmap

Future enhancements:
- [ ] Short selling support
- [ ] Advanced order types (stop-loss, trailing stop, etc.)
- [ ] Multi-currency support
- [ ] Order book simulation
- [ ] Risk attribution analysis
- [ ] Web dashboard
- [ ] Database persistence

## Support

For issues and questions:
- Create an issue on GitHub
- Check the examples for usage patterns
- Review the requirements document for detailed specifications
