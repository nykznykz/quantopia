# SimulatedExchange Requirements Document

## Project Overview

A unified simulation and paper trading system for testing LLM-based trading agents in cryptocurrency markets. The system supports both historical backtesting and live paper trading through a single, consistent interface.

---

## 1. System Architecture

### 1.1 High-Level Components

```
┌─────────────────────────────────────────────────────────┐
│                     LLM Trading Agent                    │
│              (Decision-making & Strategy)                │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ Trading Signals
                         │
┌────────────────────────▼────────────────────────────────┐
│                  SimulatedExchange                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Order Management                                │   │
│  │  - place_order()                                 │   │
│  │  - cancel_order()                                │   │
│  │  - get_open_orders()                             │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Position Management                             │   │
│  │  - Track open positions                          │   │
│  │  - Calculate unrealized PnL                      │   │
│  │  - Manage cash balance                           │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Execution Engine                                │   │
│  │  - Slippage modeling                             │   │
│  │  - Fee calculation                               │   │
│  │  - Order matching logic                          │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Performance Tracking                            │   │
│  │  - Trade history                                 │   │
│  │  - Equity curve                                  │   │
│  │  - Performance metrics                           │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ Price Data
                         │
┌────────────────────────▼────────────────────────────────┐
│                  PriceFeed (Abstract)                    │
│  ┌──────────────────────┐  ┌──────────────────────┐    │
│  │  HistoricalPriceFeed │  │   LivePriceFeed      │    │
│  │  (Backtesting)       │  │  (Paper Trading)     │    │
│  │  - CSV/Parquet       │  │  - WebSocket         │    │
│  │  - Bar-by-bar        │  │  - Real-time         │    │
│  └──────────────────────┘  └──────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 SimulatedExchange

**Purpose**: Central hub for order execution, position management, and performance tracking.

#### 2.1.1 Initialization Parameters

```python
SimulatedExchange(
    price_feed: PriceFeed,
    initial_capital: float = 10000.0,
    mode: str = 'backtest',  # 'backtest' or 'live'
    slippage_config: dict = None,
    fee_config: dict = None
)
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `price_feed` | PriceFeed | Yes | - | Historical or live price feed |
| `initial_capital` | float | No | 10000.0 | Starting capital in USD |
| `mode` | str | No | 'backtest' | Operating mode: 'backtest' or 'live' |
| `slippage_config` | dict | No | See 2.3 | Slippage model configuration |
| `fee_config` | dict | No | See 2.4 | Fee structure configuration |

#### 2.1.2 State Variables

The exchange must maintain the following state:

```python
{
    'cash': float,                          # Available cash balance
    'positions': {                          # Open positions
        'BTC-USD': {
            'size': float,                  # Position size (+ long, - short)
            'avg_entry_price': float,       # Average entry price
            'entry_timestamp': datetime,    # First entry timestamp
            'last_update': datetime         # Last position update
        }
    },
    'open_orders': [                        # Active orders not yet filled
        {
            'order_id': str,
            'symbol': str,
            'side': str,                    # 'buy' or 'sell'
            'order_type': str,              # 'market' or 'limit'
            'size': float,
            'limit_price': float,           # Only for limit orders
            'timestamp': datetime,
            'status': str                   # 'pending', 'partial', 'filled', 'cancelled'
        }
    ],
    'trade_history': [                      # All completed trades
        {
            'trade_id': str,
            'order_id': str,
            'symbol': str,
            'side': str,
            'size': float,
            'fill_price': float,
            'slippage': float,              # Actual slippage applied
            'fee': float,                   # Fee charged
            'timestamp': datetime,
            'pnl': float                    # Realized PnL (for closing trades)
        }
    ],
    'equity_curve': [                       # Time-series of portfolio value
        {
            'timestamp': datetime,
            'portfolio_value': float,
            'cash': float,
            'unrealized_pnl': float,
            'realized_pnl': float
        }
    ]
}
```

#### 2.1.3 Public Methods

##### Order Management

```python
def place_order(
    self,
    symbol: str,
    side: str,              # 'buy' or 'sell'
    size: float,
    order_type: str = 'market',  # 'market' or 'limit'
    limit_price: float = None
) -> dict:
    """
    Place a new order.
    
    Returns:
        {
            'order_id': str,
            'status': str,      # 'filled', 'partial', 'pending', 'rejected'
            'filled_size': float,
            'avg_fill_price': float,
            'fee': float,
            'message': str      # Error or status message
        }
    
    Raises:
        InsufficientFundsError: If not enough cash/position to execute
        InvalidOrderError: If order parameters are invalid
    """
```

```python
def cancel_order(self, order_id: str) -> bool:
    """
    Cancel a pending order.
    
    Returns:
        True if successfully cancelled, False if order not found or already filled
    """
```

```python
def get_open_orders(self, symbol: str = None) -> List[dict]:
    """
    Get all open orders, optionally filtered by symbol.
    
    Returns:
        List of open order dictionaries
    """
```

##### Position Management

```python
def get_position(self, symbol: str) -> dict:
    """
    Get current position for a symbol.
    
    Returns:
        {
            'symbol': str,
            'size': float,
            'avg_entry_price': float,
            'current_price': float,
            'unrealized_pnl': float,
            'unrealized_pnl_pct': float
        }
        Returns None if no position
    """
```

```python
def get_all_positions(self) -> dict:
    """
    Get all open positions.
    
    Returns:
        Dictionary mapping symbol to position info
    """
```

```python
def get_portfolio_value(self) -> float:
    """
    Get current total portfolio value (cash + position values).
    
    Returns:
        Total portfolio value in USD
    """
```

##### Performance Metrics

```python
def get_pnl(self) -> dict:
    """
    Get profit and loss summary.
    
    Returns:
        {
            'realized_pnl': float,      # From closed trades
            'unrealized_pnl': float,    # From open positions
            'total_pnl': float,         # Sum of realized + unrealized
            'total_pnl_pct': float,     # Percentage return
            'total_fees': float         # All fees paid
        }
    """
```

```python
def get_performance_metrics(self) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Returns:
        {
            'total_return': float,           # Total return %
            'total_return_usd': float,       # Total return in USD
            'sharpe_ratio': float,           # Risk-adjusted return
            'max_drawdown': float,           # Maximum drawdown %
            'max_drawdown_duration': int,    # Days in drawdown
            'win_rate': float,               # % of winning trades
            'avg_win': float,                # Average winning trade
            'avg_loss': float,               # Average losing trade
            'profit_factor': float,          # Gross profit / gross loss
            'total_trades': int,             # Number of trades
            'total_fees': float,             # Total fees paid
            'avg_holding_period': float      # Average hours per trade
        }
    """
```

##### Data Access

```python
def get_equity_curve(self) -> pd.DataFrame:
    """
    Get equity curve as pandas DataFrame.
    
    Returns:
        DataFrame with columns: timestamp, portfolio_value, cash, 
                               unrealized_pnl, realized_pnl
    """
```

```python
def get_trade_history(self) -> pd.DataFrame:
    """
    Get all trades as pandas DataFrame.
    
    Returns:
        DataFrame with all trade details
    """
```

##### Utility Methods

```python
def reset(self):
    """
    Reset the exchange to initial state.
    Useful for running multiple backtests.
    """
```

```python
def get_account_info(self) -> dict:
    """
    Get current account summary.
    
    Returns:
        {
            'cash': float,
            'portfolio_value': float,
            'buying_power': float,
            'open_positions': int,
            'open_orders': int
        }
    """
```

---

### 2.2 PriceFeed (Abstract Base Class)

**Purpose**: Provide a unified interface for both historical and live price data.

#### 2.2.1 Abstract Methods

```python
class PriceFeed(ABC):
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current/latest price for symbol"""
        pass
    
    @abstractmethod
    def get_current_volume(self, symbol: str) -> float:
        """Get current/latest volume for symbol"""
        pass
    
    @abstractmethod
    def get_current_timestamp(self) -> datetime:
        """Get current simulation/real time"""
        pass
    
    @abstractmethod
    def get_ohlcv(self, symbol: str) -> dict:
        """
        Get current OHLCV bar.
        
        Returns:
            {
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float,
                'timestamp': datetime
            }
        """
        pass
```

---

### 2.3 Slippage Model

**Purpose**: Realistically simulate the difference between expected and actual execution prices.

#### 2.3.1 Configuration

```python
slippage_config = {
    'model': 'hybrid',  # 'fixed', 'volume_based', 'hybrid', 'none'
    
    # Fixed model parameters
    'fixed_bps': 5,  # 5 basis points = 0.05%
    
    # Volume-based model parameters
    'volume_limit': 0.10,      # Max 10% of bar volume
    'price_impact': 0.1,       # 0.1 = 10 bps per 1% of volume
    
    # Hybrid model parameters (combines both)
    'base_bps': 3,             # Base slippage
    'volume_limit': 0.10,
    'price_impact': 0.1,
    'use_volatility': True,    # Adjust for volatility
    'volatility_lookback': 20  # Bars for volatility calculation
}
```

#### 2.3.2 Slippage Models

##### Fixed Slippage
- **Simple**: Apply constant percentage slippage
- **Use case**: Quick prototyping, conservative estimates
- **Formula**: 
  - Buy: `fill_price = price × (1 + fixed_bps / 10000)`
  - Sell: `fill_price = price × (1 - fixed_bps / 10000)`

##### Volume-Based Slippage
- **Realistic**: Larger orders have more market impact
- **Use case**: More accurate simulation
- **Formula**:
  - `volume_share = order_size / bar_volume`
  - `impact = volume_share × price_impact`
  - `fill_price = price × (1 ± impact)`
- **Volume limit**: Orders exceeding volume limit result in partial fills

##### Hybrid Slippage (Recommended)
- **Most realistic**: Combines base slippage + market impact + volatility
- **Components**:
  1. Base slippage (always present)
  2. Market impact (based on order size vs volume)
  3. Volatility multiplier (higher vol = more slippage)
- **Formula**:
  ```python
  base = base_bps / 10000
  market_impact = (order_size / volume) × price_impact
  volatility_mult = 1 + (recent_volatility × 10)
  total_slippage = (base + market_impact) × volatility_mult
  ```

#### 2.3.3 Slippage Constraints

- **Price bounds**: For backtesting, slipped prices should not exceed bar high/low (optional, configurable)
- **Partial fills**: If order size exceeds volume limit, fill partially
- **Order rejection**: Market orders that would slip beyond threshold can be rejected (optional)

---

### 2.4 Fee Model

**Purpose**: Simulate realistic trading fees.

#### 2.4.1 Configuration

```python
fee_config = {
    'maker_fee': 0.0002,   # 0.02% (limit orders that add liquidity)
    'taker_fee': 0.0005,   # 0.05% (market orders that take liquidity)
    'model': 'tiered'      # 'flat', 'tiered', 'none'
}
```

#### 2.4.2 Fee Models

##### Flat Fee
- **Simple**: Same fee for all orders
- **Formula**: `fee = trade_value × fee_rate`

##### Tiered Fee (Recommended)
- **Realistic**: Different fees for makers vs takers
- Market orders = taker fee (immediate execution)
- Limit orders = maker fee (add liquidity)
- **Formula**:
  ```python
  if order_type == 'market':
      fee = trade_value × taker_fee
  else:  # limit order
      fee = trade_value × maker_fee
  ```

##### Typical Crypto Exchange Fees
| Exchange | Maker Fee | Taker Fee |
|----------|-----------|-----------|
| Hyperliquid | 0.00% | 0.025% |
| Binance | 0.02% | 0.04% |
| Coinbase Pro | 0.40% | 0.60% |
| Kraken | 0.16% | 0.26% |

**Default recommendation**: Use Hyperliquid's fee structure (0% maker, 0.025% taker)

---

### 2.5 HistoricalPriceFeed

**Purpose**: Provide historical price data for backtesting.

#### 2.5.1 Initialization

```python
HistoricalPriceFeed(
    data_source: Union[str, pd.DataFrame],  # File path or DataFrame
    symbols: List[str],
    timeframe: str = '1h'  # '1m', '5m', '15m', '1h', '4h', '1d'
)
```

#### 2.5.2 Data Format

Expected DataFrame structure:

```python
# Multi-symbol format
columns = [
    'timestamp',
    'symbol',
    'open',
    'high', 
    'low',
    'close',
    'volume'
]

# OR single-symbol format with MultiIndex
index = pd.DatetimeIndex  # timestamp
columns = pd.MultiIndex.from_product([
    ['BTC-USD', 'ETH-USD'],  # symbols
    ['open', 'high', 'low', 'close', 'volume']  # fields
])
```

#### 2.5.3 Methods

```python
def next_bar(self) -> bool:
    """
    Advance to next time bar.
    
    Returns:
        True if successful, False if no more data
    """
```

```python
def has_next(self) -> bool:
    """Check if more data is available"""
```

```python
def get_date_range(self) -> Tuple[datetime, datetime]:
    """Get start and end dates of available data"""
```

---

### 2.6 LivePriceFeed

**Purpose**: Provide real-time price data for paper trading.

#### 2.6.1 Initialization

```python
LivePriceFeed(
    exchange: str,      # 'hyperliquid', 'binance', etc.
    symbols: List[str],
    api_key: str = None,
    api_secret: str = None,
    testnet: bool = True
)
```

#### 2.6.2 Connection Management

```python
def connect(self):
    """Establish WebSocket connection"""
```

```python
def disconnect(self):
    """Close WebSocket connection"""
```

```python
def is_connected(self) -> bool:
    """Check connection status"""
```

#### 2.6.3 Data Updates

- **WebSocket subscriptions**: Subscribe to real-time price updates
- **Automatic reconnection**: Handle disconnections gracefully
- **Rate limiting**: Respect exchange API rate limits
- **Latency tracking**: Optional latency measurement

---

## 3. Functional Requirements

### 3.1 Order Execution

#### 3.1.1 Market Orders
- **Immediate execution**: Fill at current market price + slippage
- **Volume checks**: Respect volume limits
- **Balance validation**: Ensure sufficient funds before execution
- **Partial fills**: Support partial fills if volume insufficient

#### 3.1.2 Limit Orders
- **Pending state**: Queue until limit price reached
- **Fill logic**: 
  - Buy limit: Fill when market price ≤ limit price
  - Sell limit: Fill when market price ≥ limit price
- **Slippage**: Even limit orders experience slippage in execution
- **Time in force**: Support GTC (Good Till Cancelled) initially
- **Cancellation**: Allow cancellation before fill

#### 3.1.3 Order Validation

Before accepting an order, validate:
- ✅ Symbol is supported
- ✅ Order size > 0
- ✅ Sufficient cash for buys
- ✅ Sufficient position for sells
- ✅ Limit price is valid (if applicable)

#### 3.1.4 Position Management

- **Long positions**: Track size, avg entry, unrealized PnL
- **Short positions**: Support optional (start with long-only)
- **Position updates**: Recalculate on each trade and price update
- **Position closing**: Full or partial position exits

### 3.2 Performance Tracking

#### 3.2.1 Real-time Metrics

Update on every trade and price update:
- Portfolio value
- Unrealized PnL
- Cash balance
- Equity curve point

#### 3.2.2 Historical Metrics

Calculate after backtest/trading session:
- Total return (%)
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- Average holding period

#### 3.2.3 Equity Curve

- **Frequency**: Record portfolio value at regular intervals
- **Backtesting**: Record at each bar
- **Live trading**: Record every minute or on significant changes
- **Storage**: Efficient time-series storage

### 3.3 Risk Management

#### 3.3.1 Position Limits (Optional for MVP)

```python
risk_config = {
    'max_position_size': 0.20,  # Max 20% of portfolio per position
    'max_leverage': 1.0,         # No leverage initially
    'max_open_positions': 5      # Max concurrent positions
}
```

#### 3.3.2 Validation

- Reject orders that would exceed position limits
- Warn on high concentration
- Track exposure by symbol

---

## 4. Non-Functional Requirements

### 4.1 Performance

#### 4.1.1 Backtesting Speed
- **Target**: Process 1 year of hourly data in < 5 seconds
- **Optimization**: Vectorize calculations where possible
- **Memory**: Handle datasets up to 10M bars

#### 4.1.2 Live Trading Latency
- **Order placement**: < 100ms from signal to execution
- **Price updates**: Process WebSocket updates within 50ms
- **Metrics calculation**: < 10ms for portfolio updates

### 4.2 Reliability

#### 4.2.1 Data Integrity
- **State consistency**: Atomic updates for trades/positions
- **Validation**: Input validation on all public methods
- **Error handling**: Graceful handling of edge cases

#### 4.2.2 Fault Tolerance
- **WebSocket**: Auto-reconnect on disconnection
- **Data gaps**: Handle missing data gracefully
- **Logging**: Comprehensive logging for debugging

### 4.3 Extensibility

#### 4.3.1 Pluggable Components
- Easy to swap slippage models
- Easy to add new fee structures
- Easy to add new price feed sources
- Easy to add new performance metrics

#### 4.3.2 Configuration
- All parameters configurable via dict/config file
- Sensible defaults provided
- Validation of configuration

### 4.4 Usability

#### 4.4.1 API Design
- **Intuitive**: Methods named clearly
- **Consistent**: Similar patterns across components
- **Documented**: Docstrings for all public methods
- **Type hints**: Full type annotations

#### 4.4.2 Error Messages
- Clear, actionable error messages
- Include context (symbol, order details, etc.)
- Suggest fixes when possible

---

## 5. Data Requirements

### 5.1 Historical Data

#### 5.1.1 Minimum Requirements
- **Symbols**: BTC-USD, ETH-USD (expand later)
- **Timeframes**: 1h (start), then 5m, 15m, 1d
- **Fields**: timestamp, open, high, low, close, volume
- **Coverage**: At least 1 year of data
- **Format**: CSV or Parquet

#### 5.1.2 Data Sources
- Hyperliquid historical API
- Binance historical data
- CryptoDataDownload
- Custom CSV upload

#### 5.1.3 Data Quality
- No gaps in timestamps
- Volume > 0 (or handle gracefully)
- Prices > 0
- High ≥ Low
- High ≥ Open, Close
- Low ≤ Open, Close

### 5.2 Live Data

#### 5.2.1 WebSocket Endpoints
- Hyperliquid: `wss://api.hyperliquid.xyz/ws`
- Binance: `wss://stream.binance.com:9443/ws`
- Format: Exchange-specific, needs parsing

#### 5.2.2 Subscription
- Subscribe to ticker/trade streams
- Handle subscription acknowledgments
- Graceful handling of subscription failures

---

## 6. Testing Requirements

### 6.1 Unit Tests

Test each component in isolation:
- ✅ Order placement logic
- ✅ Slippage calculations
- ✅ Fee calculations
- ✅ Position tracking
- ✅ PnL calculations
- ✅ Performance metrics
- ✅ Price feed parsing

**Target coverage**: 80%+

### 6.2 Integration Tests

Test component interactions:
- ✅ Full order lifecycle (place → fill → position update)
- ✅ Historical backtest end-to-end
- ✅ Live feed connection and data flow
- ✅ Multiple concurrent orders
- ✅ Position opening and closing

### 6.3 Validation Tests

Verify realistic behavior:
- ✅ Compare backtest results with known strategies
- ✅ Verify slippage is directional (always against trader)
- ✅ Verify fees are always deducted
- ✅ Verify PnL calculations match manual calculation
- ✅ Test edge cases (zero volume, extreme prices, etc.)

### 6.4 Performance Tests

Benchmark critical paths:
- ✅ Backtest 1 year hourly data
- ✅ Handle 1000 orders
- ✅ Process 10,000 price updates
- ✅ Calculate metrics for 1000 trades

---

## 7. Implementation Phases

### Phase 1: Core MVP (Week 1-2)
**Goal**: Basic backtesting with market orders

- [ ] SimulatedExchange core class
- [ ] Basic state management (cash, positions, trades)
- [ ] Market order execution
- [ ] Fixed slippage model
- [ ] Flat fee model
- [ ] HistoricalPriceFeed implementation
- [ ] Basic performance metrics (PnL, return, trade count)
- [ ] Unit tests for core logic

**Deliverable**: Can backtest simple buy-and-hold strategy

### Phase 2: Enhanced Backtesting (Week 3)
**Goal**: Production-quality backtesting

- [ ] Limit orders
- [ ] Hybrid slippage model
- [ ] Tiered fee model
- [ ] Advanced performance metrics (Sharpe, drawdown, win rate)
- [ ] Equity curve tracking
- [ ] Trade history export
- [ ] Integration tests
- [ ] Example backtest scripts

**Deliverable**: Can backtest complex multi-asset strategies

### Phase 3: Live Paper Trading (Week 4)
**Goal**: Real-time simulation

- [ ] LivePriceFeed implementation
- [ ] WebSocket connection management
- [ ] Real-time order processing
- [ ] Latency simulation
- [ ] Live performance dashboard (optional)
- [ ] Paper trading example

**Deliverable**: Can paper trade in real-time

### Phase 4: Polish & Optimization (Week 5)
**Goal**: Production-ready

- [ ] Performance optimization
- [ ] Comprehensive documentation
- [ ] Example notebooks/tutorials
- [ ] Configuration file support
- [ ] Logging and monitoring
- [ ] Error recovery mechanisms

**Deliverable**: Production-ready system

---

## 8. Success Criteria

### 8.1 Functional
- ✅ Successfully backtests 1 year of data for 2+ assets
- ✅ Produces consistent PnL calculations
- ✅ Handles both market and limit orders
- ✅ Connects to live price feeds without errors
- ✅ Tracks performance metrics accurately

### 8.2 Technical
- ✅ 80%+ test coverage
- ✅ < 5 second backtest for 1 year hourly data
- ✅ < 100ms order execution latency
- ✅ Zero memory leaks in 24h live test

### 8.3 Usability
- ✅ LLM agent can place orders in < 10 lines of code
- ✅ Can switch between backtest and live with config change
- ✅ Clear error messages for common mistakes
- ✅ Documentation covers all major use cases

---

## 9. Out of Scope (Future Enhancements)

These features are explicitly **not** included in the initial version:

- ❌ Real money trading (live execution on actual exchanges)
- ❌ Short selling / margin trading
- ❌ Options / futures / derivatives
- ❌ Advanced order types (stop-loss, OCO, trailing stop)
- ❌ Multi-currency support (everything in USD)
- ❌ Order book simulation (depth, bid-ask spread)
- ❌ Transaction cost analysis (TCA)
- ❌ Risk attribution analysis
- ❌ Portfolio optimization
- ❌ Web UI / dashboard
- ❌ Database persistence
- ❌ Multi-user support
- ❌ Cloud deployment

---

## 10. Dependencies

### 10.1 Core Libraries
```
pandas >= 2.0.0          # Data manipulation
numpy >= 1.24.0          # Numerical computations
python >= 3.9            # Language version
```

### 10.2 Optional Libraries
```
websocket-client >= 1.6.0    # WebSocket for live feeds
requests >= 2.31.0           # HTTP for REST APIs
matplotlib >= 3.7.0          # Visualization (optional)
pyarrow >= 12.0.0           # Parquet support (optional)
pytest >= 7.4.0             # Testing
```

### 10.3 Exchange-Specific SDKs
```
ccxt >= 4.0.0               # Unified crypto exchange API (optional)
hyperliquid-python-sdk      # Hyperliquid SDK (if available)
```

---

## 11. Configuration Example

```python
# config.py
from datetime import datetime

CONFIG = {
    # Exchange configuration
    'exchange': {
        'initial_capital': 10000.0,
        'mode': 'backtest',  # 'backtest' or 'live'
    },
    
    # Slippage configuration
    'slippage': {
        'model': 'hybrid',
        'base_bps': 3,
        'volume_limit': 0.10,
        'price_impact': 0.1,
        'use_volatility': True,
        'volatility_lookback': 20
    },
    
    # Fee configuration
    'fees': {
        'model': 'tiered',
        'maker_fee': 0.0000,  # Hyperliquid: 0%
        'taker_fee': 0.00025  # Hyperliquid: 0.025%
    },
    
    # Backtest configuration
    'backtest': {
        'data_source': 'data/crypto_ohlcv.csv',
        'symbols': ['BTC-USD', 'ETH-USD'],
        'start_date': datetime(2024, 1, 1),
        'end_date': datetime(2024, 12, 31),
        'timeframe': '1h'
    },
    
    # Live trading configuration
    'live': {
        'exchange': 'hyperliquid',
        'testnet': True,
        'symbols': ['BTC-USD'],
        'ws_url': 'wss://api.hyperliquid-testnet.xyz/ws'
    }
}
```

---

## 12. Usage Examples

### Example 1: Simple Backtest

```python
from simulated_exchange import SimulatedExchange, HistoricalPriceFeed

# Setup
feed = HistoricalPriceFeed('data/btc_2024.csv', symbols=['BTC-USD'])
exchange = SimulatedExchange(feed, initial_capital=10000)

# Run backtest
while feed.has_next():
    # Your LLM agent makes decision
    if should_buy():
        exchange.place_order('BTC-USD', 'buy', size=0.1)
    elif should_sell():
        position = exchange.get_position('BTC-USD')
        if position:
            exchange.place_order('BTC-USD', 'sell', size=position['size'])
    
    feed.next_bar()

# Results
metrics = exchange.get_performance_metrics()
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
```

### Example 2: Paper Trading

```python
from simulated_exchange import SimulatedExchange, LivePriceFeed

# Setup
feed = LivePriceFeed('hyperliquid', symbols=['BTC-USD'], testnet=True)
exchange = SimulatedExchange(feed, initial_capital=10000, mode='live')

# Connect
feed.connect()

# Trading loop (runs continuously)
while True:
    # Your LLM agent makes decision based on current price
    current_price = feed.get_current_price('BTC-USD')
    
    if llm_says_buy(current_price):
        exchange.place_order('BTC-USD', 'buy', size=0.1)
    
    # Check performance
    pnl = exchange.get_pnl()
    print(f"Current PnL: ${pnl['total_pnl']:.2f}")
    
    time.sleep(60)  # Check every minute
```

---

## 13. Acceptance Criteria

Before considering the project complete, verify:

1. ✅ **Functional completeness**: All Phase 1-3 features implemented
2. ✅ **Test coverage**: 80%+ code coverage
3. ✅ **Performance**: Meets all performance targets in section 4.1
4. ✅ **Documentation**: README, API docs, and 3+ examples
5. ✅ **Validation**: Backtest results match manual calculations
6. ✅ **Live testing**: 24-hour paper trading session without crashes
7. ✅ **Code quality**: Passes linting (black, pylint)
8. ✅ **LLM integration**: Successfully integrated with sample LLM agent

---

## 14. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Inaccurate slippage modeling | High | Validate against real execution data; conservative defaults |
| Performance bottlenecks | Medium | Profile early; optimize hot paths; vectorize operations |
| WebSocket instability | Medium | Implement robust reconnection; buffer data; fallback to REST |
| Data quality issues | High | Validate data on load; handle gaps gracefully; log warnings |
| Scope creep | Medium | Strict adherence to MVP; defer enhancements to Phase 4+ |

---

## Appendix A: File Structure

```
simulated-exchange/
├── src/
│   ├── simulated_exchange/
│   │   ├── __init__.py
│   │   ├── exchange.py              # SimulatedExchange class
│   │   ├── price_feed.py            # PriceFeed abstract + implementations
│   │   ├── slippage.py              # Slippage models
│   │   ├── fees.py                  # Fee models
│   │   ├── performance.py           # Performance metrics
│   │   └── utils.py                 # Utilities
├── tests/
│   ├── test_exchange.py
│   ├── test_price_feed.py
│   ├── test_slippage.py
│   ├── test_fees.py
│   └── test_integration.py
├── examples/
│   ├── simple_backtest.py
│   ├── advanced_backtest.py
│   ├── paper_trading.py
│   └── llm_agent_integration.py
├── data/
│   └── sample_data.csv
├── docs/
│   ├── API.md
│   ├── TUTORIAL.md
│   └── EXAMPLES.md
├── config/
│   ├── default_config.py
│   └── hyperliquid_config.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## Appendix B: Glossary

- **Slippage**: Difference between expected and actual execution price
- **Basis Point (BPS)**: 1/100th of a percent (0.01%)
- **Market Order**: Order to buy/sell immediately at current market price
- **Limit Order**: Order to buy/sell at a specific price or better
- **Maker Fee**: Fee for orders that add liquidity (limit orders)
- **Taker Fee**: Fee for orders that remove liquidity (market orders)
- **Unrealized PnL**: Profit/loss on open positions
- **Realized PnL**: Profit/loss on closed positions
- **Equity Curve**: Time-series of portfolio value
- **Sharpe Ratio**: Risk-adjusted return metric
- **Drawdown**: Peak-to-trough decline in portfolio value
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

---

**Document Version**: 1.0  
**Last Updated**: October 27, 2025  
**Author**: Requirements specification for SimulatedExchange  
**Status**: Draft for Review
