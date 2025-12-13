# Quantopia Development Roadmap (Retail-Focused)

**Last Updated:** 2025-12-13
**Target User:** Retail traders with modest compute/data budgets
**Timeline:** 12-18 months to institutional-grade retail platform

---

## Executive Summary

This roadmap prioritizes improvements that:
1. **Prevent overfitting** (walk-forward testing, out-of-sample validation)
2. **Enable real trading** (paper trading, live execution)
3. **Improve ML capabilities** (complete autonomous training loop)
4. **Use free/affordable data** (no Bloomberg, no proprietary feeds)
5. **Run on consumer hardware** (no GPU clusters required)

**Key Insight:** You're currently 70% of the way to a serious retail quant platform. The missing 30% is achievable in 12-18 months.

---

## Phase 1: Critical Robustness (Months 1-3) 🔴 HIGH PRIORITY

### **Why This Matters:**
Right now, strategies are likely **overfitted**. A Sharpe 2.0 in-sample might be Sharpe 0.5 out-of-sample. This phase prevents you from deploying garbage.

### 1.1 Walk-Forward Testing Framework ⭐⭐⭐⭐⭐
**Priority:** CRITICAL
**Effort:** 2-3 weeks
**Impact:** Prevents overfitting, single biggest improvement

**What to Build:**
```python
# src/validation/walk_forward.py
class WalkForwardTester:
    """
    Train on [Year 1-2], test on [Year 3]
    Train on [Year 1-3], test on [Year 4]
    Train on [Year 1-4], test on [Year 5]
    """
    def __init__(
        self,
        train_window: int = 730,    # 2 years
        test_window: int = 365,     # 1 year
        step_size: int = 90         # Re-optimize every 90 days
    ):
        pass

    def run_walk_forward(self, strategy_spec, data):
        """Returns list of out-of-sample performance periods"""
        pass
```

**Key Features:**
- Expanding window (train on all prior data) OR rolling window (fixed lookback)
- Anchored walk-forward (realistic for retail - can't retrain daily)
- Aggregate out-of-sample metrics (only thing that matters)

**Data Requirements:**
- Just historical OHLCV (you already have this)
- Minimum 3 years for meaningful validation

**Success Metric:**
- Strategy has Sharpe 1.5 in-sample, Sharpe 1.2 out-of-sample = GOOD
- Strategy has Sharpe 2.0 in-sample, Sharpe 0.3 out-of-sample = OVERFITTED, REJECT

**Implementation Notes:**
- Add `out_of_sample_sharpe`, `out_of_sample_return`, `out_of_sample_dd` to database schema
- Update `StrategyFilter` to require minimum out-of-sample Sharpe (e.g., 0.8)
- Add to `ResearchEngine` Phase 3.5 (between backtesting and filtering)

---

### 1.2 Parameter Stability Testing ⭐⭐⭐⭐
**Priority:** HIGH
**Effort:** 1 week
**Impact:** Prevents fragile strategies

**What to Build:**
Test if strategy works with slightly different parameters:
```python
# Original: RSI(14) < 44
# Test variations:
#   RSI(12) < 44, RSI(14) < 42, RSI(14) < 46
#   RSI(16) < 44, RSI(12) < 42, etc.

# If strategy breaks with RSI(14.5), it's curve-fitted
```

**Success Metric:**
- Strategy Sharpe varies <20% across parameter neighborhood = ROBUST
- Strategy Sharpe varies >50% across parameter neighborhood = FRAGILE, REJECT

**Implementation:**
```python
# src/validation/stability_test.py
def test_parameter_stability(strategy_spec, perturbation=0.1):
    """
    Perturb each numeric parameter by ±10%
    Test if performance degrades significantly
    """
    pass
```

---

### 1.3 Monte Carlo Drawdown Analysis ⭐⭐⭐⭐
**Priority:** HIGH
**Effort:** 1 week
**Impact:** Realistic risk assessment

**What to Build:**
```python
# src/validation/monte_carlo.py
def monte_carlo_drawdown(trade_history, num_simulations=1000):
    """
    Randomly shuffle trade order
    Measure worst-case drawdown distribution

    Returns:
        - 95th percentile max DD
        - Probability of >30% DD
        - Expected max DD
    """
    pass
```

**Why This Matters:**
- Historical max DD = 15% is lucky
- Monte Carlo 95th percentile DD = 28% is realistic
- Prevents undersizing position for tail risk

**Free Tool:** Already have trade history, just need NumPy

---

## Phase 2: Complete ML Training Loop (Months 2-4) 🟡 HIGH PRIORITY

### **Current State:**
ML infrastructure is 80% done, but training is **mocked** in `ml_quant_agent.py:373-417`.

### 2.1 Activate Real ML Training ⭐⭐⭐⭐⭐
**Priority:** CRITICAL
**Effort:** 2-3 weeks
**Impact:** Enables autonomous ML strategy discovery

**What to Fix:**
Replace this in `src/agents/ml_quant_agent.py`:
```python
# BEFORE (Line 373-417):
# NOTE: Actual training would happen here using ModelTrainer
metrics = {
    'train_accuracy': 0.65,  # FAKE
    'test_accuracy': 0.60,   # FAKE
}

# AFTER:
model_path, metrics = self.model_trainer.train(
    data=training_data,
    target_variable=target_config['variable'],
    features=feature_plan,
    model_type=model_config['type'],
    hyperparameters=model_config['hyperparameters']
)
```

**Testing:**
- Generate 50 pure ML strategies autonomously
- Verify models actually train and save to disk
- Check model registry tracks versions correctly

**Data Requirements:**
- You already have this (OHLCV from exchanges)
- Use free technical indicators as features (20 already implemented)

---

### 2.2 Feature Engineering Library ⭐⭐⭐
**Priority:** MEDIUM-HIGH
**Effort:** 2 weeks
**Impact:** Better ML model inputs

**What to Build:**
```python
# src/ml/features/feature_library.py

class FeatureLibrary:
    """Curated feature sets for ML models"""

    @staticmethod
    def get_momentum_features(data):
        """RSI, ROC, Stochastic, MFI, WilliamsR"""
        pass

    @staticmethod
    def get_trend_features(data):
        """EMA slopes, ADX, MACD, trend strength"""
        pass

    @staticmethod
    def get_volatility_features(data):
        """ATR, Bollinger width, historical vol"""
        pass

    @staticmethod
    def get_regime_features(data):
        """Market regime, Hurst, vol regime"""
        pass

    @staticmethod
    def get_interaction_features(data):
        """
        RSI * TrendStrength
        ATR / BollingerBands_width
        Volume_zscore * ROC
        """
        pass
```

**Free Data Sources:**
- All from existing 20 indicators
- Add cross-sectional features if trading multiple assets
- No expensive data needed

---

### 2.3 Model Evaluation & Selection ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 1 week
**Impact:** Prevents deploying bad models

**What to Build:**
```python
# src/ml/evaluation/model_selector.py

def evaluate_model_quality(model, test_data):
    """
    Beyond just accuracy:
    - Precision/recall (avoid false positives)
    - Calibration (predicted probabilities accurate?)
    - Feature importance (makes sense?)
    - Prediction distribution (balanced?)
    """

    checks = {
        'min_test_accuracy': 0.55,      # Must beat random
        'min_precision': 0.50,           # Reduce false signals
        'max_feature_importance_concentration': 0.60,  # Not relying on 1 feature
        'calibration_score': 0.80        # Probabilities meaningful
    }

    return passes_all_checks
```

**Success Metric:**
- Model with test accuracy 0.62 but poor calibration = REJECT
- Model with test accuracy 0.58 but good precision/calibration = ACCEPT

---

## Phase 3: Transaction Cost Realism (Months 3-5) 🟡 MEDIUM-HIGH PRIORITY

### **Current State:**
Good slippage/fee models, but missing key retail trading costs.

### 3.1 Bid-Ask Spread Modeling ⭐⭐⭐⭐
**Priority:** HIGH (affects all strategies)
**Effort:** 1 week
**Impact:** More realistic returns

**What to Build:**
```python
# simulated_exchange/src/simulated_exchange/spread_model.py

class SpreadModel:
    def __init__(self, typical_spread_bps=10):
        """
        For BTC: ~2-5 bps on Binance/Coinbase
        For alts: ~10-50 bps
        For illiquid coins: 100-500 bps
        """
        self.spread_bps = typical_spread_bps

    def apply_spread_cost(self, price, side):
        """
        Market buy: Pay ask = mid * (1 + spread/2)
        Market sell: Get bid = mid * (1 - spread/2)
        """
        spread_pct = self.spread_bps / 20000.0  # Half-spread

        if side == OrderSide.BUY:
            return price * (1 + spread_pct)
        else:
            return price * (1 - spread_pct)
```

**Data Source (FREE):**
- Use CCXT `fetch_order_book()` to measure average spread
- Store spread statistics by symbol/exchange
- Default to 10 bps for crypto, 2 bps for BTC/ETH

**Impact:**
- Strategies with 100+ trades/year lose ~2-5% return to spreads
- High-frequency strategies become unprofitable

---

### 3.2 Minimum Order Size & Fees ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 3 days
**Impact:** Prevents micro-trading

**What to Build:**
```python
# Exchange minimums (Binance):
# - BTC: $10 min order
# - Most alts: $10 min order
# - Fee: 0.1% maker/taker (or 0.075% with BNB)

class OrderValidator:
    def validate_order_size(self, symbol, size, price):
        min_notional = 10.0  # USD
        notional = size * price

        if notional < min_notional:
            raise ValueError(f"Order ${notional:.2f} below ${min_notional} minimum")
```

**Impact:**
- Strategy with $1000 capital can't do 100 trades (each needs to be $100+)
- Forces more concentrated positions

---

### 3.3 Realistic Slippage Calibration ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 1 week
**Impact:** Accurate backtest results

**What to Build:**
Use real order book data to calibrate slippage models:

```python
# scratchpad/calibrate_slippage.py

def calibrate_from_orderbook(symbol='BTC/USDT', exchange='binance'):
    """
    Fetch order book snapshots
    Simulate market orders of various sizes
    Measure actual slippage

    Returns calibrated slippage parameters
    """

    # Example findings:
    # - $1000 BTC order: ~2 bps slippage
    # - $10000 BTC order: ~5 bps slippage
    # - $100000 BTC order: ~15 bps slippage

    return {
        'base_bps': 2,
        'volume_impact': 0.15,  # 15 bps per $100k
    }
```

**Data Source (FREE):**
- CCXT `fetch_order_book(limit=100)` - free API
- Sample throughout day to get average
- Update monthly

---

## Phase 4: Data Quality & Validation (Months 4-6) 🟢 MEDIUM PRIORITY

### **Current State:**
Data comes from CCXT (good), but no validation.

### 4.1 Data Quality Checks ⭐⭐⭐⭐
**Priority:** MEDIUM-HIGH
**Effort:** 1 week
**Impact:** Prevents bad backtests

**What to Build:**
```python
# src/data/quality_control.py

class DataQualityChecker:
    def validate_ohlcv(self, data: pd.DataFrame):
        """
        Checks:
        - No missing timestamps (detect gaps)
        - High >= Low (data integrity)
        - Volume > 0 (no placeholder bars)
        - No extreme outliers (>10 sigma moves = bad data)
        - Close within [Low, High]
        """

        issues = []

        # Check for gaps
        expected_bars = (data.index[-1] - data.index[0]) / timedelta(hours=1)
        actual_bars = len(data)
        if actual_bars < expected_bars * 0.95:
            issues.append(f"Missing {expected_bars - actual_bars} bars")

        # Check for outliers
        returns = data['close'].pct_change()
        if (returns.abs() > 0.20).any():
            issues.append("Extreme moves detected (>20% bar)")

        # Check integrity
        if (data['high'] < data['low']).any():
            issues.append("High < Low detected")

        return issues
```

**Implementation:**
- Run on all downloaded data before backtesting
- Add `data_quality_score` to database
- Reject strategies tested on bad data

---

### 4.2 Delistings & Survivorship Bias (Partial) ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 2 weeks
**Impact:** More realistic for multi-asset strategies

**What to Build (Simplified for Retail):**
```python
# src/data/delisting_tracker.py

class DelistingTracker:
    """
    Track when coins get delisted from exchanges
    For multi-coin strategies, simulate what happens if you held a coin that died
    """

    def __init__(self):
        # Maintain manual list of major delistings
        self.delisted_coins = {
            'LUNA/USDT': datetime(2022, 5, 13),   # Terra collapse
            'FTT/USDT': datetime(2022, 11, 11),   # FTX collapse
            # ... add major ones
        }

    def check_if_active(self, symbol, date):
        """Return False if coin was delisted before this date"""
        if symbol in self.delisted_coins:
            return date < self.delisted_coins[symbol]
        return True
```

**Realistic Approach:**
- Don't need perfect survivorship bias correction
- Just need to include some failed coins in universe
- Manually track 20-30 major delistings
- Good enough for retail (institutions pay millions for complete data)

---

### 4.3 Multiple Exchanges & Data Validation ⭐⭐
**Priority:** LOW-MEDIUM
**Effort:** 1 week
**Impact:** Catch bad data

**What to Build:**
```python
# src/data/cross_validation.py

def cross_validate_price_data(symbol, date_range):
    """
    Fetch same data from Binance, Coinbase, Kraken
    Compare prices - should be within 0.1%
    If divergence > 0.5%, flag as suspicious
    """

    binance_data = download_from_binance(symbol, date_range)
    coinbase_data = download_from_coinbase(symbol, date_range)

    price_diff = abs(binance_data['close'] - coinbase_data['close']) / binance_data['close']

    if (price_diff > 0.005).any():
        return "WARNING: Price divergence detected"

    return "OK"
```

**Free Tool:** CCXT supports 100+ exchanges, all free tier

---

## Phase 5: Production Readiness (Months 5-8) 🟡 HIGH PRIORITY

### **Current State:**
System generates strategies but can't trade them.

### 5.1 Paper Trading Engine ⭐⭐⭐⭐⭐
**Priority:** CRITICAL
**Effort:** 3 weeks
**Impact:** Validation before risking real money

**What to Build:**
```python
# src/live/paper_trader.py

class PaperTrader:
    """
    Run strategies in real-time with fake money
    Track divergence vs backtest
    """

    def __init__(self, strategy, initial_capital=10000):
        self.strategy = strategy
        self.capital = initial_capital
        self.positions = {}
        self.trades = []

    async def run(self):
        """
        1. Fetch latest candle from exchange
        2. Update indicators
        3. Check entry/exit signals
        4. Execute fake trades at market price
        5. Log everything
        """
        while True:
            latest_data = await self.fetch_latest_candle()
            signal = self.strategy.get_signal(latest_data)

            if signal == 'buy':
                self.execute_paper_trade('buy', latest_data['close'])
            elif signal == 'sell':
                self.execute_paper_trade('sell', latest_data['close'])

            await asyncio.sleep(60)  # Check every minute
```

**Key Features:**
- Real-time data feed from CCXT (free)
- Simulate execution at current market price
- Log all signals and trades to database
- Compare paper vs backtest performance

**Success Metric:**
- Paper trading Sharpe matches backtest ±20% = GOOD
- Paper trading Sharpe 50% lower than backtest = OVERFIT or bad data

---

### 5.2 Live Trading Engine (Basic) ⭐⭐⭐⭐
**Priority:** HIGH
**Effort:** 2 weeks
**Impact:** Actually make money

**What to Build:**
```python
# src/live/live_trader.py

class LiveTrader:
    """
    Execute real trades on exchange
    WITH SAFETY LIMITS
    """

    def __init__(
        self,
        strategy,
        exchange,
        max_position_size=1000,      # Max $1000 per trade
        max_daily_trades=10,          # Circuit breaker
        max_daily_loss=100            # Stop if lose $100 in a day
    ):
        self.strategy = strategy
        self.exchange = exchange
        self.safety_limits = SafetyLimits(
            max_position_size, max_daily_trades, max_daily_loss
        )

    async def execute_trade(self, side, symbol):
        """
        1. Check safety limits
        2. Calculate position size
        3. Submit market order to exchange
        4. Log execution details
        5. Update database
        """

        # Safety check
        if not self.safety_limits.check_ok_to_trade():
            logger.error("Safety limit triggered, halting trading")
            return

        # Execute
        order = await self.exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=position_size
        )

        self.log_trade(order)
```

**Safety Features (CRITICAL):**
- Max position size per trade
- Max number of trades per day
- Max daily loss (circuit breaker)
- Manual approval for first N trades
- SMS/email alerts on every trade

---

### 5.3 Divergence Tracking ⭐⭐⭐⭐
**Priority:** HIGH
**Effort:** 1 week
**Impact:** Detect strategy degradation

**What to Build:**
```python
# src/live/divergence_tracker.py

class DivergenceTracker:
    """
    Compare live performance vs backtest expectations
    Alert if diverging
    """

    def check_divergence(self, live_metrics, backtest_metrics):
        """
        Compare:
        - Win rate (should be within ±10%)
        - Average trade P&L (within ±20%)
        - Sharpe ratio (within ±30%)
        - Number of trades (within ±25%)
        """

        divergences = []

        if abs(live_metrics['win_rate'] - backtest_metrics['win_rate']) > 0.10:
            divergences.append(
                f"Win rate divergence: {live_metrics['win_rate']:.1%} vs {backtest_metrics['win_rate']:.1%}"
            )

        if abs(live_metrics['sharpe'] - backtest_metrics['sharpe']) > 0.3:
            divergences.append(
                f"Sharpe divergence: {live_metrics['sharpe']:.2f} vs {backtest_metrics['sharpe']:.2f}"
            )

        return divergences
```

**Alerts:**
- Daily email summary
- Slack webhook (free)
- Telegram bot (free)

---

## Phase 6: Multi-Asset & Portfolio Features (Months 7-10) 🟢 MEDIUM PRIORITY

### 6.1 Multi-Asset Strategy Support ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 2 weeks
**Impact:** Diversification

**What to Build:**
```python
# Currently: Single asset (BTC-USD)
# After: Portfolio of assets

class MultiAssetStrategy(BaseStrategy):
    def __init__(self, symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT']):
        self.symbols = symbols
        self.allocations = {}  # Track per-asset positions

    def should_enter(self, symbol):
        """Decide per-asset"""
        pass

    def should_exit(self, symbol):
        """Decide per-asset"""
        pass
```

**Use Cases:**
- Sector rotation (rotate between BTC, ETH, alts based on momentum)
- Pairs trading (long BTC, short ETH when ratio extreme)
- Portfolio rebalancing

**Data Requirements:**
- Already have this (CCXT supports 1000s of pairs)
- Just need to download multiple symbols

---

### 6.2 Portfolio Optimization ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 2 weeks
**Impact:** Better risk-adjusted returns

**What to Build:**
```python
# src/portfolio/optimizer.py

class PortfolioOptimizer:
    """
    Given N strategies, find optimal weights
    """

    def optimize_weights(self, strategies, objective='sharpe'):
        """
        Objectives:
        - Max Sharpe
        - Min variance
        - Equal risk contribution
        - Max diversification

        Uses scipy.optimize (free library)
        """

        # Build covariance matrix
        returns = self._get_returns_matrix(strategies)
        cov_matrix = returns.cov()

        # Optimize
        if objective == 'sharpe':
            weights = self._max_sharpe(returns, cov_matrix)
        elif objective == 'min_variance':
            weights = self._min_variance(cov_matrix)

        return weights
```

**Free Tools:**
- SciPy for optimization
- NumPy for linear algebra
- Already have returns data

---

### 6.3 Regime-Aware Position Sizing ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 1 week
**Impact:** Reduce drawdowns

**What to Build:**
```python
# src/portfolio/regime_sizer.py

class RegimeAwarePositionSizer:
    """
    Reduce position size in high-volatility regimes
    Increase in low-volatility regimes
    """

    def calculate_position_size(self, base_size, market_regime):
        """
        market_regime from VolatilityRegime indicator:
        - 'low_vol': 1.2x base size
        - 'normal': 1.0x base size
        - 'high_vol': 0.6x base size
        - 'extreme_vol': 0.3x base size
        """

        multipliers = {
            'low_vol': 1.2,
            'normal': 1.0,
            'high_vol': 0.6,
            'extreme_vol': 0.3
        }

        return base_size * multipliers.get(market_regime, 1.0)
```

**Data Source:**
- Already have VolatilityRegime indicator
- Free, no extra data needed

---

## Phase 7: Advanced Strategy Types (Months 8-12) 🔵 LOWER PRIORITY

### 7.1 Mean Reversion Strategies ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 2 weeks
**Impact:** Uncorrelated to trend-following

**What to Build:**
```python
# Already have infrastructure
# Just need new strategy patterns:

# Bollinger Band Mean Reversion
entry: price < lower_band AND RSI < 40
exit: price > middle_band

# Pairs Trading (if multi-asset)
ratio = BTC_price / ETH_price
entry_long_btc: ratio < mean - 2*std
exit: ratio > mean
```

**Note:** System already supports this, just need LLM to explore these patterns

---

### 7.2 Breakout Strategies with Volume ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 1 week
**Impact:** Capture momentum moves

**What to Build:**
```python
# Breakout patterns
entry: price > Donchian_upper AND volume > 2 * avg_volume
exit: price < Donchian_middle OR ATR_stop_loss
```

---

### 7.3 Short Selling Support ⭐⭐
**Priority:** LOW-MEDIUM
**Effort:** 2 weeks
**Impact:** Double the opportunity set

**What to Build:**
```python
# Modify BaseStrategy to support shorts

class BaseStrategy:
    def should_enter_long(self):
        pass

    def should_enter_short(self):  # NEW
        pass

    def should_exit_long(self):
        pass

    def should_exit_short(self):  # NEW
        pass
```

**Exchange Support:**
- Binance Futures: supports shorts
- Many spot exchanges: no shorts
- Check if exchange supports margin trading

---

## Phase 8: Performance & Scale (Months 10-15) 🔵 LOWER PRIORITY

### 8.1 Parallel Backtesting Optimization ⭐⭐
**Priority:** LOW
**Effort:** 1 week
**Impact:** Faster iteration

**Current:** Already parallel via `batch_tester.py`
**Improvement:** GPU-accelerated indicator calculation

```python
# Use CuPy instead of NumPy for indicators
import cupy as cp  # GPU arrays

def calculate_ema_gpu(prices, period):
    """100x faster for large datasets"""
    prices_gpu = cp.array(prices)
    # ... EMA calculation on GPU
    return cp.asnumpy(result)
```

**Cost:** NVIDIA GPU ($200-$2000)
**Benefit:** 10-100x speedup for large backtests

---

### 8.2 Strategy Caching & Incremental Updates ⭐⭐
**Priority:** LOW
**Effort:** 1 week
**Impact:** Faster re-testing

**What to Build:**
```python
# Don't re-backtest entire history
# Just append new bars

class IncrementalBacktester:
    def update_with_new_data(self, strategy_id, new_bars):
        """
        Load previous equity curve
        Run strategy on new bars only
        Append to existing results
        """
        pass
```

---

## Phase 9: Advanced ML (Months 12-18) 🔵 LOWER PRIORITY

### 9.1 Ensemble Models ⭐⭐⭐
**Priority:** MEDIUM
**Effort:** 2 weeks
**Impact:** More robust predictions

**What to Build:**
```python
# Combine multiple models
ensemble = [
    XGBoostModel(),
    RandomForestModel(),
    LightGBMModel()
]

# Average predictions
final_prediction = np.mean([m.predict(X) for m in ensemble])
```

---

### 9.2 Feature Selection ⭐⭐
**Priority:** LOW
**Effort:** 1 week
**Impact:** Reduce overfitting

**What to Build:**
```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression

def select_best_features(X, y, k=20):
    """
    From 100+ features, select top 20
    Based on mutual information
    """
    selector = SelectKBest(mutual_info_regression, k=k)
    X_selected = selector.fit_transform(X, y)

    return selector.get_support(indices=True)
```

---

### 9.3 Online Learning (Adaptive Models) ⭐⭐
**Priority:** LOW
**Effort:** 3 weeks
**Impact:** Models adapt to regime changes

**What to Build:**
```python
# Update model weekly with new data
class OnlineLearner:
    def update_model(self, new_data):
        """
        Retrain on [last 6 months + new week]
        Compare to old model
        Switch if new model better out-of-sample
        """
        pass
```

**Note:** Risky (model drift), save for advanced users

---

## Summary: Prioritized Implementation Order

### **Months 1-3 (Foundation)**
1. ✅ Walk-forward testing (CRITICAL)
2. ✅ Parameter stability testing
3. ✅ Complete ML training loop
4. ✅ Monte Carlo drawdown analysis

**Goal:** Prevent overfitting, enable autonomous ML strategies

---

### **Months 4-6 (Realism)**
1. ✅ Bid-ask spread modeling
2. ✅ Data quality validation
3. ✅ Realistic slippage calibration
4. ✅ Feature engineering library

**Goal:** Backtest results match real trading

---

### **Months 7-9 (Production)**
1. ✅ Paper trading engine
2. ✅ Divergence tracking
3. ✅ Live trading engine (basic)
4. ✅ Multi-asset support (optional)

**Goal:** Deploy strategies with real capital

---

### **Months 10-15 (Scaling)**
1. ✅ Portfolio optimization
2. ✅ Regime-aware position sizing
3. ✅ Short selling support (if needed)
4. ✅ Ensemble ML models

**Goal:** Sophisticated portfolio management

---

### **Months 15-18 (Advanced)**
1. ✅ GPU acceleration (if needed)
2. ✅ Online learning
3. ✅ Advanced strategy types

**Goal:** Institutional-grade retail platform

---

## What NOT to Do (Waste of Time for Retail)

### ❌ Skip These (Institutional-Only):
1. **Tick data** - Costs $1000s/month, only for HFT
2. **Bloomberg Terminal** - $24k/year, unnecessary
3. **Proprietary data feeds** - Expensive, marginal value
4. **Ultra-low latency** - Retail can't compete with HFT
5. **Perfect survivorship bias correction** - Manual tracking of delistings is enough
6. **Order book depth modeling** - Good enough with slippage models
7. **GPU clusters** - Single GPU or CPU is fine
8. **Real-time news sentiment** - Expensive APIs, low signal

### ✅ Use These (Free/Cheap Retail Tools):
1. **CCXT** - Free exchange APIs for data/trading
2. **TA-Lib / pandas-ta** - Free technical indicators
3. **Scikit-learn / XGBoost** - Free ML models
4. **SQLite** - Free database (good for millions of rows)
5. **Python / NumPy** - Free and fast
6. **TradingView** - $10-30/month for charting (optional)

---

## Success Metrics by Phase

### Phase 1 Success:
- [ ] 80%+ strategies have out-of-sample Sharpe >50% of in-sample Sharpe
- [ ] <20% strategies flagged as parameter-fragile
- [ ] Monte Carlo 95th percentile DD calculated for all strategies

### Phase 2 Success:
- [ ] ML models train autonomously with real data
- [ ] XGBoost strategies achieving Sharpe >1.0 out-of-sample
- [ ] Model registry has 10+ unique ML models

### Phase 3 Success:
- [ ] Backtest returns within 2-3% of hand-calculated with spread costs
- [ ] Slippage calibrated from real order book data

### Phase 4 Success:
- [ ] Data quality checks catch >90% of bad data
- [ ] Multi-asset backtests include 3+ delisted coins

### Phase 5 Success:
- [ ] 3 strategies paper traded for 30 days with divergence <30%
- [ ] 1 strategy live traded with $500 capital for 60 days

### Phase 6 Success:
- [ ] Portfolio of 5 strategies has Sharpe >1.3 (vs avg 1.0 individual)
- [ ] Regime-aware sizing reduces drawdowns by 20%+

---

## Estimated Resources

### Time Commitment:
- **Solo developer:** 12-18 months part-time (10-15 hrs/week)
- **With 1 helper:** 8-12 months part-time
- **Full-time:** 6-9 months

### Financial Resources:
- **Data:** $0/month (CCXT is free)
- **Compute:** $0-50/month (cloud VM optional, local is fine)
- **APIs:** $0/month (free tier LLMs or self-hosted)
- **Trading capital:** $500-5000 (start small)
- **Total:** <$100/month

### Technical Skills Needed:
- Python (intermediate)
- Pandas/NumPy (intermediate)
- Basic statistics (mean, std dev, correlation)
- Git (basic)
- SQL (basic)

---

## Conclusion

You're **70% done** with a serious retail quant platform. The remaining 30% is:

1. **Walk-forward testing** (CRITICAL - prevents overfitting)
2. **Complete ML training loop** (HIGH - enables autonomous ML)
3. **Paper trading** (HIGH - validate before deploying)
4. **Transaction cost realism** (MEDIUM - accurate backtests)
5. **Data quality** (MEDIUM - catch bad data)

**With 12-18 months of focused work, you can have an institutional-grade retail quant research platform.**

The key constraint is time, not money. All the tools you need are free or cheap. The path is clear.

**Start with Phase 1. Everything else is useless if strategies are overfitted.**
