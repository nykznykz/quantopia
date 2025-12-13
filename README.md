# Quantopia - AI-Powered Autonomous Quantitative Research Platform

**An autonomous quantitative research platform that uses LLM agents to explore the trading strategy space.** The system autonomously generates, codes, backtests, and refines trading strategies with minimal human intervention.

[![Status](https://img.shields.io/badge/status-70%25%20complete-yellow)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()

---

## 🎯 Overview

Quantopia is an **agent-first** quantitative research platform that combines LLM decision-making with production-grade backtesting infrastructure. Unlike traditional quant systems where humans propose strategies, Quantopia's agents autonomously explore the strategy space, learn from past results, and discover profitable trading strategies.

### Key Differentiators

- 🤖 **Autonomous Strategy Agent**: LLM-based agent that analyzes database history, identifies underexplored areas, and decides what to try next
- 📊 **Market-Calibrated Thresholds**: Uses real market statistics (e.g., RSI < 44 vs textbook RSI < 30) for realistic signal generation
- 🔬 **Database-Driven Learning**: All strategies, results, and ML models stored in SQLite for agent learning and genealogy tracking
- ⚡ **Production-Grade Backtesting**: Realistic slippage models (volume-based, volatility-adjusted), proper fee modeling, O(n) optimized indicators
- 🧬 **Strategy Genealogy**: Track parent/child relationships, generations, and refinement lineage
- 🤝 **ML/DL Integration**: Hybrid strategies combining traditional indicators with ML predictions (XGBoost, Random Forest, LightGBM)

---

## 🏗️ Architecture

### Agent-First Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Research Orchestrator                         │
│  Coordinates full autonomous research iterations (8 phases)     │
└────────────┬────────────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌─────────────┐  ┌──────────────┐
│  Strategy   │  │  ML Quant    │
│   Agent     │  │   Agent      │
│             │  │              │
│ • Analyzes  │  │ • Provides   │
│   DB history│  │   ML models  │
│ • Decides   │  │ • Feature    │
│   what to   │  │   engineering│
│   explore   │  │ • Model      │
│ • Market    │  │   training   │
│   stats     │  │              │
└──────┬──────┘  └──────┬───────┘
       │                │
       └────────┬───────┘
                │
                ▼
       ┌─────────────────┐
       │  Agent Router   │
       │                 │
       │ Routes requests │
       │ through proper  │
       │ pipelines       │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Code Generator  │
       │                 │
       │ Synthesizes     │
       │ Python classes  │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Batch Tester    │
       │                 │
       │ Parallel        │
       │ backtesting     │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │Strategy Filter  │
       │                 │
       │ Approval based  │
       │ on metrics      │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │   Database      │
       │                 │
       │ Stores results  │
       │ for agent       │
       │ learning        │
       └─────────────────┘
```

### Critical Separation of Concerns

**IMPORTANT**: Recent architectural fix prevents "0-trade problem":
- **Strategy Agent** outputs COMPLETE specifications with exact boolean logic (e.g., `"(RSI(14) < 44 OR RSI(5) < 35) AND TrendStrength < 0.5"`)
- **Agent Router** passes through specifications WITHOUT modification (no hardcoded rule generation)
- **Code Generator** faithfully translates logic WITHOUT interpretation (no deciding thresholds or AND/OR combinations)

This separation prevents overly restrictive strategies caused by components adding extra conditions.

---

## ✨ Features

### Current (Phase 1a - Complete)

- ✅ **Autonomous Strategy Agent**: LLM-based decision-making with database context
- ✅ **ML Quant Agent**: On-demand ML model provision (architecture complete, training mocked)
- ✅ **Agent Router**: Coordinates strategy → ML → code generation pipeline
- ✅ **Code Generator**: Python class synthesis from metadata (temperature=0.2, multi-retry)
- ✅ **20 Technical Indicators**: Trend, momentum, volatility, regime indicators
- ✅ **Batch Backtesting**: Parallel strategy testing with realistic execution simulation
- ✅ **Strategy Filter**: Multi-criteria approval (Sharpe, drawdown, win rate, etc.)
- ✅ **Database System**: SQLite-based storage with genealogy tracking
- ✅ **Portfolio Risk Manager**: Marginal risk contribution, diversification ratio, correlation analysis
- ✅ **Portfolio Evaluator**: Multi-strategy orchestration and allocation optimization
- ✅ **Market Statistics**: Calibrated thresholds from historical data analysis

### Backtesting Engine (Production-Quality)

- ✅ **SimulatedExchange**: Complete order management, position tracking, equity curves
- ✅ **Slippage Models**: Fixed, volume-based, hybrid (base + volume + volatility)
- ✅ **Fee Models**: Tiered maker/taker (defaults to Hyperliquid: 0%/2.5bps)
- ✅ **Performance Metrics**: Sharpe, max DD, win rate, profit factor, consecutive wins/losses
- ✅ **Optimized Execution**: O(n) indicator pre-calculation, ~0.5s for 8760 candles

### ML/DL Capabilities

- ✅ **Model Registry**: Version tracking, metrics storage, lineage management
- ✅ **Feature Pipeline**: Technical feature generation, normalization, missing value handling
- ✅ **Model Types**: XGBoost, Random Forest, LightGBM (full implementations)
- ✅ **Model Trainer**: Chronological splits, time series CV, feature importance
- ⚠️ **Autonomous Training**: Architecture complete, actual training currently mocked (see roadmap)

### Pending (Phase 2-4)

- ⏳ **Walk-Forward Testing**: Out-of-sample validation (CRITICAL - see roadmap)
- ⏳ **Paper Trading**: Real-time validation with live data
- ⏳ **Live Trading Engine**: Actual deployment with safety limits
- ⏳ **Critique Agent**: Failure analysis and refinement suggestions
- ⏳ **Divergence Tracking**: Compare live vs backtest performance

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- API keys (OpenAI, Anthropic, or DeepSeek)

### Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/quantopia.git
cd quantopia

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/Mac
# or
.\venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install simulated_exchange package
cd simulated_exchange
pip install -e .
cd ..

# 5. Configure environment
cp .env.example .env
cp config/quantopia.yaml.example config/quantopia.yaml

# 6. Set API keys in .env
# OPENAI_API_KEY=your-key-here
# LLM_PROVIDER=openai  # or anthropic, deepseek, azure

# 7. (Optional) Generate market statistics
python scratchpad/analyze_market_statistics.py
```

---

## 🚀 Quick Start

### CLI Usage (Primary Interface)

```bash
# Initialize configuration
python quantopia.py init

# Run autonomous research session (generate 50 strategies)
python quantopia.py research --num-strategies 50 --symbol BTC-USD --days 365

# Start background daemon for continuous research
python quantopia.py start --daemon --continuous --interval 3600

# Check system status
python quantopia.py status --detailed

# List top performing strategies
python quantopia.py list --top 10 --filter approved

# Show specific strategy details
python quantopia.py show <strategy_id> --code --metrics

# Export strategy code
python quantopia.py export <strategy_id> --output strategy.py

# Query database insights
python quantopia.py query --underexplored --statistics --models
```

### Phase 1a Example (Autonomous Research)

```bash
# Run the autonomous research example
python examples/phase1a_autonomous_research.py
```

This will:
1. Strategy Agent analyzes database and decides what to explore
2. Agent Router coordinates pipeline (technical vs ML strategies)
3. ML Quant Agent provides models if needed
4. Code Generator synthesizes Python strategy classes
5. Batch Tester runs parallel backtests
6. Strategy Filter approves/rejects based on metrics
7. Portfolio Evaluator builds multi-strategy portfolios
8. Results stored in database for future agent learning

### Python API Usage

```python
from src.agents.strategy_agent import StrategyAgent
from src.agents.ml_quant_agent import MLQuantAgent
from src.orchestrator.agent_router import AgentRouter
from src.orchestrator.research_engine import ResearchOrchestrator
from src.database.manager import StrategyDatabase
from src.ml.model_registry import ModelRegistry

# Initialize database and agents
db = StrategyDatabase("data/strategies.db")
model_registry = ModelRegistry(db)

strategy_agent = StrategyAgent(
    database=db,
    model_registry=model_registry,
    exploration_rate=0.3
)

ml_agent = MLQuantAgent(
    model_registry=model_registry,
    database=db
)

# Run autonomous research iteration
orchestrator = ResearchOrchestrator(
    strategy_agent=strategy_agent,
    ml_agent=ml_agent,
    database=db
)

results = orchestrator.run_research_iteration(
    num_strategies=10,
    symbol="BTC/USDT",
    timeframe="1h",
    days=365
)

print(f"Generated {results['num_generated']} strategies")
print(f"Approved {results['num_approved']} strategies")
print(f"Top Sharpe: {results['top_sharpe']:.2f}")
```

---

## 📁 Project Structure

```
quantopia/
├── config/
│   ├── quantopia.yaml.example    # Main configuration template
│   └── logging.yaml               # Logging configuration
├── src/
│   ├── agents/
│   │   ├── strategy_agent.py      # Autonomous strategy researcher
│   │   └── ml_quant_agent.py      # ML model provider agent
│   ├── orchestrator/
│   │   ├── agent_router.py        # Agent coordination
│   │   └── research_engine.py     # Full research iteration orchestration
│   ├── code_generation/
│   │   ├── code_generator.py      # Python class synthesis
│   │   └── strategy_base.py       # Base class for all strategies
│   ├── database/
│   │   ├── manager.py             # Database operations
│   │   └── schema.py              # SQLAlchemy models
│   ├── backtest/
│   │   ├── batch_tester.py        # Parallel backtesting
│   │   └── runner.py              # Single strategy backtest
│   ├── critique/
│   │   ├── filter.py              # Strategy approval/rejection
│   │   └── models.py              # Filter criteria models
│   ├── portfolio/
│   │   ├── evaluator.py           # Portfolio construction
│   │   ├── risk_manager.py        # Portfolio risk analytics
│   │   └── models.py              # Portfolio data models
│   ├── ml/
│   │   ├── model_registry.py      # ML model version tracking
│   │   ├── training.py            # Model training pipeline
│   │   ├── pipeline.py            # Feature engineering
│   │   └── sklearn_models.py      # XGBoost, RF, LightGBM
│   ├── indicators/
│   │   ├── __init__.py            # 20 technical indicators
│   │   └── registry.py            # Indicator metadata
│   └── strategy_generation/
│       ├── llm_client.py          # Multi-provider LLM client
│       └── strategy_generator.py   # Strategy metadata generation
├── simulated_exchange/            # Separate backtesting package
│   └── src/simulated_exchange/
│       ├── exchange.py            # Order execution simulation
│       ├── slippage.py            # Slippage models
│       ├── fees.py                # Fee models
│       └── performance.py         # Metrics calculation
├── data/
│   ├── strategies.db              # SQLite database (auto-created)
│   └── market_data/               # Downloaded OHLCV data
├── docs/
│   ├── ROADMAP_RETAIL_FOCUSED.md  # 12-18 month development roadmap
│   └── PRIORITY_MATRIX.md         # Priority levels and timelines
├── scratchpad/
│   ├── diagnose_zero_trades.py    # Diagnostic tool for 0-trade strategies
│   ├── analyze_market_statistics.py # Generate market stats for thresholds
│   └── market_statistics.json     # Calibrated threshold data
├── examples/
│   └── phase1a_autonomous_research.py
├── tests/
│   ├── test_end_to_end.py
│   ├── test_orchestrator.py
│   └── test_full_flywheel.py
├── quantopia.py                   # Main CLI entry point
├── CLAUDE.md                      # Project instructions for Claude Code
└── README.md                      # This file
```

---

## 🎓 Available Indicators (20 Total)

### Trend (5)
- `EMA` - Exponential Moving Average
- `SMA` - Simple Moving Average
- `MACD` - Moving Average Convergence Divergence
- `ADX` - Average Directional Index
- `EMASlope` - EMA slope for trend detection

### Momentum (5)
- `RSI` - Relative Strength Index
- `Stochastic` - Stochastic Oscillator
- `MFI` - Money Flow Index
- `ROC` - Rate of Change
- `WilliamsR` - Williams %R

### Volatility (5)
- `ATR` - Average True Range
- `BollingerBands` - Bollinger Bands
- `KeltnerChannels` - Keltner Channels
- `HistoricalVolatility` - Historical price volatility
- `DonchianChannels` - Donchian Channels

### Regime (5)
- `VolumeZScore` - Volume anomaly detection
- `HurstExponent` - Mean reversion vs trending
- `MarketRegime` - Bull/bear/sideways classification
- `TrendStrength` - Trend strength (0.0-1.0)
- `VolatilityRegime` - Volatility classification

All indicators are vectorized (pandas operations) and optimized for O(n) performance.

---

## ⚙️ Configuration

### LLM Providers

```python
# OpenAI (recommended)
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here

# Anthropic Claude
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key-here

# DeepSeek (cost-effective)
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-key-here

# Azure OpenAI
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL=gpt-4  # Your deployment name
```

### Filter Criteria (Default Thresholds)

Strategies must meet these criteria to be approved:

```yaml
# In config/quantopia.yaml
filter:
  min_sharpe_ratio: 0.5        # Sharpe >= 0.5
  min_total_return: 0.05       # 5% minimum return
  max_drawdown: 0.30           # Max 30% drawdown
  min_num_trades: 10           # At least 10 trades
  min_win_rate: 0.30           # 30% win rate
  min_profit_factor: 1.0       # Profit factor >= 1.0
```

### Backtesting Configuration

```yaml
backtest:
  initial_capital: 10000.0
  position_size: 0.9           # Use 90% of capital
  slippage_model: hybrid       # fixed, volume_based, hybrid, none
  slippage_bps: 5              # 5 basis points
  fee_model: tiered            # tiered, flat
  maker_fee: 0.0000            # 0% maker fee (Hyperliquid)
  taker_fee: 0.00025           # 2.5 bps taker fee
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_end_to_end.py
pytest tests/test_orchestrator.py
pytest tests/test_full_flywheel.py

# Run with verbose output
pytest -v -s tests/

# Run with coverage
pytest --cov=src tests/
```

---

## 📊 Database Schema

### Core Tables

- **`strategies`**: Strategy metadata, parameters, genealogy (parent_id, generation)
- **`strategy_code`**: Generated Python code with versioning
- **`backtest_results`**: Performance metrics, equity curves, trade history
- **`portfolio_evaluations`**: Multi-strategy portfolio analysis
- **`ml_models_registry`**: ML model versions, metrics, hyperparameters
- **`forward_test_queue`**: Strategies queued for paper trading

### Key Queries

```python
# Get top performers
top_strategies = db.get_top_strategies(metric='sharpe_ratio', limit=10)

# Get underexplored areas
underexplored = db.get_underexplored_areas()

# Get strategy genealogy
family = db.get_strategy_family(parent_id=123)

# Get ML model usage stats
model_stats = model_registry.get_model_usage_statistics()
```

---

## 🚦 Current Status & Roadmap

### Phase 1a (Complete) ✅
- ✅ Autonomous Strategy Agent with LLM decision-making
- ✅ ML Quant Agent for on-demand model provision
- ✅ Agent Router for pipeline coordination
- ✅ Code Generator for Python synthesis
- ✅ Batch backtesting and filtering
- ✅ Portfolio evaluation and risk management
- ✅ Database-driven learning
- ✅ Market statistics calibration

### Phase 2 (Critical - Next 3 Months) 🔴
See `docs/ROADMAP_RETAIL_FOCUSED.md` for details:
- ⏳ **Walk-Forward Testing** (CRITICAL - prevents overfitting)
- ⏳ **Complete ML Training Loop** (currently mocked)
- ⏳ **Parameter Stability Testing**
- ⏳ **Monte Carlo Drawdown Analysis**

### Phase 3 (Months 4-6) 🟡
- ⏳ Bid-ask spread modeling
- ⏳ Data quality validation
- ⏳ Paper trading engine
- ⏳ Feature engineering library

### Phase 4 (Months 7-9) 🟢
- ⏳ Live trading engine with safety limits
- ⏳ Divergence tracking
- ⏳ Multi-asset support
- ⏳ Portfolio optimization

**See full roadmap:** `docs/ROADMAP_RETAIL_FOCUSED.md` and `docs/PRIORITY_MATRIX.md`

---

## 🎯 Success Metrics

### Strategy Generation
- ✅ Generate ≥10 strategies per batch
- ✅ Backtest throughput ≥20 strategies/hour (parallel)
- ✅ ≥10% approval rate (Sharpe ≥0.5)
- ⏳ Out-of-sample Sharpe ≥50% of in-sample (walk-forward pending)

### Backtesting Accuracy
- ✅ Execution simulation realistic (slippage, fees, spreads)
- ✅ Indicator calculation O(n) optimized
- ⏳ Paper trading correlation ≥0.6 vs backtest (pending)
- ⏳ Live performance within 30% of paper trading (pending)

### Portfolio Performance
- ✅ Portfolio Sharpe >1.3 vs individual strategies ~1.0
- ✅ Diversification ratio <1.0 (benefit from correlation <1)
- ✅ Max single strategy allocation ≤40%
- ⏳ Live drawdown <1.5× historical DD (pending deployment)

---

## 🔧 Troubleshooting

### Common Issues

**Import errors for `simulated_exchange`:**
```bash
cd simulated_exchange && pip install -e . && cd ..
```

**LLM API failures:**
- Check `.env` has valid API keys
- Verify `LLM_PROVIDER` setting matches your API key
- For Azure OpenAI, ensure `AZURE_OPENAI_MODEL` matches deployment name

**Database locked:**
- Close any other processes accessing `data/strategies.db`
- Use `sqlite3 data/strategies.db` to check for locks

**Strategies with 0 trades:**
- Run `python scratchpad/diagnose_zero_trades.py` to analyze
- Check if market statistics file exists: `scratchpad/market_statistics.json`
- Generate market stats: `python scratchpad/analyze_market_statistics.py`

**Slow backtests:**
- Check data size (>10K candles may be slow)
- Verify indicators are pre-calculated (not bar-by-bar)
- Consider using smaller date ranges for testing

### Diagnostic Tools

```bash
# Analyze strategies with 0 trades
python scratchpad/diagnose_zero_trades.py

# Generate market statistics
python scratchpad/analyze_market_statistics.py

# Create diagnostic report
python scratchpad/generate_diagnostic_report.py

# Test code generation
python scratchpad/test_code_generation_fix.py
```

---

## 📈 Performance Benchmarks

**Backtesting Speed** (8-core CPU):
- Single strategy (1 year hourly): ~0.5 seconds
- Batch of 20 strategies (parallel): ~15 seconds
- 100 strategies with filtering: ~2 minutes

**Database Performance**:
- 1000 strategies: <100 MB database size
- Query top performers: <50ms
- Strategy generation: ~5-10 seconds (LLM-dependent)

**Memory Usage**:
- Base system: ~200 MB
- Per strategy backtest: ~50-100 MB
- Parallel batch (20 strategies): ~2 GB

---

## 🤝 Contributing

Contributions welcome! Key areas:

1. **Walk-forward testing implementation** (high priority)
2. **Complete ML training loop** (high priority)
3. **Paper trading engine** (medium priority)
4. **Additional indicators** (low priority)
5. **Documentation improvements**

Please see `docs/ROADMAP_RETAIL_FOCUSED.md` for priority guidance.

---

## 📚 Documentation

- **`CLAUDE.md`**: Comprehensive project documentation for Claude Code
- **`docs/ROADMAP_RETAIL_FOCUSED.md`**: 12-18 month development roadmap
- **`docs/PRIORITY_MATRIX.md`**: Quick reference for priorities and timelines
- **Inline code documentation**: All modules have detailed docstrings

---

## 🔐 Security & Safety

### For Live Trading (When Implemented)

- ✅ Maximum position size limits per trade
- ✅ Maximum daily trade count (circuit breaker)
- ✅ Maximum daily loss limits
- ✅ Manual approval for first N trades
- ✅ SMS/email alerts on every trade
- ✅ Immediate stop on API failures

**IMPORTANT**: Start with paper trading. Use small amounts ($500-1000) for initial live deployment.

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **SimulatedExchange**: Production-grade backtesting framework
- **CCXT**: Exchange connectivity (free tier)
- **OpenAI/Anthropic/DeepSeek**: LLM providers for autonomous agents
- **Scikit-learn/XGBoost**: ML model implementations
- **SQLAlchemy**: Database ORM

---

## 📞 Contact & Support

- **Issues**: https://github.com/your-username/quantopia/issues
- **Documentation**: See `CLAUDE.md` for detailed technical docs
- **Roadmap**: See `docs/ROADMAP_RETAIL_FOCUSED.md`

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Trading cryptocurrencies and other financial instruments involves substantial risk of loss. Past performance does not guarantee future results. The authors and contributors are not responsible for any financial losses incurred through use of this software.

**Key Risks:**
- Strategies may be overfitted (walk-forward testing pending)
- Backtests may not reflect live performance
- Markets can change, rendering strategies ineffective
- Technical failures can cause unexpected losses

**Always:**
- Start with paper trading
- Use small position sizes
- Never risk more than you can afford to lose
- Understand the strategies you deploy
- Monitor live performance closely

---

**Built with ❤️ for the quant community**

**Current Status: 70% complete | 12-18 months to institutional-grade retail platform**
