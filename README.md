# OmniAlpha - AI-Augmented Quant Research Engine

An AI-powered quantitative research system that automates the full lifecycle of systematic strategy development: indicator discovery, strategy generation, coding, backtesting, refinement, and deployment.

## Overview

OmniAlpha leverages LLM agents (GPT-4) to accelerate hypothesis generation and validation while maintaining strict quant discipline. The system uses the `simulated_exchange` framework for backtesting and paper trading.

## Features

- **Automated Strategy Generation**: GPT-4 generates testable trading strategies from indicator combinations
- **Code Synthesis**: Automatically converts strategy descriptions into executable Python classes
- **Backtesting Engine**: Leverages SimulatedExchange for realistic execution simulation
- **Strategy Refinement**: AI-powered critique and improvement loop
- **Walk-Forward Testing**: Out-of-sample validation and robustness testing
- **Paper Trading**: Real-time forward testing with live market data
- **Portfolio Management**: Multi-strategy orchestration and risk management

## Architecture

```
OmniAlpha = Indicator Pool + LLM Strategy Generator + SimulatedExchange
```

- **Indicator Pool**: 20-30 validated technical indicators across trend, momentum, volatility, and regime categories
- **LLM Layer**: GPT-4 for strategy generation and refinement
- **SimulatedExchange**: Comprehensive backtesting and paper trading engine (existing framework)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install simulated_exchange
cd simulated_exchange
pip install -e .
cd ..
```

## Quick Start

### 1. Set up environment

```bash
# Activate virtual environment
source venv/bin/activate  # On Unix/Mac
# or
.\venv\Scripts\activate  # On Windows

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run a simple example

```bash
# Generate and backtest a single strategy
python examples/simple_strategy_generation.py

# Run end-to-end test
python tests/test_end_to_end.py
```

### 3. Use in your code

```python
from src.indicators import INDICATOR_REGISTRY
from src.strategy_generation import StrategyGenerator, LLMClient
from src.code_generation.code_generator import CodeGenerator
from src.backtest import BacktestRunner
from src.critique import StrategyFilter, FilterCriteria

# Initialize LLM client
llm_client = LLMClient(provider="openai", model="gpt-4")

# Generate strategy
generator = StrategyGenerator(
    llm_client=llm_client,
    available_indicators=list(INDICATOR_REGISTRY.keys())
)
strategy = generator.generate_strategy(strategy_type="mean_reversion")

# Convert to code
code_gen = CodeGenerator()
strategy_code = code_gen.generate_strategy_class(strategy)

# Backtest
runner = BacktestRunner(initial_capital=10000.0)
data = runner.load_data(symbol="BTC-USD", exchange="binance", timeframe="1h")
results = runner.run_from_code(strategy_code, data, symbol="BTC-USD")

# Filter
strategy_filter = StrategyFilter()
filter_result = strategy_filter.filter_strategy(results)

if filter_result.passed:
    print(f"✓ Strategy approved: {filter_result.classification}")
else:
    print(f"✗ Strategy rejected")
```

## Project Structure

```
ai_quant_research/
├── config/                 # Configuration files
├── src/
│   ├── indicators/         # Technical indicator library
│   ├── strategy_generation/# LLM-based strategy generator
│   ├── code_generation/    # Strategy code synthesizer
│   ├── backtest/           # Backtesting orchestration
│   └── utils/              # Utilities
├── simulated_exchange/     # Backtesting engine (existing)
├── strategies/             # Generated and approved strategies
├── data/                   # Historical market data
├── results/                # Backtest results
└── scratchpad/             # Temporary files
```

## Development Phases

- **Phase 1 (MVP)**: Strategy generation + backtesting loop
- **Phase 2**: Refinement agent + robustness testing
- **Phase 3**: Forward testing + deployment
- **Phase 4**: Portfolio construction + risk engine

## Success Criteria

- Generate ≥10 strategies per batch
- Backtest throughput ≥100 strategies/hour
- ≥5 strategies pass robustness filters per 500 generated
- Forward-test correlation ≥0.6 vs backtest
- Max drawdown in live < 1.5× historical DD

## License

MIT License

## Component Status

### Phase 1 (MVP) - ✅ Complete
- ✅ Indicator Pool (20 indicators across 4 categories)
- ✅ LLM Strategy Generator (OpenAI/Azure/DeepSeek support)
- ✅ Code Generator (strategy metadata → Python class)
- ✅ Backtest Runner (orchestrates SimulatedExchange)
- ✅ Strategy Filter (basic acceptance criteria)
- ✅ End-to-end pipeline test

### Phase 2 - 🚧 Pending
- ⏳ Critique agent for failure analysis
- ⏳ Walk-forward testing framework
- ⏳ Refinement loop
- ⏳ Strategy database

### Phase 3 - 🚧 Pending
- ⏳ Paper trading module
- ⏳ Divergence tracking
- ⏳ Exchange API integration

### Phase 4 - 🚧 Pending
- ⏳ Portfolio optimizer
- ⏳ Risk engine
- ⏳ Multi-strategy orchestration

## Available Indicators

**Trend** (5): EMA, SMA, MACD, ADX, EMASlope

**Momentum** (5): RSI, Stochastic, MFI, ROC, WilliamsR

**Volatility** (5): ATR, BollingerBands, KeltnerChannels, HistoricalVolatility, DonchianChannels

**Regime** (5): VolumeZScore, HurstExponent, MarketRegime, TrendStrength, VolatilityRegime

## Configuration

### LLM Providers

```python
# OpenAI
llm_client = LLMClient(provider="openai", model="gpt-4", api_key="...")

# Azure OpenAI
llm_client = LLMClient(
    provider="azure",
    model="gpt-4",
    api_key="...",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01"
)

# DeepSeek
llm_client = LLMClient(provider="deepseek", model="deepseek-chat", api_key="...")
```

### Filter Criteria

```python
criteria = FilterCriteria(
    min_total_return=0.05,      # 5% minimum return
    min_sharpe_ratio=0.5,       # Sharpe >= 0.5
    max_drawdown=0.30,          # Max 30% drawdown
    min_num_trades=10,          # At least 10 trades
    min_win_rate=0.30,          # 30% win rate minimum
    min_profit_factor=1.0       # Profit factor >= 1.0
)
```
