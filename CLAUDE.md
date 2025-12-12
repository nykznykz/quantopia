# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Quantopia** is an AI-powered autonomous quantitative research platform that explores the trading strategy space using LLM agents. The system autonomously generates, codes, backtests, and refines trading strategies with minimal human intervention.

### Key Architecture Principles

1. **Agent-First Design**: The system uses specialized LLM agents that make autonomous decisions:
   - **Strategy Agent** (`src/agents/strategy_agent.py`): Analyzes database history and decides what strategies to explore next
   - **ML Quant Agent** (`src/agents/ml_quant_agent.py`): Provides ML models on-demand for hybrid/pure ML strategies
   - **Agent Router** (`src/orchestrator/agent_router.py`): Routes strategy requests through appropriate agent pipelines
   - **Code Generator** (`src/code_generation/code_generator.py`): Synthesizes executable Python classes from strategy metadata
   - **Research Orchestrator** (`src/orchestrator/research_engine.py`): Coordinates full research iterations

2. **Database-Driven Learning**: All strategies, results, and ML models are stored in SQLite (`data/strategies.db`) to enable agents to learn from past explorations and avoid redundant work.

3. **Backtesting Engine**: Uses the `simulated_exchange` framework (separate package in `simulated_exchange/`) for realistic trade execution simulation.

## Development Setup

### Virtual Environment
ALWAYS work within the virtual environment:
```bash
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\activate  # Windows
```

### Installing Dependencies
When encountering missing packages:
1. Add to `requirements.txt` (not just pip install directly)
2. Then install: `pip install -r requirements.txt`

### Configuration
1. Copy configuration templates:
   ```bash
   cp .env.example .env
   cp config/quantopia.yaml.example config/quantopia.yaml
   ```

2. Set required API keys in `.env`:
   - `OPENAI_API_KEY` (required for default LLM operations)
   - Optional: `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `AZURE_OPENAI_KEY`

3. LLM provider can be overridden via `LLM_PROVIDER` env var or config

## Common Commands

### Main CLI Entry Point
```bash
# Initialize configuration
python quantopia.py init

# Run autonomous research session
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

### Phase 1a Example (Agent-First Architecture)
```bash
# Run autonomous research session with agents making all decisions
python examples/phase1a_autonomous_research.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_end_to_end.py
pytest tests/test_orchestrator.py
pytest tests/test_full_flywheel.py

# Run with verbose output
pytest -v -s tests/
```

## Architecture Deep Dive

### Strategy Generation Flow

```
1. Strategy Agent → Analyzes DB, decides what to explore
   ↓
2. Agent Router → Determines if ML is needed
   ↓ (if ML required)
3. ML Quant Agent → Provides or creates ML model
   ↓
4. Code Generator → Synthesizes Python class inheriting BaseStrategy
   ↓
5. Batch Tester → Runs backtest via SimulatedExchange
   ↓
6. Strategy Filter → Approves/rejects based on metrics (Sharpe, drawdown, etc.)
   ↓
7. Database → Stores results for future agent learning
```

### Key Data Flows

- **Strategy Metadata**: Dict with `strategy_name`, `strategy_type`, `indicators`, `entry_rules`, `exit_rules`, `ml_strategy_type`, etc.
- **Backtest Results**: Dict with `metrics` (Sharpe, return, drawdown, win_rate, etc.), `equity_curve`, `trade_history`
- **Agent Decisions**: JSON with `strategy_type`, `rationale`, `logic_type`, `indicators`, `exploration_mode`

### Strategy Types
- `pure_technical`: Traditional indicator-based (RSI, EMA, MACD, etc.)
- `hybrid_ml`: ML predictions combined with technical indicators
- `pure_ml`: Fully ML-driven entry/exit signals

### Logic Types
- `trend_following`: Follow price trends
- `mean_reversion`: Trade oversold/overbought conditions
- `breakout`: Trade price breakouts with volume confirmation
- `momentum`: Capitalize on strong directional moves
- `volatility`: Trade based on volatility regimes

## Critical Code Patterns

### Generated Strategy Structure
All generated strategies inherit from `BaseStrategy` (`src/code_generation/strategy_base.py`):

```python
class MyStrategy(BaseStrategy):
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Pre-calculate ALL indicators on full dataset (not bar-by-bar)"""
        # Return dict of indicator_name -> Series/DataFrame
        pass

    def should_enter(self) -> bool:
        """Use self.indicators to decide entry"""
        # Access current bar's indicators via self.indicators dict
        pass

    def should_exit(self) -> bool:
        """Use self.indicators to decide exit"""
        pass
```

**IMPORTANT**: `calculate_indicators()` receives the FULL historical dataset and should return Series/DataFrames, NOT single values. The backtesting engine extracts current values per bar.

### Database Schema
Key tables in `src/database/schema.py`:
- `strategies`: Core strategy metadata and parameters
- `strategy_code`: Generated Python code
- `backtest_results`: Performance metrics and equity curves
- `ml_models_registry`: ML model versions and metadata
- `portfolio_evaluations`: Multi-strategy portfolio analysis

### Scratchpad Usage
Per user instructions, temporary files and experimental scripts should go in `scratchpad/` folder. Once polished, integrate into appropriate locations.

## Development Phases

### Phase 1 (✓ Complete - ML/DL Foundation)
- Autonomous Strategy Agent with LLM decision-making
- ML Quant Agent for on-demand model provision
- Agent Router for pipeline coordination
- Code Generator for Python synthesis
- Batch backtesting and filtering

### Phase 2 (Pending)
- Critique agent for failure analysis
- Walk-forward testing framework
- Refinement loop
- Enhanced strategy database queries

### Phase 3 (Pending)
- Paper trading module
- Divergence tracking
- Exchange API integration

### Phase 4 (Pending)
- Portfolio optimizer
- Risk engine
- Multi-strategy orchestration

## Important Constraints

### Available Indicators (20 total)
- **Trend**: EMA, SMA, MACD, ADX, EMASlope
- **Momentum**: RSI, Stochastic, MFI, ROC, WilliamsR
- **Volatility**: ATR, BollingerBands, KeltnerChannels, HistoricalVolatility, DonchianChannels
- **Regime**: VolumeZScore, HurstExponent, MarketRegime, TrendStrength, VolatilityRegime

### Filter Criteria (Defaults)
Strategies must meet these thresholds to be approved:
- Sharpe Ratio ≥ 0.5
- Total Return ≥ 5%
- Max Drawdown ≤ 30%
- Number of Trades ≥ 10
- Win Rate ≥ 30%
- Profit Factor ≥ 1.0

Configurable via `.env` or `config/quantopia.yaml`.

## Debugging Tips

### Common Issues

1. **Import errors for `simulated_exchange`**: Ensure it's installed:
   ```bash
   cd simulated_exchange && pip install -e . && cd ..
   ```

2. **LLM API failures**: Check `.env` has valid API keys and correct `LLM_PROVIDER` setting

3. **Database locked**: Close any other processes accessing `data/strategies.db`

4. **Slow backtests**: The `BaseStrategy` pre-calculates indicators on full dataset for O(n) performance vs O(n²) bar-by-bar recalculation

5. **Generated code errors**: Code Generator uses temperature=0.2 for deterministic output. Check `src/code_generation/code_generator.py` for templates.

### Logging
- Default level: INFO
- Detailed logs: `--log-level DEBUG`
- File logging: `logs/quantopia.log` (if enabled in config)
- Structured logging via `structlog`

## Key Files to Understand

When modifying core behavior, review:
- `src/agents/strategy_agent.py` - Autonomous exploration logic and LLM prompts
- `src/orchestrator/agent_router.py` - Agent coordination and decision routing
- `src/orchestrator/research_engine.py` - Full research iteration orchestration
- `src/code_generation/code_generator.py` - Python code synthesis from metadata
- `src/code_generation/strategy_base.py` - Base class all strategies inherit from
- `src/database/manager.py` - All database operations (strategies, results, models)
- `src/backtest/batch_tester.py` - Parallel backtesting coordination
- `src/critique/filter.py` - Strategy approval/rejection logic

## Testing Philosophy

- End-to-end tests in `tests/test_end_to_end.py` validate full pipeline
- `tests/test_full_flywheel.py` tests complete autonomous research loop
- Mock LLM responses in tests to avoid API costs
- Use sample data generation for reproducible tests
