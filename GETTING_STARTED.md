# Getting Started Checklist

## Prerequisites

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] OpenAI API key (or Azure OpenAI/DeepSeek credentials)

## Setup Steps

### 1. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # Unix/Mac
# or
.\venv\Scripts\activate   # Windows

# Install project dependencies
pip install -r requirements.txt

# Install simulated_exchange
cd simulated_exchange
pip install -e .
cd ..
```

### 2. Configure API Keys

```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# For Azure OpenAI (alternative)
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# For DeepSeek (alternative)
export DEEPSEEK_API_KEY="..."
```

### 3. Verify Installation

```bash
# Test indicator imports
python -c "from src.indicators import INDICATOR_REGISTRY; print(f'{len(INDICATOR_REGISTRY)} indicators loaded')"

# Test LLM client (requires API key)
python -c "from src.strategy_generation import LLMClient; client = LLMClient(); print('LLM client initialized')"
```

## Quick Test

### Option 1: Run Simple Example

```bash
python examples/simple_strategy_generation.py
```

This will:
1. Generate a mean reversion strategy using GPT-4
2. Convert it to executable Python code
3. Download 6 months of BTC-USD data from Binance
4. Run a backtest
5. Apply acceptance criteria filter
6. Save approved strategies to `strategies/approved/`

**Expected output**: Strategy name, metrics, and APPROVED/REJECTED status

### Option 2: Run Full End-to-End Test

```bash
python tests/test_end_to_end.py
```

This runs the complete pipeline with detailed logging.

### Option 3: Use Python REPL

```python
from src.indicators import INDICATOR_REGISTRY
from src.strategy_generation import StrategyGenerator, LLMClient
from src.code_generation.code_generator import CodeGenerator
from src.backtest import BacktestRunner
from src.critique import StrategyFilter
from datetime import datetime, timedelta

# 1. Initialize components
llm_client = LLMClient(provider="openai", model="gpt-4")
generator = StrategyGenerator(llm_client, list(INDICATOR_REGISTRY.keys()))

# 2. Generate a strategy
strategy = generator.generate_strategy(strategy_type="mean_reversion")
print(f"Generated: {strategy['strategy_name']}")

# 3. Convert to code
code_gen = CodeGenerator()
code = code_gen.generate_strategy_class(strategy)
print(f"Code length: {len(code)} chars")

# 4. Load data
runner = BacktestRunner(initial_capital=10000.0)
data = runner.load_data(
    symbol="BTC-USD",
    exchange="binance",
    timeframe="1h",
    start_date=datetime.now() - timedelta(days=180),
    end_date=datetime.now()
)
print(f"Data loaded: {len(data)} candles")

# 5. Backtest
results = runner.run_from_code(code, data, "BTC-USD", strategy['strategy_name'])
print(f"Backtest status: {results['status']}")

if results['status'] == 'success':
    metrics = results['metrics']
    print(f"Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Max DD: {metrics['max_drawdown']*100:.2f}%")

# 6. Filter
filter_obj = StrategyFilter()
filter_result = filter_obj.filter_strategy(results)
print(f"\nResult: {filter_result.classification.upper()}")
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'simulated_exchange'"**
   - Solution: `cd simulated_exchange && pip install -e . && cd ..`

2. **"OpenAI API key not found"**
   - Solution: `export OPENAI_API_KEY="your-key-here"`

3. **"Failed to download data from Binance"**
   - Possible causes: Network issues, invalid symbol, ccxt not installed
   - Solution: Check internet connection, verify symbol format ("BTC-USD" not "BTCUSD")

4. **"Strategy backtest failed"**
   - Common cause: Indicator calculation error (not enough data)
   - Solution: Use longer historical window (e.g., 180+ days for 1h data)

5. **LLM generation timeout**
   - Solution: Increase timeout in llm_client.py or try a different model/provider

### Checking Logs

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python examples/simple_strategy_generation.py
```

## Directory Structure After First Run

After running an example, you should see:

```
ai_quant_research/
├── data/
│   └── historical/          # Downloaded OHLCV data
│       └── binance/
│           └── BTC-USD_1h.csv
├── strategies/
│   ├── generated/           # All generated strategy code
│   │   └── SomeStrategy.py
│   └── approved/            # Only approved strategies
│       └── ApprovedStrategy.py
└── results/
    └── backtests/           # Backtest results (CSV + JSON)
        ├── SomeStrategy_20241210_120000_trades.csv
        ├── SomeStrategy_20241210_120000_equity.csv
        └── SomeStrategy_20241210_120000_summary.json
```

## Next Steps

Once you've successfully run a test:

1. **Generate a batch** of 10+ strategies
2. **Review generated code** in `strategies/generated/`
3. **Analyze backtest results** in `results/backtests/`
4. **Adjust filter criteria** in `FilterCriteria` to tune acceptance thresholds
5. **Experiment with different**:
   - Strategy types (mean_reversion, trend_following, momentum, etc.)
   - Timeframes (1h, 4h, 1d)
   - Symbols (BTC-USD, ETH-USD, etc.)
   - LLM temperature (0.7 = conservative, 0.9 = creative)

## Batch Testing Example

```python
from src.strategy_generation import StrategyGenerator, LLMClient
from src.code_generation.code_generator import CodeGenerator
from src.backtest import BacktestRunner, BatchTester
from src.critique import StrategyFilter
from src.indicators import INDICATOR_REGISTRY
from datetime import datetime, timedelta

# Generate 10 strategies
llm_client = LLMClient(provider="openai", model="gpt-4")
generator = StrategyGenerator(llm_client, list(INDICATOR_REGISTRY.keys()))

strategies = generator.generate_batch(num_strategies=10, temperature=0.8)
print(f"Generated {len(strategies)} strategies")

# Convert to code
code_gen = CodeGenerator()
codes = [code_gen.generate_strategy_class(s) for s in strategies]
names = [s['strategy_name'] for s in strategies]

# Load data once
runner = BacktestRunner(initial_capital=10000.0)
data = runner.load_data("BTC-USD", "binance", "1h",
                       datetime.now() - timedelta(days=180),
                       datetime.now())

# Batch backtest
batch_tester = BatchTester(initial_capital=10000.0)
results = batch_tester.run_batch(codes, names, data, "BTC-USD")

# Batch filter
strategy_filter = StrategyFilter()
filter_results = strategy_filter.filter_batch(results)

# Summary
approved = [r for r in filter_results if r.passed]
print(f"\nApproved: {len(approved)}/{len(filter_results)}")

for result in filter_results:
    status = "✓" if result.passed else "✗"
    print(f"{status} {result.strategy_name}: {result.classification}")
```

## Performance Expectations

On a typical machine with GPT-4:

- **Strategy Generation**: 10-30 seconds per strategy
- **Code Generation**: < 1 second per strategy
- **Data Loading**: 5-10 seconds (first time, cached thereafter)
- **Backtesting**: 1-5 seconds per strategy (180 days of 1h data)

**Total for 10 strategies**: ~5-10 minutes

## Cost Estimates (OpenAI GPT-4)

- **Per strategy generation**: ~$0.01-0.05 (depends on prompt length)
- **10 strategies**: ~$0.10-0.50
- **100 strategies**: ~$1-5

Using DeepSeek can be significantly cheaper (~10x less).

## Support

If you encounter issues:

1. Check `GETTING_STARTED.md` (this file)
2. Review `PHASE_1_COMPLETE.md` for architecture details
3. Check logs in console output
4. Review generated code in `strategies/generated/` for debugging

## Ready to Go?

- [x] Phase 1 MVP complete
- [ ] API key configured
- [ ] Dependencies installed
- [ ] First strategy generated
- [ ] First backtest completed
- [ ] First strategy approved

**Next**: Try generating your first strategy with `python examples/simple_strategy_generation.py`!
