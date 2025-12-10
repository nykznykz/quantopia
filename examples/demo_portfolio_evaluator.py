"""
Demonstration of Portfolio Evaluator usage.

This script shows how to use the Portfolio Evaluator to analyze
multiple trading strategies at the portfolio level.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Example of how to use the Portfolio Evaluator
# (Requires running actual backtests first)

print("""
╔════════════════════════════════════════════════════════════════╗
║           Portfolio Evaluator - Usage Example                 ║
╚════════════════════════════════════════════════════════════════╝

The Portfolio Evaluator (Agent 5) is now implemented!

## Quick Start:

```python
from src.portfolio import PortfolioEvaluator
from src.backtest import BatchTester

# 1. Run backtests on multiple strategies
batch_tester = BatchTester()
backtest_results = batch_tester.run_batch(
    strategy_codes=[...],  # Your strategy code
    strategy_names=[...],  # Strategy names
    data=historical_data,   # OHLCV data
    symbol='BTC-USD'
)

# 2. Evaluate at portfolio level
evaluator = PortfolioEvaluator()
portfolio_result = evaluator.evaluate_portfolio(
    backtest_results=backtest_results,
    deployed_strategies=['Strategy_1']  # Optional: currently deployed
)

# 3. View results
print(portfolio_result.summary())

# 4. Get allocation recommendations
for allocation in portfolio_result.optimal_allocations:
    print(f"{allocation.strategy_name}: {allocation.allocation_weight*100:.1f}%")
    print(f"  Rationale: {allocation.allocation_rationale}")

# 5. Get forward test priorities
for priority in portfolio_result.forward_test_priorities[:5]:
    print(f"{priority.strategy_name}: Score {priority.priority_score:.1f}")
    print(f"  Status: {priority.deployment_readiness}")
    print(f"  Reasons: {', '.join(priority.reasons)}")

# 6. Check deployment recommendations
ready = portfolio_result.ready_for_deployment
monitor = portfolio_result.needs_monitoring
reject = portfolio_result.rejected_strategies

print(f"✅ Ready for deployment: {len(ready)}")
print(f"⚠️  Needs monitoring: {len(monitor)}")
print(f"❌ Rejected: {len(reject)}")
```

## Features:

✅ **Correlation Analysis**
   - Returns correlation matrix
   - Trade overlap detection
   - Redundant strategy identification

✅ **Portfolio Optimization**
   - Equal weight allocation
   - Max Sharpe ratio (Markowitz)
   - Risk parity (equal risk contribution)

✅ **Risk Management**
   - Portfolio-level Sharpe, return, drawdown
   - Diversification ratio
   - Concentration risk (HHI)
   - Marginal risk contribution

✅ **Forward Test Selection**
   - Multi-factor scoring system
   - Performance + Portfolio Fit + Novelty + Risk
   - Deployment readiness classification

## Components:

- **CorrelationAnalyzer**: Analyzes inter-strategy correlations
- **PortfolioRiskManager**: Portfolio-level risk metrics
- **AllocationOptimizer**: Position sizing optimization
- **ForwardTestSelector**: Priority ranking for deployment
- **PortfolioEvaluator**: Main facade orchestrating all components

## Configuration Options:

```python
from src.portfolio import (
    PortfolioEvaluator,
    CorrelationAnalyzer,
    AllocationOptimizer,
    PortfolioRiskManager,
    ForwardTestSelector
)

# Custom configuration
evaluator = PortfolioEvaluator(
    correlation_analyzer=CorrelationAnalyzer(
        min_overlap_threshold=0.7
    ),
    allocation_optimizer=AllocationOptimizer(
        optimization_method="max_sharpe",  # or "equal_weight", "risk_parity"
        allow_leverage=False,
        max_leverage=1.0
    ),
    risk_manager=PortfolioRiskManager(
        max_portfolio_dd=0.20,
        max_concentration=0.40,
        min_diversification=0.7
    ),
    forward_selector=ForwardTestSelector(
        min_backtest_sharpe=1.0,
        max_correlation_with_deployed=0.6
    )
)
```

## Integration with Existing Code:

The Portfolio Evaluator integrates seamlessly with your existing
BacktestRunner and BatchTester:

```python
from src.backtest import BacktestRunner, BatchTester
from src.critique import StrategyFilter
from src.portfolio import PortfolioEvaluator

# Step 1: Generate and backtest strategies
# (using your existing workflow)

# Step 2: Filter individual strategies
filter = StrategyFilter()
filter_results = filter.filter_batch(backtest_results)
approved = [r for r, f in zip(backtest_results, filter_results) if f.passed]

# Step 3: Portfolio-level evaluation (NEW!)
evaluator = PortfolioEvaluator()
portfolio_result = evaluator.evaluate_portfolio(approved)

# Now you have portfolio-aware recommendations!
```

## Next Steps:

1. Test the Portfolio Evaluator with your existing strategies
2. Implement Phase 2: Strategy Database (SQLite)
3. Implement Phase 3: Core Flywheel Orchestrator

See the implementation plan in:
/Users/user/.claude/plans/sequential-doodling-bengio.md

""")

print("\n" + "="*60)
print("Phase 1 Complete: Portfolio Evaluator (Agent 5) ✅")
print("="*60)
print("\nAll 5 components implemented:")
print("  models.py          - Data structures")
print("  correlation.py     - Correlation analyzer")
print("  risk_manager.py    - Risk management")
print("  allocation.py      - Allocation optimizer")
print("  forward_selector.py - Forward test selector")
print("  evaluator.py       - Main facade")
print("\nTo test:")
print("  python tests/test_portfolio_evaluator.py")
print("\nOr integrate with your existing backtest workflow!")
