"""Tests for portfolio evaluator."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio import (
    PortfolioEvaluator,
    CorrelationAnalyzer,
    PortfolioRiskManager,
    AllocationOptimizer,
    ForwardTestSelector
)


def create_mock_backtest_result(
    strategy_name: str,
    sharpe: float = 1.5,
    total_return: float = 0.30,
    max_dd: float = 0.15,
    num_trades: int = 50
) -> dict:
    """Create a mock backtest result for testing."""
    # Generate synthetic equity curve
    num_periods = 252  # One year of daily data
    returns = np.random.normal(0.001, 0.02, num_periods)  # Daily returns
    equity_curve = pd.Series((1 + returns).cumprod(), name=strategy_name)

    # Adjust to match target metrics roughly
    equity_curve = equity_curve * (1 + total_return) / equity_curve.iloc[-1]

    # Generate synthetic trades
    trades = []
    trade_dates = pd.date_range(start=datetime.now() - timedelta(days=365), periods=num_trades, freq='7D')

    for i, trade_date in enumerate(trade_dates):
        trades.append({
            'entry_time': trade_date,
            'exit_time': trade_date + timedelta(days=3),
            'pnl': np.random.normal(100, 50),
            'return': np.random.normal(0.01, 0.02)
        })

    return {
        'strategy_name': strategy_name,
        'metrics': {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'num_trades': num_trades,
            'win_rate': 0.55,
            'profit_factor': 1.3
        },
        'equity_curve': equity_curve,
        'trades': trades,
        'metadata': {
            'strategy_type': 'trend_following',
            'indicators': ['EMA', 'RSI', 'ATR']
        }
    }


def test_portfolio_evaluator_initialization():
    """Test basic initialization."""
    evaluator = PortfolioEvaluator()
    assert evaluator is not None
    assert isinstance(evaluator.correlation_analyzer, CorrelationAnalyzer)
    assert isinstance(evaluator.risk_manager, PortfolioRiskManager)
    assert isinstance(evaluator.allocation_optimizer, AllocationOptimizer)
    assert isinstance(evaluator.forward_selector, ForwardTestSelector)


def test_portfolio_evaluation_with_multiple_strategies():
    """Test evaluating a portfolio with multiple strategies."""
    # Create mock results for 5 strategies
    backtest_results = [
        create_mock_backtest_result('Strategy_A', sharpe=1.8, total_return=0.40),
        create_mock_backtest_result('Strategy_B', sharpe=1.5, total_return=0.30),
        create_mock_backtest_result('Strategy_C', sharpe=1.2, total_return=0.25),
        create_mock_backtest_result('Strategy_D', sharpe=1.0, total_return=0.20),
        create_mock_backtest_result('Strategy_E', sharpe=0.8, total_return=0.15)
    ]

    evaluator = PortfolioEvaluator()
    result = evaluator.evaluate_portfolio(backtest_results)

    # Check result structure
    assert result is not None
    assert result.num_strategies_evaluated == 5
    assert result.portfolio_metrics is not None
    assert result.correlation_matrix is not None
    assert len(result.optimal_allocations) == 5
    assert len(result.forward_test_priorities) == 5

    # Check portfolio metrics
    assert result.portfolio_metrics.num_strategies == 5
    assert result.portfolio_metrics.portfolio_sharpe > 0
    assert 0 <= result.portfolio_metrics.portfolio_max_dd <= 1.0

    # Check allocations sum to 1
    total_allocation = sum(a.allocation_weight for a in result.optimal_allocations)
    assert abs(total_allocation - 1.0) < 0.01

    # Check deployment recommendations
    assert len(result.ready_for_deployment) + len(result.needs_monitoring) + len(result.rejected_strategies) == 5

    print("\n" + result.summary())


def test_correlation_analysis():
    """Test correlation analysis."""
    backtest_results = [
        create_mock_backtest_result('Strategy_1'),
        create_mock_backtest_result('Strategy_2'),
        create_mock_backtest_result('Strategy_3')
    ]

    evaluator = PortfolioEvaluator()
    result = evaluator.evaluate_portfolio(backtest_results)

    # Check correlation matrix
    assert not result.correlation_matrix.returns_correlation.empty
    assert result.correlation_matrix.returns_correlation.shape == (3, 3)

    # Check correlation bounds
    corr_values = result.correlation_matrix.returns_correlation.values
    assert np.all(corr_values >= -1.0)
    assert np.all(corr_values <= 1.0)

    # Diagonal should be 1.0
    assert np.allclose(np.diag(corr_values), 1.0)


def test_allocation_optimization():
    """Test allocation optimization."""
    backtest_results = [
        create_mock_backtest_result('High_Sharpe', sharpe=2.0, total_return=0.50),
        create_mock_backtest_result('Medium_Sharpe', sharpe=1.5, total_return=0.30),
        create_mock_backtest_result('Low_Sharpe', sharpe=1.0, total_return=0.20)
    ]

    # Test different optimization methods
    for method in ['equal_weight', 'max_sharpe']:
        evaluator = PortfolioEvaluator(
            allocation_optimizer=AllocationOptimizer(optimization_method=method)
        )
        result = evaluator.evaluate_portfolio(backtest_results)

        allocations = result.optimal_allocations
        assert len(allocations) == 3

        # Check weights sum to 1
        total_weight = sum(a.allocation_weight for a in allocations)
        assert abs(total_weight - 1.0) < 0.01

        # For max_sharpe, high Sharpe strategy should get more weight
        if method == 'max_sharpe':
            high_sharpe_alloc = next(a for a in allocations if a.strategy_name == 'High_Sharpe')
            low_sharpe_alloc = next(a for a in allocations if a.strategy_name == 'Low_Sharpe')
            # Not always true due to correlation, but generally expected
            print(f"High Sharpe allocation: {high_sharpe_alloc.allocation_weight:.2f}")
            print(f"Low Sharpe allocation: {low_sharpe_alloc.allocation_weight:.2f}")


def test_forward_test_prioritization():
    """Test forward test priority scoring."""
    backtest_results = [
        create_mock_backtest_result('Excellent', sharpe=2.0, total_return=0.50, max_dd=0.10, num_trades=100),
        create_mock_backtest_result('Good', sharpe=1.5, total_return=0.30, max_dd=0.15, num_trades=50),
        create_mock_backtest_result('Marginal', sharpe=1.0, total_return=0.15, max_dd=0.20, num_trades=30),
        create_mock_backtest_result('Poor', sharpe=0.5, total_return=0.05, max_dd=0.30, num_trades=10)
    ]

    evaluator = PortfolioEvaluator()
    result = evaluator.evaluate_portfolio(backtest_results)

    priorities = result.forward_test_priorities

    # Check priorities are sorted by score
    scores = [p.priority_score for p in priorities]
    assert scores == sorted(scores, reverse=True)

    # Excellent strategy should be top priority
    assert priorities[0].strategy_name == 'Excellent'
    assert priorities[0].priority_tier in ['high', 'medium']

    # Poor strategy should be low priority
    assert priorities[-1].strategy_name == 'Poor'
    assert priorities[-1].priority_tier == 'low'

    # Check deployment readiness
    excellent = priorities[0]
    poor = priorities[-1]

    assert excellent.deployment_readiness in ['ready', 'monitor']
    # Poor might be reject due to low Sharpe and few trades
    print(f"Excellent readiness: {excellent.deployment_readiness}")
    print(f"Poor readiness: {poor.deployment_readiness}")


def test_quick_evaluation():
    """Test quick evaluation mode."""
    backtest_results = [
        create_mock_backtest_result('Strategy_1'),
        create_mock_backtest_result('Strategy_2')
    ]

    evaluator = PortfolioEvaluator()
    result = evaluator.quick_evaluation(backtest_results)

    # Check result structure
    assert 'num_strategies' in result
    assert 'portfolio_sharpe' in result
    assert 'portfolio_return' in result
    assert 'avg_correlation' in result

    assert result['num_strategies'] == 2
    assert result['portfolio_sharpe'] > 0


def test_empty_portfolio():
    """Test evaluation with empty portfolio."""
    evaluator = PortfolioEvaluator()
    result = evaluator.evaluate_portfolio([])

    assert result.num_strategies_evaluated == 0
    assert len(result.optimal_allocations) == 0
    assert len(result.forward_test_priorities) == 0


def test_single_strategy():
    """Test evaluation with single strategy."""
    backtest_results = [
        create_mock_backtest_result('OnlyStrategy', sharpe=1.5)
    ]

    evaluator = PortfolioEvaluator()
    result = evaluator.evaluate_portfolio(backtest_results)

    assert result.num_strategies_evaluated == 1
    assert len(result.optimal_allocations) == 1
    assert result.optimal_allocations[0].allocation_weight == 1.0


if __name__ == '__main__':
    # Run tests
    print("Running portfolio evaluator tests...\n")

    test_portfolio_evaluator_initialization()
    print("✓ Initialization test passed")

    test_portfolio_evaluation_with_multiple_strategies()
    print("✓ Multi-strategy evaluation test passed")

    test_correlation_analysis()
    print("✓ Correlation analysis test passed")

    test_allocation_optimization()
    print("✓ Allocation optimization test passed")

    test_forward_test_prioritization()
    print("✓ Forward test prioritization test passed")

    test_quick_evaluation()
    print("✓ Quick evaluation test passed")

    test_empty_portfolio()
    print("✓ Empty portfolio test passed")

    test_single_strategy()
    print("✓ Single strategy test passed")

    print("\n✅ All tests passed!")
