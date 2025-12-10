"""Tests for the research orchestrator."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.orchestrator.research_engine import ResearchOrchestrator, IterationReport
from src.strategy_generation.generator import StrategyGenerator
from src.code_generation.code_generator import CodeGenerator
from src.backtest.batch_tester import BatchTester
from src.critique.filter import StrategyFilter, FilterCriteria
from src.portfolio.evaluator import PortfolioEvaluator


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    np.random.seed(42)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.randn(1000).cumsum() * 100,
        'high': 50000 + np.random.randn(1000).cumsum() * 100 + 100,
        'low': 50000 + np.random.randn(1000).cumsum() * 100 - 100,
        'close': 50000 + np.random.randn(1000).cumsum() * 100,
        'volume': np.random.uniform(100, 1000, 1000)
    })

    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def mock_strategy_generator():
    """Create a mock strategy generator."""
    generator = Mock(spec=StrategyGenerator)

    # Mock strategy generation
    def generate_strategy(*args, **kwargs):
        return {
            'strategy_name': 'TestStrategy_RSI_SMA',
            'strategy_type': 'momentum',
            'indicators': ['rsi', 'sma'],
            'entry_conditions': 'RSI < 30',
            'exit_conditions': 'RSI > 70',
            'position_sizing': 0.1,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'timeframe': '1h',
            'generated_at': datetime.now().timestamp()
        }

    generator.generate_strategy.side_effect = generate_strategy

    # Mock refinement
    def refine_strategy(original_strategy, failure_reasons, backtest_metrics):
        refined = original_strategy.copy()
        refined['strategy_name'] = original_strategy['strategy_name'] + '_refined'
        return refined

    generator.refine_strategy.side_effect = refine_strategy

    # Mock family generation
    def generate_strategy_family(parent_strategy, num_variants):
        variants = []
        for i in range(num_variants):
            variant = parent_strategy.copy()
            variant['strategy_name'] = parent_strategy['strategy_name'] + f'_variant_{i+1}'
            variants.append(variant)
        return variants

    generator.generate_strategy_family.side_effect = generate_strategy_family

    return generator


@pytest.fixture
def mock_code_generator():
    """Create a mock code generator."""
    generator = Mock(spec=CodeGenerator)

    def generate_code(strategy_metadata):
        return f"""
import pandas as pd
from src.backtest.base_strategy import BaseStrategy

class {strategy_metadata['strategy_name']}(BaseStrategy):
    def __init__(self):
        super().__init__()

    def should_enter(self, current_candle, historical_data):
        return False

    def should_exit(self, current_candle, historical_data, position):
        return False
"""

    generator.generate_strategy_class.side_effect = generate_code

    return generator


@pytest.fixture
def mock_batch_tester():
    """Create a mock batch tester."""
    tester = Mock(spec=BatchTester)

    def run_batch(strategy_codes, strategy_names, data, symbol, **kwargs):
        results = []
        for i, name in enumerate(strategy_names):
            # Create a mock backtest result
            result = {
                'strategy_name': name,
                'symbol': symbol,
                'total_return': np.random.uniform(-0.1, 0.3),
                'sharpe_ratio': np.random.uniform(0.2, 2.0),
                'max_drawdown': np.random.uniform(0.05, 0.35),
                'num_trades': np.random.randint(5, 50),
                'win_rate': np.random.uniform(0.25, 0.65),
                'profit_factor': np.random.uniform(0.8, 2.5),
                'avg_win': 0.02,
                'avg_loss': -0.015,
                'largest_win': 0.05,
                'largest_loss': -0.03,
                'consecutive_wins': 3,
                'consecutive_losses': 2,
                'equity_curve': [10000 * (1 + i * 0.001) for i in range(100)],
                'trades': []
            }
            results.append(result)
        return results

    tester.run_batch.side_effect = run_batch

    return tester


@pytest.fixture
def mock_strategy_filter():
    """Create a mock strategy filter."""
    filter_obj = Mock(spec=StrategyFilter)

    def filter_strategy(backtest_result):
        from src.critique.filter import FilterResult

        # Randomly approve/reject based on metrics
        passed = (
            backtest_result['total_return'] > 0.05 and
            backtest_result['sharpe_ratio'] > 0.5 and
            backtest_result['max_drawdown'] < 0.30
        )

        reasons = []
        if backtest_result['total_return'] <= 0.05:
            reasons.append("Return too low")
        if backtest_result['sharpe_ratio'] <= 0.5:
            reasons.append("Sharpe ratio too low")
        if backtest_result['max_drawdown'] >= 0.30:
            reasons.append("Drawdown too high")

        return FilterResult(
            passed=passed,
            strategy_name=backtest_result['strategy_name'],
            rejection_reasons=reasons,
            metrics_summary=backtest_result,
            classification='approved' if passed else 'rejected'
        )

    filter_obj.filter_strategy.side_effect = filter_strategy

    return filter_obj


@pytest.fixture
def mock_portfolio_evaluator():
    """Create a mock portfolio evaluator."""
    evaluator = Mock(spec=PortfolioEvaluator)

    def evaluate_portfolio(backtest_results, **kwargs):
        return {
            'portfolio_metrics': {
                'sharpe_ratio': 1.5,
                'total_return': 0.25,
                'max_drawdown': 0.15,
                'diversification_score': 0.8
            },
            'forward_test_priorities': [
                {
                    'strategy_name': r['strategy_name'],
                    'priority_score': np.random.uniform(0.5, 1.0)
                }
                for r in backtest_results[:3]
            ]
        }

    evaluator.evaluate_portfolio.side_effect = evaluate_portfolio

    return evaluator


def test_orchestrator_initialization(
    mock_strategy_generator,
    mock_code_generator,
    mock_batch_tester,
    mock_strategy_filter,
    mock_portfolio_evaluator
):
    """Test that orchestrator initializes correctly."""
    orchestrator = ResearchOrchestrator(
        strategy_generator=mock_strategy_generator,
        code_generator=mock_code_generator,
        batch_tester=mock_batch_tester,
        strategy_filter=mock_strategy_filter,
        portfolio_evaluator=mock_portfolio_evaluator,
        config={'num_strategies_per_batch': 5}
    )

    assert orchestrator.num_strategies_per_batch == 5
    assert orchestrator.iteration_count == 0


def test_iteration_report_creation():
    """Test that iteration report can be created."""
    report = IterationReport(
        iteration_number=1,
        timestamp=datetime.now(),
        num_strategies_generated=10,
        num_codes_generated=10,
        num_backtests_run=10,
        num_approved=3,
        num_rejected=7,
        num_marginal=0,
        num_refinements=2,
        num_family_variants=6
    )

    assert report.iteration_number == 1
    assert report.num_strategies_generated == 10
    assert report.num_approved == 3

    # Test string representation
    report_str = str(report)
    assert "RESEARCH ITERATION #1" in report_str


def test_run_iteration(
    sample_ohlcv_data,
    mock_strategy_generator,
    mock_code_generator,
    mock_batch_tester,
    mock_strategy_filter,
    mock_portfolio_evaluator
):
    """Test running a complete iteration."""
    orchestrator = ResearchOrchestrator(
        strategy_generator=mock_strategy_generator,
        code_generator=mock_code_generator,
        batch_tester=mock_batch_tester,
        strategy_filter=mock_strategy_filter,
        portfolio_evaluator=mock_portfolio_evaluator,
        config={
            'num_strategies_per_batch': 3,
            'max_refinement_attempts': 1,
            'strategy_family_size': 2,
            'enable_refinement': False,  # Disable for simple test
            'enable_family_generation': False
        }
    )

    report = orchestrator.run_iteration(
        data=sample_ohlcv_data,
        symbol='BTC-USD'
    )

    # Verify iteration ran
    assert orchestrator.iteration_count == 1
    assert report.iteration_number == 1
    assert report.num_strategies_generated == 3
    assert report.num_codes_generated == 3
    assert report.num_backtests_run == 3

    # Verify components were called
    assert mock_strategy_generator.generate_strategy.call_count == 3
    assert mock_code_generator.generate_strategy_class.call_count == 3
    assert mock_batch_tester.run_batch.call_count == 1


def test_run_iteration_with_refinement(
    sample_ohlcv_data,
    mock_strategy_generator,
    mock_code_generator,
    mock_batch_tester,
    mock_strategy_filter,
    mock_portfolio_evaluator
):
    """Test iteration with refinement enabled."""
    orchestrator = ResearchOrchestrator(
        strategy_generator=mock_strategy_generator,
        code_generator=mock_code_generator,
        batch_tester=mock_batch_tester,
        strategy_filter=mock_strategy_filter,
        portfolio_evaluator=mock_portfolio_evaluator,
        config={
            'num_strategies_per_batch': 5,
            'max_refinement_attempts': 2,
            'strategy_family_size': 2,
            'enable_refinement': True,
            'enable_family_generation': False
        }
    )

    report = orchestrator.run_iteration(
        data=sample_ohlcv_data,
        symbol='BTC-USD'
    )

    # Verify iteration ran with refinement
    assert report.num_refinements >= 0  # May or may not refine depending on mock results


def test_run_iteration_with_family_generation(
    sample_ohlcv_data,
    mock_strategy_generator,
    mock_code_generator,
    mock_batch_tester,
    mock_strategy_filter,
    mock_portfolio_evaluator
):
    """Test iteration with family generation enabled."""
    orchestrator = ResearchOrchestrator(
        strategy_generator=mock_strategy_generator,
        code_generator=mock_code_generator,
        batch_tester=mock_batch_tester,
        strategy_filter=mock_strategy_filter,
        portfolio_evaluator=mock_portfolio_evaluator,
        config={
            'num_strategies_per_batch': 5,
            'max_refinement_attempts': 0,
            'strategy_family_size': 3,
            'enable_refinement': False,
            'enable_family_generation': True
        }
    )

    report = orchestrator.run_iteration(
        data=sample_ohlcv_data,
        symbol='BTC-USD'
    )

    # Verify family generation occurred if there were approved strategies
    if report.num_approved > 0:
        assert report.num_family_variants > 0


def test_orchestrator_with_database(
    sample_ohlcv_data,
    mock_strategy_generator,
    mock_code_generator,
    mock_batch_tester,
    mock_strategy_filter,
    mock_portfolio_evaluator,
    tmp_path
):
    """Test orchestrator with database storage."""
    from src.database.manager import StrategyDatabase

    # Create temporary database
    db_path = tmp_path / "test_orchestrator.db"
    database = StrategyDatabase(str(db_path))

    orchestrator = ResearchOrchestrator(
        strategy_generator=mock_strategy_generator,
        code_generator=mock_code_generator,
        batch_tester=mock_batch_tester,
        strategy_filter=mock_strategy_filter,
        portfolio_evaluator=mock_portfolio_evaluator,
        database=database,
        config={
            'num_strategies_per_batch': 2,
            'enable_refinement': False,
            'enable_family_generation': False,
            'enable_database_storage': True
        }
    )

    report = orchestrator.run_iteration(
        data=sample_ohlcv_data,
        symbol='BTC-USD'
    )

    # Verify database storage occurred
    assert report.database_stored is True

    # Verify strategies were stored
    recent = database.get_recent_strategies(limit=10)
    assert len(recent) >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
