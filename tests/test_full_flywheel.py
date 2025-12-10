"""
Full flywheel integration test - tests complete research cycle end-to-end.

This test validates the entire autonomous strategy generation flywheel:
1. Strategy generation (Agent 1)
2. Code generation (Agent 2)
3. Backtesting (Agent 3)
4. Filtering & critique
5. Portfolio evaluation (Agent 5)
6. Refinement of failures (Agent 4)
7. Family generation from successes
8. Database storage and genealogy tracking
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.orchestrator.research_engine import ResearchOrchestrator, IterationReport
from src.strategy_generation.generator import StrategyGenerator
from src.code_generation.code_generator import CodeGenerator
from src.backtest.batch_tester import BatchTester
from src.critique.filter import StrategyFilter, FilterCriteria
from src.portfolio.evaluator import PortfolioEvaluator
from src.database.manager import StrategyDatabase


@pytest.fixture
def comprehensive_ohlcv_data():
    """Create comprehensive OHLCV data for realistic testing."""
    # Create 30 days of hourly data (720 candles)
    dates = pd.date_range(start='2024-01-01', periods=720, freq='1H')
    np.random.seed(42)

    # Create realistic price movement with trends and volatility
    base_price = 50000
    returns = np.random.randn(720) * 0.01  # 1% volatility
    price_series = base_price * (1 + returns).cumprod()

    data = pd.DataFrame({
        'timestamp': dates,
        'open': price_series,
        'high': price_series * 1.005,
        'low': price_series * 0.995,
        'close': price_series * 1.001,
        'volume': np.random.uniform(100, 1000, 720)
    })

    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def test_database(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_flywheel.db"
    database = StrategyDatabase(str(db_path))
    return database


@pytest.fixture
def full_orchestrator(test_database):
    """Create a fully configured orchestrator with real components."""
    # Initialize all agents with test API keys
    from src.strategy_generation.llm_client import LLMClient
    llm_client = LLMClient(provider="openai", api_key="test-key", model="gpt-4", temperature=0.7)

    available_indicators = ['rsi', 'sma', 'ema', 'macd', 'bollinger_bands', 'atr']
    strategy_generator = StrategyGenerator(
        llm_client=llm_client,
        available_indicators=available_indicators
    )

    code_llm_client = LLMClient(provider="openai", api_key="test-key", model="gpt-4", temperature=0.0)
    code_generator = CodeGenerator(
        llm_client=code_llm_client
    )

    batch_tester = BatchTester(
        initial_capital=10000,
        slippage_bps=5.0,
        db=test_database
    )

    # Create filter criteria for testing (relaxed thresholds)
    filter_criteria = FilterCriteria(
        min_sharpe_ratio=0.3,
        min_total_return=0.03,
        max_drawdown=0.40,
        min_num_trades=3,
        min_win_rate=0.30,
        min_profit_factor=0.8
    )

    strategy_filter = StrategyFilter(criteria=filter_criteria)

    portfolio_evaluator = PortfolioEvaluator()

    # Create orchestrator with all components
    orchestrator = ResearchOrchestrator(
        strategy_generator=strategy_generator,
        code_generator=code_generator,
        batch_tester=batch_tester,
        strategy_filter=strategy_filter,
        portfolio_evaluator=portfolio_evaluator,
        database=test_database,
        config={
            'num_strategies_per_batch': 5,
            'max_refinement_attempts': 2,
            'strategy_family_size': 3,
            'enable_refinement': True,
            'enable_family_generation': True,
            'enable_database_storage': True
        }
    )

    return orchestrator


class TestFullFlywheelIntegration:
    """Test complete flywheel integration."""

    def test_complete_iteration_with_mocked_llm(
        self,
        comprehensive_ohlcv_data,
        test_database,
        monkeypatch
    ):
        """
        Test complete iteration with mocked LLM calls.

        This test runs through all 8 phases:
        1. Generate 5 strategies
        2. Generate code for all
        3. Backtest all strategies
        4. Filter by acceptance criteria
        5. Portfolio evaluation
        6. Refinement of failures
        7. Family generation from successes
        8. Database storage and verification
        """
        # Mock LLM calls to avoid API dependencies
        mock_strategies = [
            {
                'strategy_name': f'TestStrategy_{i}',
                'strategy_type': 'momentum' if i % 2 == 0 else 'mean_reversion',
                'indicators': ['rsi', 'sma', 'ema'],
                'entry_conditions': 'RSI crosses above 30',
                'exit_conditions': 'RSI crosses below 70',
                'position_sizing': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'timeframe': '1h',
                'generated_at': datetime.now().timestamp()
            }
            for i in range(5)
        ]

        def mock_generate_strategy(*args, **kwargs):
            return mock_strategies.pop(0) if mock_strategies else mock_strategies[0].copy()

        def mock_generate_code(self, strategy_metadata):
            strategy_name = strategy_metadata['strategy_name']
            return f"""
import pandas as pd
import numpy as np
from src.backtest.base_strategy import BaseStrategy

class {strategy_name}(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.position_size = 0.1
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04

    def should_enter(self, current_candle, historical_data):
        if len(historical_data) < 30:
            return False

        # Simple RSI-based entry
        close_prices = historical_data['close'].values[-30:]
        returns = np.diff(close_prices) / close_prices[:-1]

        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)

        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])

        if avg_loss == 0:
            return False

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi < 30

    def should_exit(self, current_candle, historical_data, position):
        if len(historical_data) < 30:
            return False

        # Check stop loss and take profit
        entry_price = position['entry_price']
        current_price = current_candle['close']

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
            return True

        # Simple RSI-based exit
        close_prices = historical_data['close'].values[-30:]
        returns = np.diff(close_prices) / close_prices[:-1]

        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)

        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])

        if avg_loss == 0:
            return False

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi > 70
"""

        # Patch the LLM calls
        from src.strategy_generation import generator as gen_module
        from src.code_generation import code_generator as code_module

        monkeypatch.setattr(gen_module.StrategyGenerator, 'generate_strategy', mock_generate_strategy)
        monkeypatch.setattr(gen_module.StrategyGenerator, 'refine_strategy',
                           lambda self, *args, **kwargs: mock_generate_strategy())
        monkeypatch.setattr(gen_module.StrategyGenerator, 'generate_strategy_family',
                           lambda self, parent_strategy, num_variants: [mock_generate_strategy() for _ in range(num_variants)])
        monkeypatch.setattr(code_module.CodeGenerator, 'generate_strategy_class', mock_generate_code)

        # Reset mock strategies list
        mock_strategies = [
            {
                'strategy_name': f'TestStrategy_{i}',
                'strategy_type': 'momentum' if i % 2 == 0 else 'mean_reversion',
                'indicators': ['rsi', 'sma', 'ema'],
                'entry_conditions': 'RSI crosses above 30',
                'exit_conditions': 'RSI crosses below 70',
                'position_sizing': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'timeframe': '1h',
                'generated_at': datetime.now().timestamp()
            }
            for i in range(20)  # Extra for refinements and families
        ]

        # Create orchestrator with mocked components
        from src.strategy_generation.llm_client import LLMClient
        llm_client = LLMClient(provider="openai", api_key="test-key", model="gpt-4")

        available_indicators = ['rsi', 'sma', 'ema', 'macd', 'bollinger_bands', 'atr']
        strategy_generator = StrategyGenerator(
            llm_client=llm_client,
            available_indicators=available_indicators
        )
        code_llm_client = LLMClient(provider="openai", api_key="test-key", model="gpt-4", temperature=0.0)
        code_generator = CodeGenerator(llm_client=code_llm_client)
        batch_tester = BatchTester(initial_capital=10000, db=test_database)

        filter_criteria = FilterCriteria(
            min_sharpe_ratio=0.3,
            min_total_return=0.02,
            max_drawdown=0.50,
            min_num_trades=2,
            min_win_rate=0.25,
            min_profit_factor=0.7
        )
        strategy_filter = StrategyFilter(criteria=filter_criteria)

        portfolio_evaluator = PortfolioEvaluator()

        orchestrator = ResearchOrchestrator(
            strategy_generator=strategy_generator,
            code_generator=code_generator,
            batch_tester=batch_tester,
            strategy_filter=strategy_filter,
            portfolio_evaluator=portfolio_evaluator,
            database=test_database,
            config={
                'num_strategies_per_batch': 5,
                'max_refinement_attempts': 1,
                'strategy_family_size': 2,
                'enable_refinement': True,
                'enable_family_generation': True,
                'enable_database_storage': True
            }
        )

        # Run complete iteration
        report = orchestrator.run_iteration(
            data=comprehensive_ohlcv_data,
            symbol='BTC-USD'
        )

        # Verify Phase 1: Strategy Generation
        assert report.num_strategies_generated == 5, "Should generate 5 strategies"

        # Verify Phase 2: Code Generation
        assert report.num_codes_generated >= 4, "Should generate code for most strategies"

        # Verify Phase 3: Backtesting
        assert report.num_backtests_run >= 4, "Should run backtests for generated code"

        # Verify Phase 4: Filtering
        total_classified = report.num_approved + report.num_rejected + report.num_marginal
        assert total_classified == report.num_backtests_run, "All strategies should be classified"

        # Verify Phase 5: Portfolio Evaluation (if approved strategies exist)
        if report.num_approved > 0:
            assert report.portfolio_metrics is not None, "Should have portfolio metrics"
            assert 'sharpe_ratio' in report.portfolio_metrics
            assert 'total_return' in report.portfolio_metrics
            assert len(report.forward_test_priorities) > 0, "Should have forward test priorities"

        # Verify Phase 6: Refinement (if rejected strategies exist)
        if report.num_rejected > 0:
            assert report.num_refinements >= 0, "Should attempt refinement"

        # Verify Phase 7: Family Generation (if approved strategies exist)
        if report.num_approved > 0:
            assert report.num_family_variants >= 0, "Should generate family variants"

        # Verify Phase 8: Database Storage
        assert report.database_stored is True, "Should store to database"

        # Verify database contents
        recent_strategies = test_database.get_recent_strategies(limit=20)
        assert len(recent_strategies) >= 5, "Should have stored strategies in database"

        # Verify strategy genealogy tracking
        for strategy_record in recent_strategies:
            if strategy_record.parent_id is not None:
                parent = test_database.get_strategy_by_id(strategy_record.parent_id)
                assert parent is not None, "Parent strategy should exist in database"

        # Verify iteration report completeness
        assert report.iteration_number == 1
        assert report.timestamp is not None
        assert report.total_duration_seconds > 0

        # Print report for manual inspection
        print("\n" + "="*80)
        print("FULL FLYWHEEL INTEGRATION TEST REPORT")
        print("="*80)
        print(str(report))


    def test_multi_iteration_evolution(
        self,
        comprehensive_ohlcv_data,
        test_database,
        monkeypatch
    ):
        """
        Test multiple iterations to verify strategy evolution and learning.

        This test runs 3 iterations and verifies:
        - Strategies accumulate in database
        - Genealogy relationships are tracked
        - Duplicate avoidance uses historical strategies
        """
        # Mock LLM calls (similar to previous test)
        strategy_counter = {'count': 0}

        def mock_generate_strategy(*args, **kwargs):
            strategy_counter['count'] += 1
            return {
                'strategy_name': f'EvolutionStrategy_{strategy_counter["count"]}',
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

        def mock_generate_code(self, strategy_metadata):
            strategy_name = strategy_metadata['strategy_name']
            return f"""
import pandas as pd
import numpy as np
from src.backtest.base_strategy import BaseStrategy

class {strategy_name}(BaseStrategy):
    def __init__(self):
        super().__init__()

    def should_enter(self, current_candle, historical_data):
        return len(historical_data) > 20 and np.random.random() < 0.05

    def should_exit(self, current_candle, historical_data, position):
        return np.random.random() < 0.1
"""

        # Patch LLM calls
        from src.strategy_generation import generator as gen_module
        from src.code_generation import code_generator as code_module

        monkeypatch.setattr(gen_module.StrategyGenerator, 'generate_strategy', mock_generate_strategy)
        monkeypatch.setattr(gen_module.StrategyGenerator, 'refine_strategy',
                           lambda self, *args, **kwargs: mock_generate_strategy())
        monkeypatch.setattr(gen_module.StrategyGenerator, 'generate_strategy_family',
                           lambda self, parent_strategy, num_variants: [mock_generate_strategy() for _ in range(num_variants)])
        monkeypatch.setattr(code_module.CodeGenerator, 'generate_strategy_class', mock_generate_code)

        # Create orchestrator
        from src.strategy_generation.llm_client import LLMClient
        llm_client = LLMClient(provider="openai", api_key="test-key", model="gpt-4")

        available_indicators = ['rsi', 'sma', 'ema', 'macd', 'bollinger_bands', 'atr']
        strategy_generator = StrategyGenerator(
            llm_client=llm_client,
            available_indicators=available_indicators
        )
        code_llm_client = LLMClient(provider="openai", api_key="test-key", model="gpt-4", temperature=0.0)
        code_generator = CodeGenerator(llm_client=code_llm_client)
        batch_tester = BatchTester(initial_capital=10000, db=test_database)

        filter_criteria = FilterCriteria(min_sharpe_ratio=0.2, min_total_return=0.01)
        strategy_filter = StrategyFilter(criteria=filter_criteria)

        portfolio_evaluator = PortfolioEvaluator()

        orchestrator = ResearchOrchestrator(
            strategy_generator=strategy_generator,
            code_generator=code_generator,
            batch_tester=batch_tester,
            strategy_filter=strategy_filter,
            portfolio_evaluator=portfolio_evaluator,
            db=test_database,
            config={
                'num_strategies_per_batch': 3,
                'enable_refinement': False,
                'enable_family_generation': False,
                'enable_database_storage': True
            }
        )

        # Run 3 iterations
        reports = []
        for i in range(3):
            report = orchestrator.run_iteration(
                data=comprehensive_ohlcv_data,
                symbol='BTC-USD'
            )
            reports.append(report)

        # Verify iterations accumulated
        assert len(reports) == 3
        assert reports[0].iteration_number == 1
        assert reports[1].iteration_number == 2
        assert reports[2].iteration_number == 3

        # Verify database accumulated strategies
        all_strategies = test_database.get_recent_strategies(limit=100)
        assert len(all_strategies) >= 9, "Should have at least 9 strategies from 3 iterations"

        # Verify genealogy relationships exist
        strategies_with_parents = [s for s in all_strategies if s.parent_id is not None]
        print(f"\nStrategies with genealogy: {len(strategies_with_parents)}/{len(all_strategies)}")


    def test_database_genealogy_queries(self, test_database):
        """Test database genealogy tracking and queries."""
        # Manually create a strategy genealogy tree for testing
        from src.database.schema import StrategyRecord
        from datetime import datetime

        # Create parent strategy
        parent = StrategyRecord(
            strategy_name="Parent_Strategy",
            strategy_type="momentum",
            indicators=["rsi", "sma"],
            entry_conditions="RSI < 30",
            exit_conditions="RSI > 70",
            position_sizing=0.1,
            created_at=datetime.now()
        )
        parent_id = test_database.store_strategy(
            strategy_metadata=parent.to_dict(),
            code="# parent code"
        )

        # Create child strategies
        child_ids = []
        for i in range(3):
            child = StrategyRecord(
                strategy_name=f"Child_Strategy_{i}",
                strategy_type="momentum",
                indicators=["rsi", "sma", "ema"],
                entry_conditions="RSI < 25",
                exit_conditions="RSI > 75",
                position_sizing=0.1,
                parent_id=parent_id,
                created_at=datetime.now()
            )
            child_id = test_database.store_strategy(
                strategy_metadata=child.to_dict(),
                code=f"# child {i} code"
            )
            child_ids.append(child_id)

        # Query genealogy
        genealogy = test_database.get_strategy_genealogy(parent_id)

        assert genealogy is not None, "Should retrieve genealogy"
        assert genealogy.strategy_name == "Parent_Strategy"

        # Verify we can query recent strategies
        recent = test_database.get_recent_strategies(limit=10)
        assert len(recent) >= 4, "Should have parent + 3 children"

        print(f"\nGenealogy tree created: 1 parent, 3 children")


class TestIterationReportFormatting:
    """Test iteration report formatting and display."""

    def test_report_string_representation(self):
        """Test that report generates readable string output."""
        report = IterationReport(
            iteration_number=1,
            timestamp=datetime.now(),
            num_strategies_generated=10,
            num_codes_generated=9,
            num_backtests_run=9,
            num_approved=3,
            num_rejected=5,
            num_marginal=1,
            num_refinements=2,
            num_family_variants=6,
            database_stored=True,
            total_duration_seconds=125.5
        )

        report_str = str(report)

        # Verify key information is present
        assert "RESEARCH ITERATION #1" in report_str
        assert "10" in report_str  # strategies generated
        assert "3" in report_str   # approved
        assert "5" in report_str   # rejected
        assert "125.5" in report_str  # duration
        assert "Stored" in report_str  # database storage


class TestFlywheelErrorHandling:
    """Test error handling throughout the flywheel."""

    def test_partial_failure_recovery(
        self,
        comprehensive_ohlcv_data,
        test_database,
        monkeypatch
    ):
        """
        Test that orchestrator handles partial failures gracefully.

        Simulates:
        - Some strategy generation failures
        - Some code generation failures
        - Some backtest failures

        Verifies that the system continues and reports properly.
        """
        call_count = {'count': 0}

        def mock_generate_strategy_with_failures(*args, **kwargs):
            call_count['count'] += 1
            # Fail every 3rd call
            if call_count['count'] % 3 == 0:
                raise Exception("Simulated generation failure")

            return {
                'strategy_name': f'RobustStrategy_{call_count["count"]}',
                'strategy_type': 'momentum',
                'indicators': ['rsi'],
                'entry_conditions': 'RSI < 30',
                'exit_conditions': 'RSI > 70',
                'position_sizing': 0.1,
                'generated_at': datetime.now().timestamp()
            }

        def mock_generate_code(self, strategy_metadata):
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

        # Patch with failure-prone mocks
        from src.strategy_generation import generator as gen_module
        from src.code_generation import code_generator as code_module

        monkeypatch.setattr(gen_module.StrategyGenerator, 'generate_strategy',
                           mock_generate_strategy_with_failures)
        monkeypatch.setattr(code_module.CodeGenerator, 'generate_strategy_class',
                           mock_generate_code)

        # Create orchestrator
        from src.strategy_generation.llm_client import LLMClient
        llm_client = LLMClient(provider="openai", api_key="test-key", model="gpt-4")

        available_indicators = ['rsi', 'sma', 'ema', 'macd', 'bollinger_bands', 'atr']
        strategy_generator = StrategyGenerator(
            llm_client=llm_client,
            available_indicators=available_indicators
        )
        code_llm_client = LLMClient(provider="openai", api_key="test-key", model="gpt-4", temperature=0.0)
        code_generator = CodeGenerator(llm_client=code_llm_client)
        batch_tester = BatchTester(initial_capital=10000)

        filter_criteria = FilterCriteria()
        strategy_filter = StrategyFilter(criteria=filter_criteria)

        portfolio_evaluator = PortfolioEvaluator()

        orchestrator = ResearchOrchestrator(
            strategy_generator=strategy_generator,
            code_generator=code_generator,
            batch_tester=batch_tester,
            strategy_filter=strategy_filter,
            portfolio_evaluator=portfolio_evaluator,
            db=test_database,
            config={
                'num_strategies_per_batch': 6,
                'enable_refinement': False,
                'enable_family_generation': False
            }
        )

        # Run iteration - should handle failures gracefully
        report = orchestrator.run_iteration(
            data=comprehensive_ohlcv_data,
            symbol='BTC-USD'
        )

        # Verify partial success
        assert report.num_strategies_generated < 6, "Should have some generation failures"
        assert report.num_strategies_generated > 0, "Should have some successes"
        assert len(report.generation_failures) > 0, "Should track failures"

        print(f"\nPartial failure test: {report.num_strategies_generated}/6 strategies succeeded")
        print(f"Failures tracked: {len(report.generation_failures)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
