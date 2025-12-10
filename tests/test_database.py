"""Tests for database functionality."""

import os
import tempfile
from datetime import datetime, timedelta
import pytest

from src.database import StrategyDatabase
from src.database.schema import init_database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    # Initialize database
    init_database(db_path)
    db = StrategyDatabase(db_path)

    yield db

    # Cleanup
    try:
        os.unlink(db_path)
    except Exception:
        pass


class TestStrategyStorage:
    """Test strategy storage and retrieval."""

    def test_store_strategy(self, temp_db):
        """Test storing a strategy."""
        strategy_id = temp_db.store_strategy(
            name="Test Strategy 1",
            strategy_type="trend",
            indicators=["SMA", "EMA", "RSI"],
            description="A test trend following strategy",
            parameters={"sma_period": 20, "ema_period": 50},
            status="generated"
        )

        assert strategy_id > 0

        # Retrieve and verify
        strategy = temp_db.get_strategy_by_id(strategy_id)
        assert strategy is not None
        assert strategy['name'] == "Test Strategy 1"
        assert strategy['strategy_type'] == "trend"
        assert "SMA" in strategy['indicators']
        assert strategy['parameters']['sma_period'] == 20

    def test_duplicate_strategy_name(self, temp_db):
        """Test that duplicate strategy names are rejected."""
        temp_db.store_strategy(
            name="Unique Strategy",
            strategy_type="trend",
            indicators=["SMA"]
        )

        with pytest.raises(ValueError, match="already exists"):
            temp_db.store_strategy(
                name="Unique Strategy",
                strategy_type="mean_reversion",
                indicators=["RSI"]
            )

    def test_update_strategy_status(self, temp_db):
        """Test updating strategy status."""
        strategy_id = temp_db.store_strategy(
            name="Status Test",
            strategy_type="trend",
            indicators=["SMA"],
            status="generated"
        )

        # Update status
        temp_db.update_strategy_status(strategy_id, "tested")

        # Verify
        strategy = temp_db.get_strategy_by_id(strategy_id)
        assert strategy['status'] == "tested"

    def test_get_recent_strategies(self, temp_db):
        """Test retrieving recent strategies."""
        # Create several strategies
        for i in range(5):
            temp_db.store_strategy(
                name=f"Recent Strategy {i}",
                strategy_type="trend",
                indicators=["SMA"]
            )

        # Get recent strategies
        recent = temp_db.get_recent_strategies(days=1, limit=3)
        assert len(recent) == 3
        assert all('name' in s for s in recent)


class TestStrategyCode:
    """Test strategy code storage."""

    def test_store_strategy_code(self, temp_db):
        """Test storing generated code."""
        # First create a strategy
        strategy_id = temp_db.store_strategy(
            name="Code Test Strategy",
            strategy_type="trend",
            indicators=["SMA"]
        )

        # Store code
        code_text = "class TestStrategy:\n    pass"
        code_id = temp_db.store_strategy_code(
            strategy_id=strategy_id,
            code_text=code_text,
            validation_status="valid",
            llm_model="gpt-4",
            generation_attempts=1
        )

        assert code_id > 0

        # Verify status was updated
        strategy = temp_db.get_strategy_by_id(strategy_id)
        assert strategy['status'] == "coded"

    def test_multiple_code_versions(self, temp_db):
        """Test storing multiple code versions."""
        strategy_id = temp_db.store_strategy(
            name="Versioned Strategy",
            strategy_type="trend",
            indicators=["SMA"]
        )

        # Store first version
        code_id_1 = temp_db.store_strategy_code(
            strategy_id=strategy_id,
            code_text="# Version 1",
            validation_status="valid"
        )

        # Store second version
        code_id_2 = temp_db.store_strategy_code(
            strategy_id=strategy_id,
            code_text="# Version 2",
            validation_status="valid"
        )

        assert code_id_2 > code_id_1


class TestBacktestResults:
    """Test backtest result storage."""

    def test_store_backtest_results(self, temp_db):
        """Test storing backtest results."""
        # Create strategy
        strategy_id = temp_db.store_strategy(
            name="Backtest Strategy",
            strategy_type="trend",
            indicators=["SMA"]
        )

        # Store backtest results
        metrics = {
            'total_return_pct': 25.5,
            'sharpe_ratio': 1.8,
            'max_drawdown_pct': -15.2,
            'win_rate': 0.65,
            'profit_factor': 2.1,
            'num_trades': 50,
            'avg_trade_return_pct': 0.5
        }

        result_id = temp_db.store_backtest_results(
            strategy_id=strategy_id,
            symbol="BTC-USD",
            timeframe="1h",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=10000.0,
            metrics=metrics,
            passed_filters=True,
            filter_category="elite"
        )

        assert result_id > 0

        # Verify strategy status was updated
        strategy = temp_db.get_strategy_by_id(strategy_id)
        assert strategy['status'] == "approved"

    def test_backtest_with_equity_curve(self, temp_db):
        """Test storing backtest with equity curve."""
        strategy_id = temp_db.store_strategy(
            name="Equity Curve Test",
            strategy_type="trend",
            indicators=["SMA"]
        )

        equity_curve = [
            {'timestamp': '2023-01-01', 'equity': 10000},
            {'timestamp': '2023-01-02', 'equity': 10100},
            {'timestamp': '2023-01-03', 'equity': 10200}
        ]

        result_id = temp_db.store_backtest_results(
            strategy_id=strategy_id,
            symbol="BTC-USD",
            timeframe="1d",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3),
            initial_capital=10000.0,
            metrics={'total_return_pct': 2.0},
            equity_curve=equity_curve
        )

        assert result_id > 0


class TestGenealogy:
    """Test strategy genealogy tracking."""

    def test_parent_child_relationship(self, temp_db):
        """Test parent-child strategy relationship."""
        # Create parent strategy
        parent_id = temp_db.store_strategy(
            name="Parent Strategy",
            strategy_type="trend",
            indicators=["SMA"],
            generation=0
        )

        # Create child strategy
        child_id = temp_db.store_strategy(
            name="Child Strategy",
            strategy_type="trend",
            indicators=["SMA", "EMA"],
            parent_id=parent_id,
            generation=1,
            refinement_type="refinement"
        )

        # Get genealogy
        genealogy = temp_db.get_strategy_genealogy(child_id)

        assert genealogy['current']['id'] == child_id
        assert len(genealogy['ancestors']) == 1
        assert genealogy['ancestors'][0]['id'] == parent_id

    def test_multi_generation_genealogy(self, temp_db):
        """Test multi-generation genealogy."""
        # Create generation 0
        gen0_id = temp_db.store_strategy(
            name="Gen 0",
            strategy_type="trend",
            indicators=["SMA"],
            generation=0
        )

        # Create generation 1
        gen1_id = temp_db.store_strategy(
            name="Gen 1",
            strategy_type="trend",
            indicators=["SMA", "EMA"],
            parent_id=gen0_id,
            generation=1,
            refinement_type="refinement"
        )

        # Create generation 2
        gen2_id = temp_db.store_strategy(
            name="Gen 2",
            strategy_type="trend",
            indicators=["SMA", "EMA", "RSI"],
            parent_id=gen1_id,
            generation=2,
            refinement_type="refinement"
        )

        # Get genealogy
        genealogy = temp_db.get_strategy_genealogy(gen2_id)

        assert genealogy['current']['id'] == gen2_id
        assert len(genealogy['ancestors']) == 2
        assert genealogy['ancestors'][0]['id'] == gen1_id
        assert genealogy['ancestors'][1]['id'] == gen0_id

    def test_strategy_family(self, temp_db):
        """Test strategy family (parent with multiple children)."""
        # Create parent
        parent_id = temp_db.store_strategy(
            name="Parent",
            strategy_type="trend",
            indicators=["SMA"],
            generation=0
        )

        # Create multiple children (family members)
        child_ids = []
        for i in range(3):
            child_id = temp_db.store_strategy(
                name=f"Family Member {i}",
                strategy_type="trend",
                indicators=["SMA"],
                parent_id=parent_id,
                generation=1,
                refinement_type="family_member"
            )
            child_ids.append(child_id)

        # Get genealogy from parent
        genealogy = temp_db.get_strategy_genealogy(parent_id)

        assert genealogy['current']['id'] == parent_id
        assert len(genealogy['descendants']) == 3


class TestPortfolioEvaluation:
    """Test portfolio evaluation storage."""

    def test_store_portfolio_evaluation(self, temp_db):
        """Test storing portfolio evaluation."""
        # Create strategies
        strategy_ids = []
        for i in range(3):
            sid = temp_db.store_strategy(
                name=f"Portfolio Strategy {i}",
                strategy_type="trend",
                indicators=["SMA"]
            )
            strategy_ids.append(sid)

        # Store portfolio evaluation
        portfolio_metrics = {
            'portfolio_sharpe': 2.0,
            'portfolio_return_pct': 30.0,
            'portfolio_max_dd_pct': -12.0,
            'diversification_ratio': 1.5,
            'avg_correlation': 0.3
        }

        eval_id = temp_db.store_portfolio_evaluation(
            num_strategies=3,
            portfolio_metrics=portfolio_metrics,
            allocation_method="max_sharpe"
        )

        assert eval_id > 0

    def test_store_strategy_allocations(self, temp_db):
        """Test storing strategy allocations."""
        # Create evaluation
        eval_id = temp_db.store_portfolio_evaluation(
            num_strategies=2,
            portfolio_metrics={'portfolio_sharpe': 1.5}
        )

        # Create strategies
        strategy_ids = []
        for i in range(2):
            sid = temp_db.store_strategy(
                name=f"Allocation Strategy {i}",
                strategy_type="trend",
                indicators=["SMA"]
            )
            strategy_ids.append(sid)

        # Store allocations
        alloc_id_1 = temp_db.store_strategy_allocation(
            evaluation_id=eval_id,
            strategy_id=strategy_ids[0],
            weight=0.6,
            recommended_capital=6000.0
        )

        alloc_id_2 = temp_db.store_strategy_allocation(
            evaluation_id=eval_id,
            strategy_id=strategy_ids[1],
            weight=0.4,
            recommended_capital=4000.0
        )

        assert alloc_id_1 > 0
        assert alloc_id_2 > 0


class TestForwardTestQueue:
    """Test forward test queue management."""

    def test_add_to_queue(self, temp_db):
        """Test adding strategy to forward test queue."""
        # Create strategy
        strategy_id = temp_db.store_strategy(
            name="Queue Test",
            strategy_type="trend",
            indicators=["SMA"]
        )

        # Add to queue
        queue_id = temp_db.add_to_forward_test_queue(
            strategy_id=strategy_id,
            priority_score=85.0,
            priority_tier="ready",
            performance_score=90.0,
            portfolio_fit_score=80.0
        )

        assert queue_id > 0

    def test_get_forward_test_queue(self, temp_db):
        """Test retrieving forward test queue."""
        # Create strategies and add to queue
        for i in range(5):
            strategy_id = temp_db.store_strategy(
                name=f"Queue Strategy {i}",
                strategy_type="trend",
                indicators=["SMA"]
            )

            temp_db.add_to_forward_test_queue(
                strategy_id=strategy_id,
                priority_score=100.0 - (i * 10),  # Decreasing priority
                priority_tier="ready"
            )

        # Get queue
        queue = temp_db.get_forward_test_queue(status='queued', limit=3)

        assert len(queue) == 3
        # Should be sorted by priority (highest first)
        assert queue[0]['priority_score'] == 100.0
        assert queue[1]['priority_score'] == 90.0
        assert queue[2]['priority_score'] == 80.0

    def test_update_queue_entry(self, temp_db):
        """Test updating existing queue entry."""
        strategy_id = temp_db.store_strategy(
            name="Update Queue Test",
            strategy_type="trend",
            indicators=["SMA"]
        )

        # Add to queue
        queue_id_1 = temp_db.add_to_forward_test_queue(
            strategy_id=strategy_id,
            priority_score=70.0,
            priority_tier="monitor"
        )

        # Update (should update existing entry, not create new one)
        queue_id_2 = temp_db.add_to_forward_test_queue(
            strategy_id=strategy_id,
            priority_score=90.0,
            priority_tier="ready"
        )

        # Should be same ID
        assert queue_id_1 == queue_id_2


class TestFailurePatterns:
    """Test failure pattern analysis."""

    def test_get_failure_patterns(self, temp_db):
        """Test analyzing failure patterns."""
        # Create strategies and failed backtests
        for i in range(5):
            strategy_id = temp_db.store_strategy(
                name=f"Failed Strategy {i}",
                strategy_type="trend" if i < 3 else "mean_reversion",
                indicators=["SMA"]
            )

            # Store failed backtest
            temp_db.store_backtest_results(
                strategy_id=strategy_id,
                symbol="BTC-USD",
                timeframe="1h",
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now(),
                initial_capital=10000.0,
                metrics={'sharpe_ratio': -0.5},
                passed_filters=False,
                filter_category="rejected",
                rejection_reasons=["Low Sharpe ratio", "Insufficient trades"]
            )

        # Get failure patterns
        patterns = temp_db.get_failure_patterns(days=7, min_occurrences=2)

        assert len(patterns) > 0
        # Should have patterns for common rejection reasons
        reason_texts = [p['failure_reason'] for p in patterns]
        assert "Low Sharpe ratio" in reason_texts


class TestQueryStrategies:
    """Test strategy query interface."""

    def test_query_by_type(self, temp_db):
        """Test querying strategies by type."""
        # Create strategies of different types
        temp_db.store_strategy(
            name="Trend 1",
            strategy_type="trend",
            indicators=["SMA"]
        )
        temp_db.store_strategy(
            name="Mean Rev 1",
            strategy_type="mean_reversion",
            indicators=["RSI"]
        )

        # Query by type
        trend_strategies = temp_db.query_strategies(strategy_type="trend")
        assert len(trend_strategies) == 1
        assert trend_strategies[0]['strategy_type'] == "trend"

    def test_query_by_status(self, temp_db):
        """Test querying strategies by status."""
        # Create strategies with different statuses
        sid1 = temp_db.store_strategy(
            name="Generated",
            strategy_type="trend",
            indicators=["SMA"],
            status="generated"
        )
        sid2 = temp_db.store_strategy(
            name="Approved",
            strategy_type="trend",
            indicators=["SMA"],
            status="generated"
        )

        # Update one to approved
        temp_db.update_strategy_status(sid2, "approved")

        # Query
        approved = temp_db.query_strategies(status="approved")
        assert len(approved) == 1
        assert approved[0]['status'] == "approved"

    def test_query_by_sharpe(self, temp_db):
        """Test querying strategies with minimum Sharpe ratio."""
        # Create strategies with different Sharpe ratios
        for i in range(3):
            strategy_id = temp_db.store_strategy(
                name=f"Sharpe Test {i}",
                strategy_type="trend",
                indicators=["SMA"]
            )

            # Store backtest with different Sharpe
            temp_db.store_backtest_results(
                strategy_id=strategy_id,
                symbol="BTC-USD",
                timeframe="1h",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=10000.0,
                metrics={'sharpe_ratio': float(i)},  # 0.0, 1.0, 2.0
                passed_filters=True
            )

        # Query with min Sharpe
        high_sharpe = temp_db.query_strategies(min_sharpe=1.5)
        assert len(high_sharpe) == 1
        assert high_sharpe[0]['latest_backtest']['sharpe_ratio'] >= 1.5


class TestStatistics:
    """Test database statistics."""

    def test_get_statistics(self, temp_db):
        """Test retrieving database statistics."""
        # Create some data
        for i in range(3):
            strategy_id = temp_db.store_strategy(
                name=f"Stats Test {i}",
                strategy_type="trend" if i < 2 else "mean_reversion",
                indicators=["SMA"]
            )

            temp_db.store_backtest_results(
                strategy_id=strategy_id,
                symbol="BTC-USD",
                timeframe="1h",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=10000.0,
                metrics={'sharpe_ratio': 1.5},
                passed_filters=True if i < 2 else False
            )

        # Get statistics
        stats = temp_db.get_statistics()

        assert stats['total_strategies'] == 3
        assert stats['total_backtests'] == 3
        assert stats['approved_strategies'] == 2
        assert stats['strategies_by_type']['trend'] == 2
        assert stats['strategies_by_type']['mean_reversion'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
