"""Main portfolio evaluator facade."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from src.portfolio.models import (
    StrategyPortfolioProfile,
    PortfolioEvaluationResult
)
from src.portfolio.correlation import CorrelationAnalyzer
from src.portfolio.risk_manager import PortfolioRiskManager
from src.portfolio.allocation import AllocationOptimizer
from src.portfolio.forward_selector import ForwardTestSelector

logger = logging.getLogger(__name__)


class PortfolioEvaluator:
    """Main interface for portfolio-level strategy evaluation."""

    def __init__(
        self,
        correlation_analyzer: Optional[CorrelationAnalyzer] = None,
        risk_manager: Optional[PortfolioRiskManager] = None,
        allocation_optimizer: Optional[AllocationOptimizer] = None,
        forward_selector: Optional[ForwardTestSelector] = None
    ):
        """
        Initialize portfolio evaluator with components.

        Args:
            correlation_analyzer: Correlation analyzer (uses default if None)
            risk_manager: Risk manager (uses default if None)
            allocation_optimizer: Allocation optimizer (uses default if None)
            forward_selector: Forward test selector (uses default if None)
        """
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()
        self.risk_manager = risk_manager or PortfolioRiskManager()
        self.allocation_optimizer = allocation_optimizer or AllocationOptimizer()
        self.forward_selector = forward_selector or ForwardTestSelector()

        logger.info("Initialized PortfolioEvaluator with all components")

    def evaluate_portfolio(
        self,
        backtest_results: List[Dict[str, Any]],
        historical_data: Optional[pd.DataFrame] = None,
        deployed_strategies: Optional[List[str]] = None
    ) -> PortfolioEvaluationResult:
        """
        Main evaluation method - analyzes portfolio holistically.

        Args:
            backtest_results: List of results from BacktestRunner
            historical_data: OHLCV data for regime analysis (Phase 2)
            deployed_strategies: Currently deployed strategy names

        Returns:
            Comprehensive evaluation report
        """
        logger.info(f"Evaluating portfolio with {len(backtest_results)} strategies")

        # 1. Build strategy profiles
        strategies = self._build_strategy_profiles(backtest_results)

        if len(strategies) == 0:
            logger.error("No valid strategies to evaluate")
            return self._create_empty_result()

        logger.info(f"Built {len(strategies)} strategy profiles")

        # 2. Correlation analysis
        correlation_result = self.correlation_analyzer.analyze(strategies)

        # 3. Optimize allocation
        optimal_weights = self.allocation_optimizer.optimize_weights(strategies)

        # 4. Calculate portfolio metrics
        portfolio_metrics = self.risk_manager.calculate_portfolio_metrics(
            strategies, optimal_weights
        )

        # 5. Generate allocation recommendations
        allocations = self.allocation_optimizer.generate_allocation_recommendations(
            strategies, optimal_weights, portfolio_metrics
        )

        # 6. Prioritize for forward testing
        priorities = self.forward_selector.score_strategies(
            strategies,
            portfolio_metrics,
            correlation_result.returns_correlation,
            regime_coverage=None,  # Phase 2: add regime analysis
            deployed_strategies=deployed_strategies
        )

        # 7. Apply acceptance criteria
        categorized = self.forward_selector.apply_acceptance_criteria(priorities)

        # 8. Compile result
        result = PortfolioEvaluationResult(
            portfolio_metrics=portfolio_metrics,
            correlation_matrix=correlation_result,
            optimal_allocations=allocations,
            forward_test_priorities=priorities,
            trade_overlap=correlation_result.trade_overlap_matrix,
            redundant_groups=correlation_result.redundant_groups,
            ready_for_deployment=[p.strategy_name for p in categorized['ready']],
            needs_monitoring=[p.strategy_name for p in categorized['monitor']],
            rejected_strategies=[p.strategy_name for p in categorized['reject']],
            evaluation_timestamp=datetime.now(),
            num_strategies_evaluated=len(strategies)
        )

        logger.info("Portfolio evaluation complete")
        logger.info(f"\n{result.summary()}")

        return result

    def quick_evaluation(
        self,
        backtest_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simplified evaluation without regime analysis (faster).

        For rapid iteration, skips regime analysis and uses simpler methods.

        Args:
            backtest_results: List of results from BacktestRunner

        Returns:
            Dictionary with simplified evaluation results
        """
        logger.info(f"Quick evaluation of {len(backtest_results)} strategies")

        # Build profiles
        strategies = self._build_strategy_profiles(backtest_results)

        if len(strategies) == 0:
            return {'error': 'No valid strategies'}

        # Correlation only
        correlation_result = self.correlation_analyzer.analyze(strategies)

        # Equal weights (no optimization)
        n = len(strategies)
        equal_weights = np.ones(n) / n

        # Basic portfolio metrics
        portfolio_metrics = self.risk_manager.calculate_portfolio_metrics(
            strategies, equal_weights
        )

        result = {
            'num_strategies': len(strategies),
            'portfolio_sharpe': portfolio_metrics.portfolio_sharpe,
            'portfolio_return': portfolio_metrics.portfolio_return,
            'portfolio_max_dd': portfolio_metrics.portfolio_max_dd,
            'avg_correlation': correlation_result.avg_correlation,
            'max_correlation': correlation_result.max_correlation,
            'diversification_ratio': portfolio_metrics.diversification_ratio
        }

        logger.info(f"Quick evaluation complete: Sharpe={result['portfolio_sharpe']:.2f}")

        return result

    def _build_strategy_profiles(
        self,
        backtest_results: List[Dict[str, Any]]
    ) -> List[StrategyPortfolioProfile]:
        """
        Build strategy profiles from backtest results.

        Args:
            backtest_results: Raw backtest results

        Returns:
            List of strategy profiles
        """
        profiles = []

        for result in backtest_results:
            try:
                profile = self._build_single_profile(result)
                if profile:
                    profiles.append(profile)
            except Exception as e:
                strategy_name = result.get('strategy_name', 'Unknown')
                logger.error(f"Failed to build profile for {strategy_name}: {e}")
                continue

        return profiles

    def _build_single_profile(
        self,
        backtest_result: Dict[str, Any]
    ) -> Optional[StrategyPortfolioProfile]:
        """
        Build a single strategy profile from backtest result.

        Args:
            backtest_result: Single backtest result dict

        Returns:
            Strategy profile or None if invalid
        """
        strategy_name = backtest_result.get('strategy_name', 'Unknown')

        # Extract equity curve
        equity_curve = backtest_result.get('equity_curve')
        if equity_curve is None or len(equity_curve) < 2:
            logger.warning(f"No valid equity curve for {strategy_name}")
            return None

        # Calculate returns series
        if isinstance(equity_curve, pd.Series):
            returns_series = equity_curve.pct_change().dropna()
        else:
            # Convert to Series if it's a list/array
            equity_curve = pd.Series(equity_curve)
            returns_series = equity_curve.pct_change().dropna()

        # Extract trade information
        trades = backtest_result.get('trades', [])
        trade_timestamps = []
        active_periods = []

        if trades:
            for trade in trades:
                entry_time = trade.get('entry_time')
                exit_time = trade.get('exit_time')

                if entry_time:
                    trade_timestamps.append(entry_time)

                if entry_time and exit_time:
                    active_periods.append((entry_time, exit_time))

        # Get metadata if available
        metadata = backtest_result.get('metadata', {})

        # Calculate average holding period
        avg_holding_period = None
        if active_periods:
            holding_periods = [(end - start).total_seconds() for start, end in active_periods]
            avg_holding_period = pd.Timedelta(seconds=np.mean(holding_periods))

        profile = StrategyPortfolioProfile(
            strategy_name=strategy_name,
            backtest_results=backtest_result,
            returns_series=returns_series,
            trade_timestamps=trade_timestamps,
            active_periods=active_periods,
            strategy_type=metadata.get('strategy_type'),
            market_regime=metadata.get('market_regime'),
            indicators_used=metadata.get('indicators', []),
            avg_holding_period=avg_holding_period
        )

        return profile

    def _create_empty_result(self) -> PortfolioEvaluationResult:
        """Create empty result for error cases."""
        from src.portfolio.models import PortfolioMetrics, CorrelationMatrix

        empty_metrics = PortfolioMetrics(
            portfolio_sharpe=0.0,
            portfolio_return=0.0,
            portfolio_max_dd=0.0,
            portfolio_volatility=0.0,
            diversification_ratio=1.0,
            concentration_risk=1.0,
            correlation_adjusted_sharpe=0.0,
            num_strategies=0,
            total_trades=0
        )

        empty_correlation = CorrelationMatrix(
            returns_correlation=pd.DataFrame(),
            trade_overlap_matrix=pd.DataFrame(),
            redundant_groups=[]
        )

        return PortfolioEvaluationResult(
            portfolio_metrics=empty_metrics,
            correlation_matrix=empty_correlation,
            optimal_allocations=[],
            forward_test_priorities=[],
            num_strategies_evaluated=0
        )

    def compare_portfolios(
        self,
        portfolio_a: List[Dict[str, Any]],
        portfolio_b: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare two portfolios side-by-side.

        Args:
            portfolio_a: First portfolio backtest results
            portfolio_b: Second portfolio backtest results

        Returns:
            Comparison results
        """
        logger.info("Comparing two portfolios")

        eval_a = self.evaluate_portfolio(portfolio_a)
        eval_b = self.evaluate_portfolio(portfolio_b)

        comparison = {
            'portfolio_a': {
                'sharpe': eval_a.portfolio_metrics.portfolio_sharpe,
                'return': eval_a.portfolio_metrics.portfolio_return,
                'max_dd': eval_a.portfolio_metrics.portfolio_max_dd,
                'num_strategies': eval_a.num_strategies_evaluated,
                'ready_for_deployment': len(eval_a.ready_for_deployment)
            },
            'portfolio_b': {
                'sharpe': eval_b.portfolio_metrics.portfolio_sharpe,
                'return': eval_b.portfolio_metrics.portfolio_return,
                'max_dd': eval_b.portfolio_metrics.portfolio_max_dd,
                'num_strategies': eval_b.num_strategies_evaluated,
                'ready_for_deployment': len(eval_b.ready_for_deployment)
            },
            'winner': self._determine_winner(eval_a, eval_b)
        }

        return comparison

    def _determine_winner(
        self,
        eval_a: PortfolioEvaluationResult,
        eval_b: PortfolioEvaluationResult
    ) -> str:
        """Determine which portfolio is better."""
        score_a = (
            eval_a.portfolio_metrics.portfolio_sharpe * 0.4 +
            eval_a.portfolio_metrics.portfolio_return * 100 * 0.3 -
            eval_a.portfolio_metrics.portfolio_max_dd * 100 * 0.3
        )

        score_b = (
            eval_b.portfolio_metrics.portfolio_sharpe * 0.4 +
            eval_b.portfolio_metrics.portfolio_return * 100 * 0.3 -
            eval_b.portfolio_metrics.portfolio_max_dd * 100 * 0.3
        )

        if score_a > score_b:
            return 'portfolio_a'
        elif score_b > score_a:
            return 'portfolio_b'
        else:
            return 'tie'
