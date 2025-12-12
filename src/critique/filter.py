"""Strategy filter for basic acceptance criteria."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterCriteria:
    """Acceptance criteria for strategy filtering."""

    # Return thresholds
    min_total_return: float = 0.05  # 5%
    min_sharpe_ratio: float = 0.5

    # Risk thresholds
    max_drawdown: float = 0.30  # 30%
    max_consecutive_losses: Optional[int] = 10

    # Trading activity thresholds
    min_num_trades: int = 10
    max_num_trades: Optional[int] = None  # No upper limit by default

    # Win rate and profit factor
    min_win_rate: float = 0.30  # 30%
    min_profit_factor: float = 1.0  # Must be profitable overall

    # Statistical significance (optional)
    min_sample_size: int = 20  # Minimum trades for statistical significance

    def __str__(self) -> str:
        """String representation of criteria."""
        return (
            f"FilterCriteria(\n"
            f"  min_return={self.min_total_return*100:.1f}%, "
            f"min_sharpe={self.min_sharpe_ratio:.2f}\n"
            f"  max_drawdown={self.max_drawdown*100:.1f}%, "
            f"min_win_rate={self.min_win_rate*100:.1f}%\n"
            f"  min_trades={self.min_num_trades}, "
            f"min_profit_factor={self.min_profit_factor:.2f}\n"
            f")"
        )


@dataclass
class FilterResult:
    """Result of strategy filtering."""

    passed: bool
    strategy_name: str
    rejection_reasons: List[str]
    metrics_summary: Dict[str, Any]
    classification: str  # 'approved', 'rejected', 'marginal'

    def __str__(self) -> str:
        """String representation of filter result."""
        status = "✓ APPROVED" if self.passed else "✗ REJECTED"
        reasons = "\n    ".join(self.rejection_reasons) if self.rejection_reasons else "All criteria met"

        return (
            f"{status}: {self.strategy_name}\n"
            f"  Classification: {self.classification}\n"
            f"  Reasons: {reasons}\n"
            f"  Return: {self.metrics_summary.get('total_return', 0)*100:.2f}%, "
            f"Sharpe: {self.metrics_summary.get('sharpe_ratio', 0):.2f}, "
            f"DD: {self.metrics_summary.get('max_drawdown', 0)*100:.2f}%"
        )


class StrategyFilter:
    """Filters strategies based on acceptance criteria."""

    def __init__(self, criteria: Optional[FilterCriteria] = None):
        """Initialize filter with criteria.

        Args:
            criteria: FilterCriteria instance (uses defaults if None)
        """
        self.criteria = criteria or FilterCriteria()
        logger.info(f"Initialized StrategyFilter with criteria:\n{self.criteria}")

    def filter_strategy(
        self,
        backtest_results: Dict[str, Any],
        strict: bool = False
    ) -> FilterResult:
        """Filter a single strategy based on backtest results.

        Args:
            backtest_results: Backtest results dict from BacktestRunner
            strict: If True, use stricter thresholds

        Returns:
            FilterResult with pass/fail status and reasons
        """
        strategy_name = backtest_results.get('strategy_name', 'Unknown')
        metrics = backtest_results.get('metrics', {})

        rejection_reasons = []

        # Check if backtest succeeded
        if backtest_results.get('status') != 'success':
            rejection_reasons.append(
                f"Backtest failed: {backtest_results.get('error', 'unknown error')}"
            )
            return FilterResult(
                passed=False,
                strategy_name=strategy_name,
                rejection_reasons=rejection_reasons,
                metrics_summary=metrics,
                classification='rejected'
            )

        # Extract key metrics
        total_return = metrics.get('total_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        num_trades = metrics.get('total_trades', 0)

        # Apply filters
        if total_return < self.criteria.min_total_return:
            rejection_reasons.append(
                f"Total return {total_return*100:.2f}% < {self.criteria.min_total_return*100:.2f}%"
            )

        if sharpe_ratio < self.criteria.min_sharpe_ratio:
            rejection_reasons.append(
                f"Sharpe ratio {sharpe_ratio:.2f} < {self.criteria.min_sharpe_ratio:.2f}"
            )

        if max_drawdown > self.criteria.max_drawdown:
            rejection_reasons.append(
                f"Max drawdown {max_drawdown*100:.2f}% > {self.criteria.max_drawdown*100:.2f}%"
            )

        if num_trades < self.criteria.min_num_trades:
            rejection_reasons.append(
                f"Number of trades {num_trades} < {self.criteria.min_num_trades}"
            )

        if self.criteria.max_num_trades and num_trades > self.criteria.max_num_trades:
            rejection_reasons.append(
                f"Number of trades {num_trades} > {self.criteria.max_num_trades} (overtrading)"
            )

        if win_rate < self.criteria.min_win_rate:
            rejection_reasons.append(
                f"Win rate {win_rate*100:.2f}% < {self.criteria.min_win_rate*100:.2f}%"
            )

        if profit_factor < self.criteria.min_profit_factor:
            rejection_reasons.append(
                f"Profit factor {profit_factor:.2f} < {self.criteria.min_profit_factor:.2f}"
            )

        # Statistical significance check
        if num_trades < self.criteria.min_sample_size:
            if strict:
                rejection_reasons.append(
                    f"Insufficient sample size {num_trades} < {self.criteria.min_sample_size}"
                )
            else:
                # Just warn, don't reject
                logger.warning(
                    f"{strategy_name}: Low sample size ({num_trades} trades), "
                    "results may not be statistically significant"
                )

        # Determine classification
        passed = len(rejection_reasons) == 0

        if passed:
            # Further classify approved strategies
            if sharpe_ratio >= 1.5 and total_return >= 0.20:
                classification = 'excellent'
            elif sharpe_ratio >= 1.0 and total_return >= 0.10:
                classification = 'approved'
            else:
                classification = 'marginal'
        else:
            classification = 'rejected'

        result = FilterResult(
            passed=passed,
            strategy_name=strategy_name,
            rejection_reasons=rejection_reasons,
            metrics_summary={
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trades': num_trades
            },
            classification=classification
        )

        if passed:
            logger.info(f"✓ {strategy_name} PASSED filter ({classification})")
        else:
            logger.info(
                f"✗ {strategy_name} REJECTED: "
                f"{len(rejection_reasons)} criteria failed"
            )

        return result

    def filter_batch(
        self,
        batch_results: List[Dict[str, Any]],
        strict: bool = False
    ) -> List[FilterResult]:
        """Filter multiple strategies.

        Args:
            batch_results: List of backtest results
            strict: If True, use stricter thresholds

        Returns:
            List of FilterResult objects
        """
        logger.info(f"Filtering {len(batch_results)} strategies")

        filter_results = []
        for result in batch_results:
            filter_result = self.filter_strategy(result, strict=strict)
            filter_results.append(filter_result)

        # Summary statistics
        approved = sum(1 for r in filter_results if r.passed)
        excellent = sum(1 for r in filter_results if r.classification == 'excellent')
        marginal = sum(1 for r in filter_results if r.classification == 'marginal')

        logger.info(
            f"Filter complete: {approved}/{len(batch_results)} passed "
            f"({excellent} excellent, {marginal} marginal)"
        )

        return filter_results

    def get_approved_strategies(
        self,
        filter_results: List[FilterResult],
        include_marginal: bool = False
    ) -> List[str]:
        """Get list of approved strategy names.

        Args:
            filter_results: List of FilterResult objects
            include_marginal: Whether to include marginal strategies

        Returns:
            List of strategy names that passed
        """
        approved = []
        for result in filter_results:
            if result.passed:
                if include_marginal or result.classification != 'marginal':
                    approved.append(result.strategy_name)

        return approved

    def analyze_rejection_patterns(
        self,
        filter_results: List[FilterResult]
    ) -> Dict[str, Any]:
        """Analyze common rejection patterns.

        Args:
            filter_results: List of FilterResult objects

        Returns:
            Dict with rejection pattern analysis
        """
        all_reasons = []
        for result in filter_results:
            if not result.passed:
                all_reasons.extend(result.rejection_reasons)

        # Count rejection reason frequency
        reason_counts = {}
        for reason in all_reasons:
            # Extract the metric type from the reason
            metric = reason.split()[0]
            reason_counts[metric] = reason_counts.get(metric, 0) + 1

        # Sort by frequency
        sorted_reasons = sorted(
            reason_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        total_rejected = sum(1 for r in filter_results if not r.passed)

        return {
            'total_filtered': len(filter_results),
            'total_rejected': total_rejected,
            'rejection_rate': total_rejected / len(filter_results) if filter_results else 0,
            'common_failure_modes': sorted_reasons,
            'most_common_failure': sorted_reasons[0][0] if sorted_reasons else None
        }
