"""Data models for portfolio evaluation."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class StrategyPortfolioProfile:
    """Enhanced strategy profile with portfolio-relevant data."""

    strategy_name: str
    backtest_results: Dict[str, Any]  # From BacktestRunner

    # Derived metrics
    returns_series: pd.Series  # Daily returns from equity curve
    trade_timestamps: List[datetime]
    active_periods: List[Tuple[datetime, datetime]]  # When in position

    # Strategy characteristics
    strategy_type: Optional[str] = None  # mean_reversion, trend_following, etc.
    market_regime: Optional[str] = None  # trending, ranging, volatile, etc.
    indicators_used: List[str] = field(default_factory=list)
    avg_holding_period: Optional[timedelta] = None

    # Performance metrics (from backtest_results)
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0

    def __post_init__(self):
        """Extract key metrics from backtest_results after initialization."""
        if self.backtest_results:
            metrics = self.backtest_results.get('metrics', {})
            self.sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
            self.total_return = metrics.get('total_return', 0.0)
            self.max_drawdown = metrics.get('max_drawdown', 0.0)
            self.num_trades = metrics.get('num_trades', 0)
            self.win_rate = metrics.get('win_rate', 0.0)


@dataclass
class CorrelationMatrix:
    """Correlation analysis results."""

    returns_correlation: pd.DataFrame  # Strategy returns correlation
    trade_overlap_matrix: pd.DataFrame  # Temporal trade overlap
    redundant_groups: List[List[str]]  # Groups of highly correlated strategies

    # Summary statistics
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    min_correlation: float = 0.0

    def __post_init__(self):
        """Calculate summary statistics after initialization."""
        if not self.returns_correlation.empty:
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(self.returns_correlation, dtype=bool), k=1)
            correlations = self.returns_correlation.values[mask]

            self.avg_correlation = float(np.mean(correlations))
            self.max_correlation = float(np.max(correlations))
            self.min_correlation = float(np.min(correlations))


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics."""

    portfolio_sharpe: float
    portfolio_return: float
    portfolio_max_dd: float
    portfolio_volatility: float

    # Diversification metrics
    diversification_ratio: float  # Portfolio vol / weighted avg vol
    concentration_risk: float  # Herfindahl index (sum of squared weights)
    correlation_adjusted_sharpe: float

    # Component metrics
    num_strategies: int
    total_trades: int

    # Equity curve
    portfolio_equity_curve: Optional[pd.Series] = None

    def __str__(self) -> str:
        """String representation of portfolio metrics."""
        return (
            f"Portfolio Metrics:\n"
            f"  Sharpe: {self.portfolio_sharpe:.2f}\n"
            f"  Return: {self.portfolio_return*100:.2f}%\n"
            f"  Max DD: {self.portfolio_max_dd*100:.2f}%\n"
            f"  Volatility: {self.portfolio_volatility*100:.2f}%\n"
            f"  Diversification Ratio: {self.diversification_ratio:.2f}\n"
            f"  Concentration Risk: {self.concentration_risk:.2f}\n"
            f"  Strategies: {self.num_strategies}\n"
            f"  Total Trades: {self.total_trades}"
        )


@dataclass
class AllocationRecommendation:
    """Position sizing recommendation for each strategy."""

    strategy_name: str
    allocation_weight: float  # 0.0 to 1.0
    allocation_rationale: str
    risk_contribution: float  # Marginal contribution to portfolio risk
    expected_sharpe_contribution: float

    # Additional context
    strategy_sharpe: float = 0.0
    strategy_return: float = 0.0
    correlation_with_portfolio: float = 0.0

    def __str__(self) -> str:
        """String representation of allocation recommendation."""
        return (
            f"{self.strategy_name}: {self.allocation_weight*100:.1f}%\n"
            f"  Rationale: {self.allocation_rationale}\n"
            f"  Risk Contribution: {self.risk_contribution*100:.2f}%\n"
            f"  Expected Sharpe Contribution: {self.expected_sharpe_contribution:.2f}"
        )


@dataclass
class ForwardTestPriority:
    """Priority ranking for forward testing."""

    strategy_name: str
    priority_score: float  # 0-100
    priority_tier: str  # "high", "medium", "low"
    reasons: List[str]
    deployment_readiness: str  # "ready", "monitor", "reject"

    # Scoring breakdown
    performance_score: float = 0.0
    portfolio_fit_score: float = 0.0
    novelty_score: float = 0.0
    risk_score: float = 0.0

    # Strategy metrics
    strategy_sharpe: float = 0.0
    correlation_with_deployed: float = 0.0
    fills_regime_gap: bool = False

    def __str__(self) -> str:
        """String representation of priority ranking."""
        status_emoji = {
            "ready": "✅",
            "monitor": "⚠️",
            "reject": "❌"
        }
        emoji = status_emoji.get(self.deployment_readiness, "")

        reasons_str = "\n    ".join(self.reasons)

        return (
            f"{emoji} {self.strategy_name} - Score: {self.priority_score:.1f} ({self.priority_tier})\n"
            f"  Status: {self.deployment_readiness.upper()}\n"
            f"  Reasons:\n    {reasons_str}\n"
            f"  Performance: {self.performance_score:.1f}, "
            f"Portfolio Fit: {self.portfolio_fit_score:.1f}, "
            f"Novelty: {self.novelty_score:.1f}"
        )


@dataclass
class PortfolioAcceptanceCriteria:
    """Portfolio-level acceptance criteria."""

    # Portfolio metrics
    min_portfolio_sharpe: float = 1.2  # Higher than individual
    max_portfolio_dd: float = 0.20  # Lower than individual
    min_diversification_ratio: float = 0.7  # Must see diversification benefit

    # Strategy composition
    max_correlation: float = 0.70  # No two strategies >70% correlated
    min_strategies: int = 3  # Minimum portfolio size
    max_strategies: int = 10  # Maximum to manage
    max_single_allocation: float = 0.40  # No strategy >40% of portfolio

    # Risk management
    max_trade_overlap: float = 0.60  # Max 60% simultaneous positions

    def __str__(self) -> str:
        """String representation of criteria."""
        return (
            f"Portfolio Acceptance Criteria:\n"
            f"  Min Sharpe: {self.min_portfolio_sharpe:.2f}\n"
            f"  Max Drawdown: {self.max_portfolio_dd*100:.1f}%\n"
            f"  Max Correlation: {self.max_correlation*100:.0f}%\n"
            f"  Portfolio Size: {self.min_strategies}-{self.max_strategies} strategies\n"
            f"  Max Single Allocation: {self.max_single_allocation*100:.0f}%"
        )


@dataclass
class PortfolioEvaluationResult:
    """Complete portfolio evaluation results."""

    # Core results
    portfolio_metrics: PortfolioMetrics
    correlation_matrix: CorrelationMatrix
    optimal_allocations: List[AllocationRecommendation]
    forward_test_priorities: List[ForwardTestPriority]

    # Additional analysis
    trade_overlap: Optional[pd.DataFrame] = None
    redundant_groups: List[List[str]] = field(default_factory=list)

    # Deployment recommendations
    ready_for_deployment: List[str] = field(default_factory=list)
    needs_monitoring: List[str] = field(default_factory=list)
    rejected_strategies: List[str] = field(default_factory=list)

    # Metadata
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    num_strategies_evaluated: int = 0

    def get_deployment_recommendations(self) -> Dict[str, List[str]]:
        """Get deployment recommendations grouped by status."""
        return {
            'ready': self.ready_for_deployment,
            'monitor': self.needs_monitoring,
            'reject': self.rejected_strategies
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        summary = f"\n{'='*60}\n"
        summary += f"Portfolio Evaluation Summary\n"
        summary += f"{'='*60}\n"
        summary += f"Timestamp: {self.evaluation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Strategies Evaluated: {self.num_strategies_evaluated}\n\n"

        summary += str(self.portfolio_metrics) + "\n\n"

        summary += f"Correlation Analysis:\n"
        summary += f"  Avg Correlation: {self.correlation_matrix.avg_correlation:.2f}\n"
        summary += f"  Max Correlation: {self.correlation_matrix.max_correlation:.2f}\n"
        summary += f"  Redundant Groups: {len(self.correlation_matrix.redundant_groups)}\n\n"

        summary += f"Deployment Recommendations:\n"
        summary += f"  ✅ Ready: {len(self.ready_for_deployment)}\n"
        summary += f"  ⚠️  Monitor: {len(self.needs_monitoring)}\n"
        summary += f"  ❌ Reject: {len(self.rejected_strategies)}\n\n"

        if self.optimal_allocations:
            summary += "Top Allocations:\n"
            for alloc in sorted(self.optimal_allocations,
                              key=lambda x: x.allocation_weight,
                              reverse=True)[:5]:
                summary += f"  {alloc.strategy_name}: {alloc.allocation_weight*100:.1f}%\n"

        summary += f"{'='*60}\n"

        return summary
