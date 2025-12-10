"""Portfolio risk management and metrics calculation."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from src.portfolio.models import (
    StrategyPortfolioProfile,
    PortfolioMetrics,
    PortfolioAcceptanceCriteria
)

logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """Assesses portfolio-level risk metrics."""

    def __init__(
        self,
        max_portfolio_dd: float = 0.25,
        max_concentration: float = 0.40,  # Max weight in single strategy
        min_diversification: float = 0.7  # Min diversification ratio
    ):
        """
        Initialize portfolio risk manager.

        Args:
            max_portfolio_dd: Maximum allowed portfolio drawdown
            max_concentration: Maximum allocation to single strategy
            min_diversification: Minimum diversification ratio required
        """
        self.max_portfolio_dd = max_portfolio_dd
        self.max_concentration = max_concentration
        self.min_diversification = min_diversification

        logger.info(
            f"Initialized PortfolioRiskManager with max_dd={max_portfolio_dd}, "
            f"max_concentration={max_concentration}, min_diversification={min_diversification}"
        )

    def calculate_portfolio_metrics(
        self,
        strategies: List[StrategyPortfolioProfile],
        weights: np.ndarray
    ) -> PortfolioMetrics:
        """
        Calculate portfolio-level Sharpe, DD, diversification.

        Aggregates individual equity curves using weights.

        Args:
            strategies: List of strategy profiles
            weights: Allocation weights (must sum to 1.0)

        Returns:
            PortfolioMetrics with aggregated performance
        """
        if len(strategies) != len(weights):
            raise ValueError("Number of strategies must match number of weights")

        if not np.isclose(weights.sum(), 1.0):
            logger.warning(f"Weights sum to {weights.sum():.4f}, not 1.0. Normalizing...")
            weights = weights / weights.sum()

        logger.info(f"Calculating portfolio metrics for {len(strategies)} strategies")

        # Build portfolio equity curve
        portfolio_equity = self._build_portfolio_equity_curve(strategies, weights)

        # Calculate portfolio returns
        portfolio_returns = portfolio_equity.pct_change().dropna()

        # Portfolio-level metrics
        portfolio_return = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0]) - 1
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # Annualized
        portfolio_sharpe = self._calculate_sharpe(portfolio_returns)
        portfolio_max_dd = self._calculate_max_drawdown(portfolio_equity)

        # Diversification metrics
        weighted_avg_vol = self._calculate_weighted_avg_volatility(strategies, weights)
        diversification_ratio = portfolio_vol / weighted_avg_vol if weighted_avg_vol > 0 else 1.0

        # Concentration risk (Herfindahl index)
        concentration_risk = np.sum(weights ** 2)

        # Correlation-adjusted Sharpe
        # This is a simplified version - assumes average correlation
        avg_correlation = self._estimate_avg_correlation(strategies)
        correlation_adjusted_sharpe = portfolio_sharpe / np.sqrt(
            1 + (len(strategies) - 1) * avg_correlation
        )

        # Total trades across all strategies
        total_trades = sum(s.num_trades for s in strategies)

        metrics = PortfolioMetrics(
            portfolio_sharpe=portfolio_sharpe,
            portfolio_return=portfolio_return,
            portfolio_max_dd=portfolio_max_dd,
            portfolio_volatility=portfolio_vol,
            diversification_ratio=diversification_ratio,
            concentration_risk=concentration_risk,
            correlation_adjusted_sharpe=correlation_adjusted_sharpe,
            num_strategies=len(strategies),
            total_trades=total_trades,
            portfolio_equity_curve=portfolio_equity
        )

        logger.info(
            f"Portfolio metrics: Sharpe={portfolio_sharpe:.2f}, "
            f"Return={portfolio_return*100:.2f}%, DD={portfolio_max_dd*100:.2f}%, "
            f"DiversificationRatio={diversification_ratio:.2f}"
        )

        return metrics

    def _build_portfolio_equity_curve(
        self,
        strategies: List[StrategyPortfolioProfile],
        weights: np.ndarray
    ) -> pd.Series:
        """
        Build weighted portfolio equity curve.

        Args:
            strategies: List of strategy profiles
            weights: Allocation weights

        Returns:
            Combined equity curve
        """
        # Collect equity curves
        equity_dict = {}
        for i, strategy in enumerate(strategies):
            # Get equity curve from backtest results
            equity_curve = strategy.backtest_results.get('equity_curve')
            if equity_curve is not None:
                equity_dict[strategy.strategy_name] = equity_curve

        if not equity_dict:
            logger.error("No equity curves found in strategies")
            return pd.Series([1.0])

        # Combine into DataFrame
        equity_df = pd.DataFrame(equity_dict)

        # Forward fill missing values (strategies may start at different times)
        equity_df = equity_df.fillna(method='ffill')
        equity_df = equity_df.fillna(1.0)  # Fill initial NaNs with 1.0 (starting capital)

        # Normalize all curves to start at 1.0
        for col in equity_df.columns:
            first_valid = equity_df[col].ne(0).idxmax()
            equity_df[col] = equity_df[col] / equity_df.loc[first_valid, col]

        # Weight and sum
        strategy_names = [s.strategy_name for s in strategies]
        weighted_equity = pd.Series(0.0, index=equity_df.index)

        for i, name in enumerate(strategy_names):
            if name in equity_df.columns:
                weighted_equity += equity_df[name] * weights[i]

        return weighted_equity

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (risk_free_rate / 252)
        if returns.std() == 0:
            return 0.0

        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
        return sharpe

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Equity curve series

        Returns:
            Maximum drawdown (positive value)
        """
        if len(equity_curve) < 2:
            return 0.0

        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = abs(drawdown.min())

        return max_dd

    def _calculate_weighted_avg_volatility(
        self,
        strategies: List[StrategyPortfolioProfile],
        weights: np.ndarray
    ) -> float:
        """
        Calculate weighted average of individual strategy volatilities.

        Args:
            strategies: List of strategy profiles
            weights: Allocation weights

        Returns:
            Weighted average volatility
        """
        volatilities = []
        for strategy in strategies:
            if strategy.returns_series is not None and len(strategy.returns_series) > 1:
                vol = strategy.returns_series.std() * np.sqrt(252)
                volatilities.append(vol)
            else:
                volatilities.append(0.0)

        volatilities = np.array(volatilities)
        weighted_avg = np.dot(weights, volatilities)

        return weighted_avg

    def _estimate_avg_correlation(self, strategies: List[StrategyPortfolioProfile]) -> float:
        """
        Estimate average correlation between strategies.

        Args:
            strategies: List of strategy profiles

        Returns:
            Average correlation
        """
        if len(strategies) < 2:
            return 0.0

        # Build returns dataframe
        returns_dict = {}
        for strategy in strategies:
            if strategy.returns_series is not None and len(strategy.returns_series) > 0:
                returns_dict[strategy.strategy_name] = strategy.returns_series

        if len(returns_dict) < 2:
            return 0.0

        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.fillna(method='ffill').dropna()

        if len(returns_df) < 2:
            return 0.0

        corr_matrix = returns_df.corr()

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.values[mask].mean()

        return avg_corr

    def assess_marginal_risk_contribution(
        self,
        strategies: List[StrategyPortfolioProfile],
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate each strategy's contribution to portfolio risk.

        Uses variance decomposition: MRC_i = w_i * (∂σ_p / ∂w_i)

        Args:
            strategies: List of strategy profiles
            weights: Allocation weights

        Returns:
            Dict mapping strategy name to risk contribution
        """
        if len(strategies) != len(weights):
            raise ValueError("Number of strategies must match number of weights")

        # Build covariance matrix
        returns_dict = {}
        for strategy in strategies:
            if strategy.returns_series is not None and len(strategy.returns_series) > 0:
                returns_dict[strategy.strategy_name] = strategy.returns_series

        if len(returns_dict) < 2:
            # Can't calculate for single strategy
            return {s.strategy_name: weights[i] for i, s in enumerate(strategies)}

        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.fillna(method='ffill').dropna()

        if len(returns_df) < 2:
            return {s.strategy_name: weights[i] for i, s in enumerate(strategies)}

        cov_matrix = returns_df.cov().values

        # Portfolio variance
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(max(0, portfolio_variance))

        if portfolio_vol == 0:
            return {s.strategy_name: 1.0 / len(strategies) for s in strategies}

        # Marginal contribution to risk
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol

        # Risk contribution (marginal * weight)
        risk_contrib = weights * marginal_contrib

        # Normalize to sum to 1.0
        risk_contrib = risk_contrib / risk_contrib.sum()

        risk_contrib_dict = {
            strategies[i].strategy_name: risk_contrib[i]
            for i in range(len(strategies))
        }

        logger.info("Calculated marginal risk contributions")
        for name, contrib in sorted(risk_contrib_dict.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {name}: {contrib*100:.2f}%")

        return risk_contrib_dict

    def check_risk_limits(
        self,
        portfolio_metrics: PortfolioMetrics,
        weights: np.ndarray,
        criteria: Optional[PortfolioAcceptanceCriteria] = None
    ) -> List[str]:
        """
        Check if portfolio violates risk constraints.

        Args:
            portfolio_metrics: Portfolio metrics to check
            weights: Allocation weights
            criteria: Acceptance criteria (uses defaults if None)

        Returns:
            List of violated constraints
        """
        if criteria is None:
            criteria = PortfolioAcceptanceCriteria()

        violations = []

        # Check portfolio-level metrics
        if portfolio_metrics.portfolio_max_dd > criteria.max_portfolio_dd:
            violations.append(
                f"Portfolio drawdown {portfolio_metrics.portfolio_max_dd*100:.1f}% "
                f"exceeds limit {criteria.max_portfolio_dd*100:.1f}%"
            )

        if portfolio_metrics.portfolio_sharpe < criteria.min_portfolio_sharpe:
            violations.append(
                f"Portfolio Sharpe {portfolio_metrics.portfolio_sharpe:.2f} "
                f"below minimum {criteria.min_portfolio_sharpe:.2f}"
            )

        if portfolio_metrics.diversification_ratio > 1.0 / criteria.min_diversification_ratio:
            violations.append(
                f"Diversification ratio {portfolio_metrics.diversification_ratio:.2f} "
                f"indicates insufficient diversification"
            )

        # Check allocation constraints
        max_weight = weights.max()
        if max_weight > criteria.max_single_allocation:
            violations.append(
                f"Single strategy allocation {max_weight*100:.1f}% "
                f"exceeds limit {criteria.max_single_allocation*100:.1f}%"
            )

        if portfolio_metrics.num_strategies < criteria.min_strategies:
            violations.append(
                f"Portfolio has {portfolio_metrics.num_strategies} strategies, "
                f"minimum is {criteria.min_strategies}"
            )

        if portfolio_metrics.num_strategies > criteria.max_strategies:
            violations.append(
                f"Portfolio has {portfolio_metrics.num_strategies} strategies, "
                f"maximum is {criteria.max_strategies}"
            )

        # Check concentration risk
        if portfolio_metrics.concentration_risk > 0.5:
            violations.append(
                f"High concentration risk (HHI={portfolio_metrics.concentration_risk:.2f})"
            )

        if violations:
            logger.warning(f"Portfolio violates {len(violations)} risk constraints")
            for violation in violations:
                logger.warning(f"  - {violation}")
        else:
            logger.info("Portfolio passes all risk constraints")

        return violations
