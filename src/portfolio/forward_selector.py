"""Forward test selector for prioritizing strategies."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

from src.portfolio.models import (
    StrategyPortfolioProfile,
    ForwardTestPriority,
    PortfolioMetrics
)

logger = logging.getLogger(__name__)


class ForwardTestSelector:
    """Prioritizes strategies for forward testing and deployment."""

    def __init__(
        self,
        min_backtest_sharpe: float = 1.0,
        max_correlation_with_deployed: float = 0.6,
        min_regime_coverage: int = 1  # MVP: simplified, Phase 2: use 2+
    ):
        """
        Initialize forward test selector.

        Args:
            min_backtest_sharpe: Minimum Sharpe ratio for consideration
            max_correlation_with_deployed: Max correlation with existing strategies
            min_regime_coverage: Minimum number of regimes strategy should work in
        """
        self.min_backtest_sharpe = min_backtest_sharpe
        self.max_correlation_with_deployed = max_correlation_with_deployed
        self.min_regime_coverage = min_regime_coverage

        logger.info(
            f"Initialized ForwardTestSelector with min_sharpe={min_backtest_sharpe}, "
            f"max_corr={max_correlation_with_deployed}"
        )

    def score_strategies(
        self,
        strategies: List[StrategyPortfolioProfile],
        portfolio_metrics: PortfolioMetrics,
        correlation_matrix: pd.DataFrame,
        regime_coverage: Optional[pd.DataFrame] = None,
        deployed_strategies: Optional[List[str]] = None
    ) -> List[ForwardTestPriority]:
        """
        Score and rank strategies for forward testing.

        Scoring factors:
        1. Individual performance (Sharpe, return, DD)
        2. Portfolio benefit (low correlation, fills regime gap)
        3. Risk characteristics (stable, not overfit)
        4. Novelty (different from deployed strategies)

        Args:
            strategies: List of strategy profiles
            portfolio_metrics: Portfolio-level metrics
            correlation_matrix: Returns correlation matrix
            regime_coverage: Regime performance dataframe (optional, Phase 2)
            deployed_strategies: Currently deployed strategy names (optional)

        Returns:
            List of priority rankings
        """
        logger.info(f"Scoring {len(strategies)} strategies for forward testing")

        priorities = []

        for strategy in strategies:
            # Calculate component scores
            performance_score = self._calculate_performance_score(strategy)
            portfolio_fit_score = self._calculate_portfolio_fit_score(
                strategy, correlation_matrix, deployed_strategies
            )
            novelty_score = self._calculate_novelty_score(
                strategy, deployed_strategies, correlation_matrix
            )
            risk_score = self._calculate_risk_score(strategy)

            # Combined priority score (0-100)
            # Weights: performance (40%), portfolio_fit (30%), novelty (20%), risk (10%)
            priority_score = (
                0.40 * performance_score +
                0.30 * portfolio_fit_score +
                0.20 * novelty_score +
                0.10 * risk_score
            )

            # Determine tier
            if priority_score >= 75:
                tier = "high"
            elif priority_score >= 50:
                tier = "medium"
            else:
                tier = "low"

            # Determine deployment readiness
            readiness, reasons = self._assess_deployment_readiness(
                strategy,
                priority_score,
                correlation_matrix,
                deployed_strategies
            )

            # Create priority object
            priority = ForwardTestPriority(
                strategy_name=strategy.strategy_name,
                priority_score=priority_score,
                priority_tier=tier,
                reasons=reasons,
                deployment_readiness=readiness,
                performance_score=performance_score,
                portfolio_fit_score=portfolio_fit_score,
                novelty_score=novelty_score,
                risk_score=risk_score,
                strategy_sharpe=strategy.sharpe_ratio,
                correlation_with_deployed=self._get_max_correlation_with_deployed(
                    strategy, deployed_strategies, correlation_matrix
                ),
                fills_regime_gap=False  # Phase 2: use regime_coverage
            )

            priorities.append(priority)

        # Sort by priority score (descending)
        priorities.sort(key=lambda x: x.priority_score, reverse=True)

        logger.info(f"Scoring complete. Top strategy: {priorities[0].strategy_name} ({priorities[0].priority_score:.1f})")

        return priorities

    def _calculate_performance_score(self, strategy: StrategyPortfolioProfile) -> float:
        """
        Calculate performance score (0-100).

        Based on Sharpe ratio, return, and drawdown.

        Args:
            strategy: Strategy profile

        Returns:
            Performance score
        """
        # Sharpe component (0-50 points, capped at Sharpe=2.0)
        sharpe_score = min(strategy.sharpe_ratio / 2.0, 1.0) * 50

        # Return component (0-30 points, capped at 50% return)
        return_score = min(strategy.total_return / 0.5, 1.0) * 30

        # Drawdown component (0-20 points, best if DD < 10%)
        dd_score = max(0, 1 - strategy.max_drawdown / 0.10) * 20

        total_score = sharpe_score + return_score + dd_score

        return total_score

    def _calculate_portfolio_fit_score(
        self,
        strategy: StrategyPortfolioProfile,
        correlation_matrix: pd.DataFrame,
        deployed_strategies: Optional[List[str]]
    ) -> float:
        """
        Calculate portfolio fit score (0-100).

        Rewards low correlation with existing strategies.

        Args:
            strategy: Strategy profile
            correlation_matrix: Returns correlation matrix
            deployed_strategies: Deployed strategy names

        Returns:
            Portfolio fit score
        """
        if correlation_matrix.empty or strategy.strategy_name not in correlation_matrix.index:
            return 50.0  # Neutral score if no correlation data

        # Get correlations with all other strategies
        correlations = correlation_matrix.loc[strategy.strategy_name]
        correlations = correlations[correlations.index != strategy.strategy_name]

        if len(correlations) == 0:
            return 100.0  # Perfect fit if only strategy

        # Average correlation
        avg_correlation = correlations.mean()

        # Score: 100 for zero correlation, 0 for perfect correlation
        score = max(0, (1 - avg_correlation) * 100)

        # Bonus penalty if highly correlated with deployed strategies
        if deployed_strategies:
            deployed_correlations = correlations[
                correlations.index.isin(deployed_strategies)
            ]
            if len(deployed_correlations) > 0:
                max_deployed_corr = deployed_correlations.max()
                if max_deployed_corr > self.max_correlation_with_deployed:
                    penalty = (max_deployed_corr - self.max_correlation_with_deployed) * 50
                    score = max(0, score - penalty)

        return score

    def _calculate_novelty_score(
        self,
        strategy: StrategyPortfolioProfile,
        deployed_strategies: Optional[List[str]],
        correlation_matrix: pd.DataFrame
    ) -> float:
        """
        Calculate novelty score (0-100).

        Rewards strategies different from what's already deployed.

        Args:
            strategy: Strategy profile
            deployed_strategies: Deployed strategy names
            correlation_matrix: Returns correlation matrix

        Returns:
            Novelty score
        """
        if not deployed_strategies:
            return 100.0  # Maximum novelty if nothing deployed

        if correlation_matrix.empty or strategy.strategy_name not in correlation_matrix.index:
            return 50.0  # Neutral score

        # Get correlations with deployed strategies
        correlations = correlation_matrix.loc[strategy.strategy_name]
        deployed_corr = correlations[correlations.index.isin(deployed_strategies)]

        if len(deployed_corr) == 0:
            return 100.0

        # Max correlation with any deployed strategy
        max_corr = deployed_corr.max()

        # Score: 100 for zero correlation, 0 for perfect correlation
        score = max(0, (1 - max_corr) * 100)

        return score

    def _calculate_risk_score(self, strategy: StrategyPortfolioProfile) -> float:
        """
        Calculate risk score (0-100).

        Assesses risk characteristics like stable returns, reasonable trade count.

        Args:
            strategy: Strategy profile

        Returns:
            Risk score
        """
        score = 100.0

        # Penalty for high drawdown
        if strategy.max_drawdown > 0.20:
            score -= (strategy.max_drawdown - 0.20) * 100

        # Penalty for too few trades (potential overfitting)
        if strategy.num_trades < 30:
            score -= (30 - strategy.num_trades)

        # Penalty for too many trades (potential data mining)
        if strategy.num_trades > 500:
            score -= (strategy.num_trades - 500) * 0.1

        # Bonus for good win rate
        if strategy.win_rate > 0.50:
            score += 10

        score = max(0, min(100, score))

        return score

    def _get_max_correlation_with_deployed(
        self,
        strategy: StrategyPortfolioProfile,
        deployed_strategies: Optional[List[str]],
        correlation_matrix: pd.DataFrame
    ) -> float:
        """Get maximum correlation with deployed strategies."""
        if not deployed_strategies or correlation_matrix.empty:
            return 0.0

        if strategy.strategy_name not in correlation_matrix.index:
            return 0.0

        correlations = correlation_matrix.loc[strategy.strategy_name]
        deployed_corr = correlations[correlations.index.isin(deployed_strategies)]

        if len(deployed_corr) == 0:
            return 0.0

        return deployed_corr.max()

    def _assess_deployment_readiness(
        self,
        strategy: StrategyPortfolioProfile,
        priority_score: float,
        correlation_matrix: pd.DataFrame,
        deployed_strategies: Optional[List[str]]
    ) -> tuple[str, List[str]]:
        """
        Determine if strategy is ready for deployment.

        Returns:
            Tuple of (readiness status, list of reasons)
        """
        reasons = []

        # Check minimum criteria
        if strategy.sharpe_ratio < self.min_backtest_sharpe:
            reasons.append(f"Sharpe {strategy.sharpe_ratio:.2f} below minimum {self.min_backtest_sharpe:.2f}")

        if strategy.num_trades < 20:
            reasons.append(f"Only {strategy.num_trades} trades - insufficient sample size")

        if strategy.max_drawdown > 0.30:
            reasons.append(f"High drawdown ({strategy.max_drawdown*100:.1f}%)")

        # Check correlation with deployed
        max_deployed_corr = self._get_max_correlation_with_deployed(
            strategy, deployed_strategies, correlation_matrix
        )

        if max_deployed_corr > self.max_correlation_with_deployed:
            reasons.append(
                f"Too correlated ({max_deployed_corr:.2f}) with deployed strategies"
            )

        # Determine readiness
        if len(reasons) == 0 and priority_score >= 70:
            readiness = "ready"
            reasons.append("✓ Passes all criteria")
            reasons.append(f"✓ Strong performance (Sharpe: {strategy.sharpe_ratio:.2f})")
            reasons.append(f"✓ High priority score ({priority_score:.1f})")

        elif len(reasons) == 0 or (len(reasons) <= 1 and priority_score >= 60):
            readiness = "monitor"
            if not reasons:
                reasons.append("Needs monitoring before full deployment")
            reasons.append("Consider forward testing with reduced capital")

        else:
            readiness = "reject"
            if not reasons:
                reasons.append("Does not meet deployment criteria")

        return readiness, reasons

    def apply_acceptance_criteria(
        self,
        priority_list: List[ForwardTestPriority]
    ) -> Dict[str, List[ForwardTestPriority]]:
        """
        Classify strategies into deployment readiness tiers.

        Args:
            priority_list: List of priority rankings

        Returns:
            Dict with 'ready', 'monitor', 'reject' lists
        """
        categorized = {
            'ready': [],
            'monitor': [],
            'reject': []
        }

        for priority in priority_list:
            categorized[priority.deployment_readiness].append(priority)

        logger.info(
            f"Categorized strategies: "
            f"{len(categorized['ready'])} ready, "
            f"{len(categorized['monitor'])} monitor, "
            f"{len(categorized['reject'])} reject"
        )

        return categorized
