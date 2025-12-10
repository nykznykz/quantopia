"""Portfolio allocation optimization."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy.optimize import minimize
import logging

from src.portfolio.models import (
    StrategyPortfolioProfile,
    AllocationRecommendation,
    PortfolioMetrics
)

logger = logging.getLogger(__name__)


class AllocationOptimizer:
    """Optimizes position sizing across strategies."""

    def __init__(
        self,
        optimization_method: str = "max_sharpe",  # "max_sharpe", "equal_weight", "risk_parity"
        allow_leverage: bool = False,
        max_leverage: float = 1.0
    ):
        """
        Initialize allocation optimizer.

        Args:
            optimization_method: Optimization method to use
            allow_leverage: Whether to allow leveraged portfolios
            max_leverage: Maximum leverage (1.0 = no leverage)
        """
        self.optimization_method = optimization_method
        self.allow_leverage = allow_leverage
        self.max_leverage = max_leverage

        logger.info(
            f"Initialized AllocationOptimizer with method={optimization_method}, "
            f"leverage={allow_leverage}, max_leverage={max_leverage}"
        )

    def optimize_weights(
        self,
        strategies: List[StrategyPortfolioProfile],
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Find optimal allocation weights.

        Methods:
        - max_sharpe: Maximize Sharpe ratio (Markowitz)
        - equal_weight: Equal allocation to all strategies
        - risk_parity: Equal risk contribution from each strategy

        Args:
            strategies: List of strategy profiles
            constraints: Additional constraints (optional)

        Returns:
            Optimal weight array
        """
        if not strategies:
            return np.array([])

        logger.info(
            f"Optimizing weights for {len(strategies)} strategies "
            f"using {self.optimization_method}"
        )

        if self.optimization_method == "equal_weight":
            weights = self._equal_weights(strategies)

        elif self.optimization_method == "max_sharpe":
            weights = self._max_sharpe_weights(strategies, constraints)

        elif self.optimization_method == "risk_parity":
            weights = self._risk_parity_weights(strategies)

        else:
            logger.warning(
                f"Unknown optimization method '{self.optimization_method}', "
                "using equal weights"
            )
            weights = self._equal_weights(strategies)

        # Ensure weights are valid
        weights = np.maximum(weights, 0)  # No negative weights
        weights = weights / weights.sum()  # Normalize to sum to 1

        logger.info(f"Optimized weights: {dict(zip([s.strategy_name for s in strategies], weights))}")

        return weights

    def _equal_weights(self, strategies: List[StrategyPortfolioProfile]) -> np.ndarray:
        """
        Equal weight allocation.

        Args:
            strategies: List of strategy profiles

        Returns:
            Equal weight array
        """
        n = len(strategies)
        return np.ones(n) / n

    def _max_sharpe_weights(
        self,
        strategies: List[StrategyPortfolioProfile],
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Markowitz optimization for max Sharpe.

        Args:
            strategies: List of strategy profiles
            constraints: Additional constraints

        Returns:
            Optimal weight array
        """
        # Build expected returns and covariance matrix
        returns_dict = {}
        for strategy in strategies:
            if strategy.returns_series is not None and len(strategy.returns_series) > 0:
                returns_dict[strategy.strategy_name] = strategy.returns_series

        if len(returns_dict) < 2:
            logger.warning("Insufficient data for optimization, using equal weights")
            return self._equal_weights(strategies)

        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.fillna(method='ffill').dropna()

        if len(returns_df) < 10:
            logger.warning(f"Only {len(returns_df)} data points, using equal weights")
            return self._equal_weights(strategies)

        # Expected returns (annualized mean)
        expected_returns = returns_df.mean() * 252

        # Covariance matrix (annualized)
        cov_matrix = returns_df.cov() * 252

        # Match order to strategies
        strategy_names = [s.strategy_name for s in strategies]
        expected_returns = expected_returns[strategy_names].values
        cov_matrix = cov_matrix.loc[strategy_names, strategy_names].values

        # Optimization
        n = len(strategies)
        init_weights = np.ones(n) / n

        # Objective: Negative Sharpe ratio (we minimize)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 0
            sharpe = portfolio_return / portfolio_vol
            return -sharpe  # Negative because we minimize

        # Constraints
        cons = []

        # Weights sum to 1 (or max_leverage)
        target_sum = self.max_leverage if self.allow_leverage else 1.0
        cons.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - target_sum
        })

        # Additional user constraints
        if constraints:
            if 'max_weight' in constraints:
                for i in range(n):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda w, i=i: constraints['max_weight'] - w[i]
                    })

        # Bounds: weights between 0 and 1
        bounds = tuple((0, 1) for _ in range(n))

        # Optimize
        try:
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )

            if result.success:
                optimal_weights = result.x
                portfolio_sharpe = -result.fun
                logger.info(f"Optimization successful. Portfolio Sharpe: {portfolio_sharpe:.2f}")
                return optimal_weights
            else:
                logger.warning(f"Optimization failed: {result.message}. Using equal weights.")
                return self._equal_weights(strategies)

        except Exception as e:
            logger.error(f"Optimization error: {e}. Using equal weights.")
            return self._equal_weights(strategies)

    def _risk_parity_weights(self, strategies: List[StrategyPortfolioProfile]) -> np.ndarray:
        """
        Risk parity allocation (equal risk contribution).

        Uses iterative algorithm to find weights where each strategy
        contributes equally to portfolio risk.

        Args:
            strategies: List of strategy profiles

        Returns:
            Risk parity weight array
        """
        # Build covariance matrix
        returns_dict = {}
        for strategy in strategies:
            if strategy.returns_series is not None and len(strategy.returns_series) > 0:
                returns_dict[strategy.strategy_name] = strategy.returns_series

        if len(returns_dict) < 2:
            logger.warning("Insufficient data for risk parity, using equal weights")
            return self._equal_weights(strategies)

        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.fillna(method='ffill').dropna()

        strategy_names = [s.strategy_name for s in strategies]
        cov_matrix = returns_df.cov().loc[strategy_names, strategy_names].values

        n = len(strategies)

        # Objective: minimize sum of squared differences from equal risk contribution
        def objective(weights):
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            if portfolio_var == 0:
                return 0

            # Marginal risk contribution
            marginal_contrib = np.dot(cov_matrix, weights)

            # Risk contribution
            risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)

            # Target: equal contribution (1/n each)
            target = 1.0 / n

            # Sum of squared deviations from target
            return np.sum((risk_contrib - target) ** 2)

        # Constraints: weights sum to 1
        cons = [{
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        }]

        # Bounds
        bounds = tuple((0, 1) for _ in range(n))

        # Initial guess: equal weights
        init_weights = np.ones(n) / n

        try:
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000}
            )

            if result.success:
                logger.info("Risk parity optimization successful")
                return result.x
            else:
                logger.warning(f"Risk parity optimization failed: {result.message}")
                return self._equal_weights(strategies)

        except Exception as e:
            logger.error(f"Risk parity error: {e}. Using equal weights.")
            return self._equal_weights(strategies)

    def generate_allocation_recommendations(
        self,
        strategies: List[StrategyPortfolioProfile],
        optimal_weights: np.ndarray,
        portfolio_metrics: PortfolioMetrics
    ) -> List[AllocationRecommendation]:
        """
        Generate human-readable allocation recommendations.

        Args:
            strategies: List of strategy profiles
            optimal_weights: Optimal allocation weights
            portfolio_metrics: Portfolio-level metrics

        Returns:
            List of allocation recommendations
        """
        if len(strategies) != len(optimal_weights):
            raise ValueError("Number of strategies must match number of weights")

        recommendations = []

        for i, strategy in enumerate(strategies):
            weight = optimal_weights[i]

            # Generate rationale
            rationale_parts = []

            if weight > 0.25:
                rationale_parts.append("High allocation due to strong performance")
            elif weight > 0.15:
                rationale_parts.append("Moderate allocation")
            elif weight > 0.05:
                rationale_parts.append("Small allocation")
            else:
                rationale_parts.append("Minimal allocation")

            if strategy.sharpe_ratio > 1.5:
                rationale_parts.append("excellent Sharpe ratio")
            elif strategy.sharpe_ratio > 1.0:
                rationale_parts.append("good Sharpe ratio")

            rationale = " - ".join(rationale_parts)

            # Calculate risk contribution (simplified)
            risk_contribution = weight  # Simplified - actual calculation requires covariance

            # Expected Sharpe contribution (weight * strategy Sharpe)
            expected_sharpe_contrib = weight * strategy.sharpe_ratio

            recommendation = AllocationRecommendation(
                strategy_name=strategy.strategy_name,
                allocation_weight=weight,
                allocation_rationale=rationale,
                risk_contribution=risk_contribution,
                expected_sharpe_contribution=expected_sharpe_contrib,
                strategy_sharpe=strategy.sharpe_ratio,
                strategy_return=strategy.total_return,
                correlation_with_portfolio=0.0  # Will be calculated by evaluator
            )

            recommendations.append(recommendation)

        # Sort by weight (descending)
        recommendations.sort(key=lambda x: x.allocation_weight, reverse=True)

        logger.info(f"Generated {len(recommendations)} allocation recommendations")

        return recommendations
