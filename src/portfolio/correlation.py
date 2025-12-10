"""Correlation analysis for portfolio strategies."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import logging

from src.portfolio.models import StrategyPortfolioProfile, CorrelationMatrix

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyzes inter-strategy correlations and redundancy."""

    def __init__(self, min_overlap_threshold: float = 0.7):
        """
        Initialize correlation analyzer.

        Args:
            min_overlap_threshold: Correlation above this = redundant
        """
        self.min_overlap_threshold = min_overlap_threshold
        logger.info(
            f"Initialized CorrelationAnalyzer with overlap_threshold={min_overlap_threshold}"
        )

    def analyze_returns_correlation(
        self,
        strategies: List[StrategyPortfolioProfile]
    ) -> pd.DataFrame:
        """
        Calculate returns correlation matrix.

        Uses equity curve returns resampled to common frequency.
        Handles different length equity curves.

        Args:
            strategies: List of strategy profiles

        Returns:
            Correlation matrix DataFrame
        """
        if len(strategies) < 2:
            logger.warning("Need at least 2 strategies for correlation analysis")
            return pd.DataFrame()

        # Collect all returns series
        returns_dict = {}
        for strategy in strategies:
            if strategy.returns_series is not None and len(strategy.returns_series) > 0:
                returns_dict[strategy.strategy_name] = strategy.returns_series

        if len(returns_dict) < 2:
            logger.warning("Not enough valid returns series for correlation")
            return pd.DataFrame()

        # Combine into DataFrame and align
        returns_df = pd.DataFrame(returns_dict)

        # Forward fill missing values (strategies may have different start dates)
        returns_df = returns_df.fillna(method='ffill')

        # Drop any remaining NaN rows (before all strategies started)
        returns_df = returns_df.dropna()

        if len(returns_df) < 10:
            logger.warning(f"Only {len(returns_df)} overlapping periods - correlation may be unreliable")

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        logger.info(
            f"Calculated correlation matrix for {len(strategies)} strategies. "
            f"Avg correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.2f}"
        )

        return corr_matrix

    def analyze_trade_overlap(
        self,
        strategies: List[StrategyPortfolioProfile]
    ) -> pd.DataFrame:
        """
        Calculate temporal trade overlap.

        Returns matrix where overlap[i,j] = % of time both strategies
        have positions open simultaneously.

        Args:
            strategies: List of strategy profiles

        Returns:
            Overlap matrix DataFrame
        """
        if len(strategies) < 2:
            return pd.DataFrame()

        strategy_names = [s.strategy_name for s in strategies]
        n = len(strategies)

        # Initialize overlap matrix
        overlap_matrix = np.zeros((n, n))

        # Calculate pairwise overlaps
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    overlap = self._calculate_period_overlap(
                        strategies[i].active_periods,
                        strategies[j].active_periods
                    )
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap

        overlap_df = pd.DataFrame(
            overlap_matrix,
            index=strategy_names,
            columns=strategy_names
        )

        logger.info(f"Calculated trade overlap matrix. Avg overlap: {overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)].mean():.2f}")

        return overlap_df

    def _calculate_period_overlap(
        self,
        periods_a: List[Tuple],
        periods_b: List[Tuple]
    ) -> float:
        """
        Calculate overlap between two sets of time periods.

        Args:
            periods_a: List of (start, end) tuples for strategy A
            periods_b: List of (start, end) tuples for strategy B

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not periods_a or not periods_b:
            return 0.0

        # Convert periods to sorted events
        events_a = []
        for start, end in periods_a:
            events_a.append((start, 1))   # Position open
            events_a.append((end, -1))     # Position close

        events_b = []
        for start, end in periods_b:
            events_b.append((start, 1))
            events_b.append((end, -1))

        events_a.sort()
        events_b.sort()

        # Calculate total time both are in position
        overlap_time = 0
        i, j = 0, 0
        count_a, count_b = 0, 0
        last_time = None

        all_events = sorted(events_a + events_b, key=lambda x: x[0])

        for time, event_type in all_events:
            # Check if both were in position since last event
            if last_time is not None and count_a > 0 and count_b > 0:
                overlap_time += (time - last_time).total_seconds()

            # Update counts
            if (time, event_type) in events_a:
                count_a += event_type
            if (time, event_type) in events_b:
                count_b += event_type

            last_time = time

        # Calculate total time either was in position (union)
        total_time_a = sum((end - start).total_seconds() for start, end in periods_a)
        total_time_b = sum((end - start).total_seconds() for start, end in periods_b)
        union_time = total_time_a + total_time_b - overlap_time

        if union_time == 0:
            return 0.0

        return overlap_time / union_time

    def identify_redundant_strategies(
        self,
        correlation_matrix: pd.DataFrame,
        overlap_matrix: pd.DataFrame
    ) -> List[List[str]]:
        """
        Group highly correlated/overlapping strategies.

        Uses clustering algorithm to identify redundant groups.

        Args:
            correlation_matrix: Returns correlation matrix
            overlap_matrix: Trade overlap matrix

        Returns:
            List of strategy groups that are too similar
        """
        if correlation_matrix.empty:
            return []

        # Combine correlation and overlap for similarity measure
        if not overlap_matrix.empty:
            # Average of correlation and overlap
            similarity = (correlation_matrix.values + overlap_matrix.values) / 2
        else:
            similarity = correlation_matrix.values

        # Convert similarity to distance
        distance = 1 - similarity

        # Ensure diagonal is zero
        np.fill_diagonal(distance, 0)

        # Convert to condensed distance matrix for clustering
        distance_condensed = squareform(distance, checks=False)

        # Hierarchical clustering
        linkage = hierarchy.linkage(distance_condensed, method='average')

        # Cut tree at threshold
        distance_threshold = 1 - self.min_overlap_threshold
        clusters = hierarchy.fcluster(linkage, distance_threshold, criterion='distance')

        # Group strategies by cluster
        strategy_names = correlation_matrix.index.tolist()
        redundant_groups = []

        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            cluster_members = [
                strategy_names[i]
                for i, c in enumerate(clusters)
                if c == cluster_id
            ]
            if len(cluster_members) > 1:
                redundant_groups.append(cluster_members)

        if redundant_groups:
            logger.info(f"Identified {len(redundant_groups)} redundant strategy groups")
            for i, group in enumerate(redundant_groups):
                logger.info(f"  Group {i+1}: {', '.join(group)}")
        else:
            logger.info("No redundant strategy groups found")

        return redundant_groups

    def calculate_diversification_score(
        self,
        strategies: List[StrategyPortfolioProfile],
        weights: np.ndarray
    ) -> float:
        """
        Calculate portfolio diversification ratio.

        DR = portfolio_volatility / weighted_avg_strategy_volatility
        DR < 1 indicates diversification benefit.

        Args:
            strategies: List of strategy profiles
            weights: Allocation weights

        Returns:
            Diversification ratio
        """
        if len(strategies) != len(weights):
            raise ValueError("Number of strategies must match number of weights")

        # Get volatilities from returns series
        volatilities = []
        for strategy in strategies:
            if strategy.returns_series is not None and len(strategy.returns_series) > 0:
                vol = strategy.returns_series.std()
                volatilities.append(vol)
            else:
                volatilities.append(0.0)

        volatilities = np.array(volatilities)

        # Weighted average volatility
        weighted_avg_vol = np.dot(weights, volatilities)

        if weighted_avg_vol == 0:
            return 1.0

        # Get correlation matrix
        corr_matrix = self.analyze_returns_correlation(strategies)

        if corr_matrix.empty:
            # No diversification benefit if we can't calculate correlation
            return 1.0

        # Portfolio volatility: sqrt(w^T * Cov * w)
        # Cov = diag(vol) * Corr * diag(vol)
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix.values
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(max(0, portfolio_variance))

        if weighted_avg_vol == 0:
            return 1.0

        diversification_ratio = portfolio_vol / weighted_avg_vol

        logger.info(
            f"Diversification ratio: {diversification_ratio:.2f} "
            f"(portfolio vol: {portfolio_vol:.4f}, weighted avg vol: {weighted_avg_vol:.4f})"
        )

        return diversification_ratio

    def analyze(
        self,
        strategies: List[StrategyPortfolioProfile]
    ) -> CorrelationMatrix:
        """
        Perform complete correlation analysis.

        Args:
            strategies: List of strategy profiles

        Returns:
            CorrelationMatrix with all analysis results
        """
        logger.info(f"Performing correlation analysis on {len(strategies)} strategies")

        # Calculate correlation and overlap matrices
        returns_corr = self.analyze_returns_correlation(strategies)
        overlap = self.analyze_trade_overlap(strategies)

        # Identify redundant groups
        redundant_groups = self.identify_redundant_strategies(
            returns_corr,
            overlap
        )

        # Create and return result
        result = CorrelationMatrix(
            returns_correlation=returns_corr,
            trade_overlap_matrix=overlap,
            redundant_groups=redundant_groups
        )

        logger.info(f"Correlation analysis complete. Avg correlation: {result.avg_correlation:.2f}")

        return result
