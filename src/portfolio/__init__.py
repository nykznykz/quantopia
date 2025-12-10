"""Portfolio evaluation module for multi-strategy analysis."""

from src.portfolio.models import (
    StrategyPortfolioProfile,
    CorrelationMatrix,
    PortfolioMetrics,
    AllocationRecommendation,
    ForwardTestPriority,
    PortfolioAcceptanceCriteria,
    PortfolioEvaluationResult
)

from src.portfolio.correlation import CorrelationAnalyzer
from src.portfolio.risk_manager import PortfolioRiskManager
from src.portfolio.allocation import AllocationOptimizer
from src.portfolio.forward_selector import ForwardTestSelector
from src.portfolio.evaluator import PortfolioEvaluator

__all__ = [
    # Data models
    'StrategyPortfolioProfile',
    'CorrelationMatrix',
    'PortfolioMetrics',
    'AllocationRecommendation',
    'ForwardTestPriority',
    'PortfolioAcceptanceCriteria',
    'PortfolioEvaluationResult',

    # Components
    'CorrelationAnalyzer',
    'PortfolioRiskManager',
    'AllocationOptimizer',
    'ForwardTestSelector',

    # Main interface
    'PortfolioEvaluator',
]

__version__ = '0.1.0'
