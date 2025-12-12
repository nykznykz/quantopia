"""
Agent-first architecture components for autonomous strategy research.

Phase 1a agents:
- StrategyAgent: Autonomous strategy explorer
- MLQuantAgent: On-demand ML expert
"""

from src.agents.strategy_agent import StrategyAgent
from src.agents.ml_quant_agent import MLQuantAgent

__all__ = [
    'StrategyAgent',
    'MLQuantAgent',
]
