"""
Database package for strategy storage and management.

This package provides persistent storage for:
- Strategy metadata and genealogy
- Backtest results
- Generated code
- Portfolio evaluations
- Forward test queue

Main classes:
- StrategyDatabase: Main interface for all database operations
- Strategy, BacktestResult, StrategyCode, etc.: SQLAlchemy ORM models
"""

from src.database.schema import (
    Base,
    Strategy,
    BacktestResult,
    StrategyCode,
    PortfolioEvaluation,
    StrategyAllocation,
    ForwardTestQueue,
    create_database,
    init_database
)

from src.database.manager import StrategyDatabase

__all__ = [
    # Main interface
    'StrategyDatabase',

    # ORM Models
    'Base',
    'Strategy',
    'BacktestResult',
    'StrategyCode',
    'PortfolioEvaluation',
    'StrategyAllocation',
    'ForwardTestQueue',

    # Utility functions
    'create_database',
    'init_database'
]
