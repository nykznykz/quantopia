"""
Database schema for strategy storage and genealogy tracking.

This module defines SQLAlchemy ORM models for persistent storage of:
- Strategy metadata and genealogy
- Backtest results and performance metrics
- Generated code and validation status
- Portfolio evaluations and allocations
- Forward test queue and priorities
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Text, Boolean, ForeignKey, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Strategy(Base):
    """Strategy metadata and genealogy tracking."""
    __tablename__ = 'strategies'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    strategy_type = Column(String(100), nullable=False)  # trend, mean_reversion, breakout, etc.
    description = Column(Text, nullable=True)

    # Indicators and parameters (stored as JSON)
    indicators = Column(JSON, nullable=False)  # List of indicator names
    parameters = Column(JSON, nullable=True)   # Strategy-specific parameters

    # ML Strategy Support (Phase 1a)
    ml_strategy_type = Column(String(50), default='pure_technical')  # pure_technical | hybrid_ml | pure_ml
    ml_models_used = Column(JSON, nullable=True)  # List of model IDs (e.g., ['XGBoost_direction_v3'])

    # Genealogy tracking (enhanced for Phase 1a)
    parent_id = Column(Integer, ForeignKey('strategies.id'), nullable=True, index=True)
    grandparent_id = Column(Integer, nullable=True)  # Two-hop genealogy
    generation = Column(Integer, default=0)  # 0 for original, increments with each refinement
    refinement_type = Column(String(50), nullable=True)  # refinement, family_member, variation
    exploration_vector = Column(JSON, nullable=True)  # What changed from parent (e.g., {'modification': 'upgraded_model', 'details': '...'})

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    llm_model = Column(String(100), nullable=True)  # Which LLM generated this
    prompt_version = Column(String(50), nullable=True)  # Prompt template version

    # Status tracking
    status = Column(String(50), default='generated')  # generated, coded, tested, approved, rejected

    # Relationships
    parent = relationship('Strategy', remote_side=[id], backref='children')
    backtest_results = relationship('BacktestResult', back_populates='strategy', cascade='all, delete-orphan')
    code_versions = relationship('StrategyCode', back_populates='strategy', cascade='all, delete-orphan')
    allocations = relationship('StrategyAllocation', back_populates='strategy', cascade='all, delete-orphan')
    forward_tests = relationship('ForwardTestQueue', back_populates='strategy', cascade='all, delete-orphan')

    def __repr__(self):
        ml_type = f", ml_type='{self.ml_strategy_type}'" if self.ml_strategy_type != 'pure_technical' else ""
        return f"<Strategy(id={self.id}, name='{self.name}', type='{self.strategy_type}'{ml_type}, status='{self.status}')>"


class BacktestResult(Base):
    """Backtest performance metrics for strategies."""
    __tablename__ = 'backtest_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)

    # Test parameters
    symbol = Column(String(50), nullable=False)
    timeframe = Column(String(20), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)

    # Performance metrics
    total_return_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)

    # Trade statistics
    num_trades = Column(Integer, nullable=True)
    avg_trade_return_pct = Column(Float, nullable=True)
    max_consecutive_wins = Column(Integer, nullable=True)
    max_consecutive_losses = Column(Integer, nullable=True)

    # Equity curve (stored as JSON for space efficiency)
    equity_curve = Column(JSON, nullable=True)  # List of {timestamp, equity} dicts

    # Filter result
    passed_filters = Column(Boolean, default=False)
    filter_category = Column(String(50), nullable=True)  # elite, good, acceptable, poor, rejected
    rejection_reasons = Column(JSON, nullable=True)  # List of failure reasons

    # Metadata
    tested_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    backtest_config = Column(JSON, nullable=True)  # Slippage, fees, etc.

    # Relationships
    strategy = relationship('Strategy', back_populates='backtest_results')

    # Indexes for common queries
    __table_args__ = (
        Index('idx_strategy_symbol_timeframe', 'strategy_id', 'symbol', 'timeframe'),
        Index('idx_sharpe_ratio', 'sharpe_ratio'),
        Index('idx_passed_filters', 'passed_filters'),
    )

    def __repr__(self):
        return f"<BacktestResult(id={self.id}, strategy_id={self.strategy_id}, sharpe={self.sharpe_ratio:.2f if self.sharpe_ratio else 'N/A'})>"


class StrategyCode(Base):
    """Generated code storage and validation tracking."""
    __tablename__ = 'strategy_code'

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)

    # Code content
    code_text = Column(Text, nullable=False)
    version = Column(Integer, default=1)  # Increments if code is regenerated

    # Validation status
    validation_status = Column(String(50), default='pending')  # pending, valid, invalid
    validation_errors = Column(JSON, nullable=True)  # List of validation error messages

    # Code generation metadata
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    llm_model = Column(String(100), nullable=True)
    generation_attempts = Column(Integer, default=1)  # Number of retry attempts

    # Relationships
    strategy = relationship('Strategy', back_populates='code_versions')

    def __repr__(self):
        return f"<StrategyCode(id={self.id}, strategy_id={self.strategy_id}, status='{self.validation_status}', version={self.version})>"


class PortfolioEvaluation(Base):
    """Portfolio-level evaluation results."""
    __tablename__ = 'portfolio_evaluations'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Evaluation metadata
    evaluation_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    num_strategies = Column(Integer, nullable=False)

    # Portfolio metrics
    portfolio_sharpe = Column(Float, nullable=True)
    portfolio_return_pct = Column(Float, nullable=True)
    portfolio_max_dd_pct = Column(Float, nullable=True)
    diversification_ratio = Column(Float, nullable=True)
    avg_correlation = Column(Float, nullable=True)

    # Correlation matrix (stored as JSON)
    correlation_matrix = Column(JSON, nullable=True)  # Dict of strategy pairs to correlation values

    # Risk metrics
    portfolio_volatility = Column(Float, nullable=True)
    risk_adjusted_return = Column(Float, nullable=True)

    # Allocation method used
    allocation_method = Column(String(50), nullable=True)  # equal_weight, max_sharpe, risk_parity

    # Configuration snapshot
    evaluation_config = Column(JSON, nullable=True)

    # Relationships
    allocations = relationship('StrategyAllocation', back_populates='portfolio_evaluation', cascade='all, delete-orphan')

    def __repr__(self):
        return f"<PortfolioEvaluation(id={self.id}, num_strategies={self.num_strategies}, sharpe={self.portfolio_sharpe:.2f if self.portfolio_sharpe else 'N/A'})>"


class StrategyAllocation(Base):
    """Allocation recommendations for strategies in portfolio."""
    __tablename__ = 'strategy_allocations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey('portfolio_evaluations.id'), nullable=False, index=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, index=True)

    # Allocation details
    weight = Column(Float, nullable=False)  # Portfolio weight (0-1)
    recommended_capital = Column(Float, nullable=True)  # Absolute capital allocation

    # Contribution metrics
    marginal_sharpe_contribution = Column(Float, nullable=True)
    marginal_risk_contribution = Column(Float, nullable=True)
    correlation_with_portfolio = Column(Float, nullable=True)

    # Metadata
    allocated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    portfolio_evaluation = relationship('PortfolioEvaluation', back_populates='allocations')
    strategy = relationship('Strategy', back_populates='allocations')

    __table_args__ = (
        Index('idx_evaluation_strategy', 'evaluation_id', 'strategy_id'),
    )

    def __repr__(self):
        return f"<StrategyAllocation(strategy_id={self.strategy_id}, weight={self.weight:.2%})>"


class ForwardTestQueue(Base):
    """Priority queue for forward testing strategies."""
    __tablename__ = 'forward_test_queue'

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False, unique=True, index=True)

    # Priority scoring
    priority_score = Column(Float, nullable=False, index=True)  # Higher = test sooner
    priority_tier = Column(String(50), nullable=True)  # ready, monitor, reject

    # Scoring components
    performance_score = Column(Float, nullable=True)
    portfolio_fit_score = Column(Float, nullable=True)
    novelty_score = Column(Float, nullable=True)

    # Status tracking
    status = Column(String(50), default='queued')  # queued, testing, completed, failed
    queued_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Forward test results (if completed)
    forward_test_sharpe = Column(Float, nullable=True)
    forward_test_return_pct = Column(Float, nullable=True)
    backtest_vs_forward_divergence = Column(Float, nullable=True)

    # Notes and metadata
    notes = Column(Text, nullable=True)
    acceptance_criteria = Column(JSON, nullable=True)

    # Relationships
    strategy = relationship('Strategy', back_populates='forward_tests')

    __table_args__ = (
        Index('idx_priority_status', 'priority_score', 'status'),
    )

    def __repr__(self):
        return f"<ForwardTestQueue(strategy_id={self.strategy_id}, priority={self.priority_score:.2f}, status='{self.status}')>"


def create_database(db_path: str = "data/strategies.db") -> tuple:
    """
    Create database engine and session factory.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Tuple of (engine, SessionLocal)
    """
    engine = create_engine(f'sqlite:///{db_path}', echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal


def init_database(db_path: str = "data/strategies.db"):
    """
    Initialize database with schema.

    Args:
        db_path: Path to SQLite database file
    """
    engine, _ = create_database(db_path)
    print(f"Database initialized at {db_path}")
    print(f"Tables created: {list(Base.metadata.tables.keys())}")
    return engine
