"""
Database manager for strategy storage and retrieval.

This module provides the StrategyDatabase class which handles all database
operations for strategies, backtest results, portfolio evaluations, and more.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from sqlalchemy import desc, and_, or_, func
from sqlalchemy.orm import Session

from src.database.schema import (
    Base, Strategy, BacktestResult, StrategyCode, PortfolioEvaluation,
    StrategyAllocation, ForwardTestQueue, create_database
)


class StrategyDatabase:
    """
    Main interface for strategy database operations.

    Handles storage and retrieval of:
    - Strategy metadata and genealogy
    - Backtest results
    - Generated code
    - Portfolio evaluations
    - Forward test queue
    """

    def __init__(self, db_path: str = "data/strategies.db", echo: bool = False):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            echo: If True, log all SQL statements (useful for debugging)
        """
        # Ensure data directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.engine, self.SessionLocal = create_database(db_path)

        if echo:
            from sqlalchemy import create_engine
            self.engine = create_engine(f'sqlite:///{db_path}', echo=True)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # ============================================================================
    # Strategy Storage
    # ============================================================================

    def store_strategy(
        self,
        name: str,
        strategy_type: str,
        indicators: List[str],
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        parent_id: Optional[int] = None,
        grandparent_id: Optional[int] = None,
        generation: int = 0,
        refinement_type: Optional[str] = None,
        exploration_vector: Optional[Dict[str, Any]] = None,
        ml_strategy_type: str = 'pure_technical',
        ml_models_used: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        prompt_version: Optional[str] = None,
        status: str = 'generated'
    ) -> int:
        """
        Store a new strategy in the database.

        Args:
            name: Unique strategy name
            strategy_type: Type of strategy (trend, mean_reversion, etc.)
            indicators: List of indicator names used
            description: Optional strategy description
            parameters: Optional strategy-specific parameters
            parent_id: Optional parent strategy ID for genealogy
            generation: Generation number (0 for original)
            refinement_type: Type of refinement (refinement, family_member, variation)
            llm_model: LLM model used to generate strategy
            prompt_version: Prompt template version
            status: Strategy status (generated, coded, tested, etc.)

        Returns:
            Strategy ID

        Raises:
            ValueError: If strategy name already exists
        """
        session = self.get_session()
        try:
            # Check for duplicate name
            existing = session.query(Strategy).filter_by(name=name).first()
            if existing:
                raise ValueError(f"Strategy with name '{name}' already exists (ID: {existing.id})")

            strategy = Strategy(
                name=name,
                strategy_type=strategy_type,
                description=description,
                indicators=indicators,
                parameters=parameters or {},
                parent_id=parent_id,
                grandparent_id=grandparent_id,
                generation=generation,
                refinement_type=refinement_type,
                exploration_vector=exploration_vector or {},
                ml_strategy_type=ml_strategy_type,
                ml_models_used=ml_models_used or [],
                llm_model=llm_model,
                prompt_version=prompt_version,
                status=status,
                created_at=datetime.utcnow()
            )

            session.add(strategy)
            session.commit()
            session.refresh(strategy)
            return strategy.id
        finally:
            session.close()

    def store_strategy_code(
        self,
        strategy_id: int,
        code_text: str,
        validation_status: str = 'valid',
        validation_errors: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        generation_attempts: int = 1
    ) -> int:
        """
        Store generated code for a strategy.

        Args:
            strategy_id: ID of the strategy
            code_text: Generated Python code
            validation_status: Validation status (pending, valid, invalid)
            validation_errors: List of validation error messages
            llm_model: LLM model used for code generation
            generation_attempts: Number of generation attempts

        Returns:
            StrategyCode ID
        """
        session = self.get_session()
        try:
            # Get latest version number for this strategy
            latest = session.query(func.max(StrategyCode.version)).filter_by(
                strategy_id=strategy_id
            ).scalar()
            version = (latest or 0) + 1

            code = StrategyCode(
                strategy_id=strategy_id,
                code_text=code_text,
                version=version,
                validation_status=validation_status,
                validation_errors=validation_errors or [],
                llm_model=llm_model,
                generation_attempts=generation_attempts,
                generated_at=datetime.utcnow()
            )

            session.add(code)
            session.commit()
            session.refresh(code)

            # Update strategy status
            strategy = session.query(Strategy).get(strategy_id)
            if strategy and validation_status == 'valid':
                strategy.status = 'coded'
                session.commit()

            return code.id
        finally:
            session.close()

    def update_strategy_status(self, strategy_id: int, status: str):
        """
        Update strategy status.

        Args:
            strategy_id: Strategy ID
            status: New status (generated, coded, tested, approved, rejected)
        """
        session = self.get_session()
        try:
            strategy = session.query(Strategy).get(strategy_id)
            if strategy:
                strategy.status = status
                session.commit()
        finally:
            session.close()

    # ============================================================================
    # Backtest Results Storage
    # ============================================================================

    def store_backtest_results(
        self,
        strategy_id: int,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        metrics: Dict[str, Any],
        equity_curve: Optional[List[Dict[str, Any]]] = None,
        passed_filters: bool = False,
        filter_category: Optional[str] = None,
        rejection_reasons: Optional[List[str]] = None,
        backtest_config: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store backtest results for a strategy.

        Args:
            strategy_id: Strategy ID
            symbol: Trading symbol
            timeframe: Timeframe (1h, 4h, 1d, etc.)
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
            metrics: Performance metrics dictionary
            equity_curve: Optional equity curve data
            passed_filters: Whether strategy passed filters
            filter_category: Filter category (elite, good, acceptable, poor, rejected)
            rejection_reasons: List of rejection reasons
            backtest_config: Backtest configuration (slippage, fees, etc.)

        Returns:
            BacktestResult ID
        """
        session = self.get_session()
        try:
            result = BacktestResult(
                strategy_id=strategy_id,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                total_return_pct=metrics.get('total_return_pct'),
                sharpe_ratio=metrics.get('sharpe_ratio'),
                max_drawdown_pct=metrics.get('max_drawdown_pct'),
                win_rate=metrics.get('win_rate'),
                profit_factor=metrics.get('profit_factor'),
                num_trades=metrics.get('num_trades'),
                avg_trade_return_pct=metrics.get('avg_trade_return_pct'),
                max_consecutive_wins=metrics.get('max_consecutive_wins'),
                max_consecutive_losses=metrics.get('max_consecutive_losses'),
                equity_curve=equity_curve,
                passed_filters=passed_filters,
                filter_category=filter_category,
                rejection_reasons=rejection_reasons or [],
                backtest_config=backtest_config or {},
                tested_at=datetime.utcnow()
            )

            session.add(result)
            session.commit()
            session.refresh(result)

            # Update strategy status
            strategy = session.query(Strategy).get(strategy_id)
            if strategy:
                if passed_filters:
                    strategy.status = 'approved'
                else:
                    strategy.status = 'tested'
                session.commit()

            return result.id
        finally:
            session.close()

    # ============================================================================
    # Portfolio Evaluation Storage
    # ============================================================================

    def store_portfolio_evaluation(
        self,
        num_strategies: int,
        portfolio_metrics: Dict[str, Any],
        correlation_matrix: Optional[Dict[str, float]] = None,
        allocation_method: str = 'equal_weight',
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store portfolio evaluation results.

        Args:
            num_strategies: Number of strategies in portfolio
            portfolio_metrics: Portfolio-level metrics
            correlation_matrix: Correlation matrix data
            allocation_method: Allocation method used
            evaluation_config: Configuration snapshot

        Returns:
            PortfolioEvaluation ID
        """
        session = self.get_session()
        try:
            evaluation = PortfolioEvaluation(
                num_strategies=num_strategies,
                portfolio_sharpe=portfolio_metrics.get('portfolio_sharpe'),
                portfolio_return_pct=portfolio_metrics.get('portfolio_return_pct'),
                portfolio_max_dd_pct=portfolio_metrics.get('portfolio_max_dd_pct'),
                diversification_ratio=portfolio_metrics.get('diversification_ratio'),
                avg_correlation=portfolio_metrics.get('avg_correlation'),
                correlation_matrix=correlation_matrix or {},
                portfolio_volatility=portfolio_metrics.get('portfolio_volatility'),
                risk_adjusted_return=portfolio_metrics.get('risk_adjusted_return'),
                allocation_method=allocation_method,
                evaluation_config=evaluation_config or {},
                evaluation_timestamp=datetime.utcnow()
            )

            session.add(evaluation)
            session.commit()
            session.refresh(evaluation)
            return evaluation.id
        finally:
            session.close()

    def store_strategy_allocation(
        self,
        evaluation_id: int,
        strategy_id: int,
        weight: float,
        recommended_capital: Optional[float] = None,
        marginal_sharpe: Optional[float] = None,
        marginal_risk: Optional[float] = None,
        correlation_with_portfolio: Optional[float] = None
    ) -> int:
        """
        Store strategy allocation recommendation.

        Args:
            evaluation_id: Portfolio evaluation ID
            strategy_id: Strategy ID
            weight: Portfolio weight (0-1)
            recommended_capital: Absolute capital allocation
            marginal_sharpe: Marginal Sharpe contribution
            marginal_risk: Marginal risk contribution
            correlation_with_portfolio: Correlation with rest of portfolio

        Returns:
            StrategyAllocation ID
        """
        session = self.get_session()
        try:
            allocation = StrategyAllocation(
                evaluation_id=evaluation_id,
                strategy_id=strategy_id,
                weight=weight,
                recommended_capital=recommended_capital,
                marginal_sharpe_contribution=marginal_sharpe,
                marginal_risk_contribution=marginal_risk,
                correlation_with_portfolio=correlation_with_portfolio,
                allocated_at=datetime.utcnow()
            )

            session.add(allocation)
            session.commit()
            session.refresh(allocation)
            return allocation.id
        finally:
            session.close()

    # ============================================================================
    # Forward Test Queue
    # ============================================================================

    def add_to_forward_test_queue(
        self,
        strategy_id: int,
        priority_score: float,
        priority_tier: str,
        performance_score: Optional[float] = None,
        portfolio_fit_score: Optional[float] = None,
        novelty_score: Optional[float] = None,
        notes: Optional[str] = None,
        acceptance_criteria: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add strategy to forward test queue.

        Args:
            strategy_id: Strategy ID
            priority_score: Overall priority score (higher = test sooner)
            priority_tier: Priority tier (ready, monitor, reject)
            performance_score: Individual performance score
            portfolio_fit_score: Portfolio fit score
            novelty_score: Novelty/uniqueness score
            notes: Optional notes
            acceptance_criteria: Acceptance criteria

        Returns:
            ForwardTestQueue ID
        """
        session = self.get_session()
        try:
            # Check if already in queue
            existing = session.query(ForwardTestQueue).filter_by(strategy_id=strategy_id).first()
            if existing:
                # Update existing entry
                existing.priority_score = priority_score
                existing.priority_tier = priority_tier
                existing.performance_score = performance_score
                existing.portfolio_fit_score = portfolio_fit_score
                existing.novelty_score = novelty_score
                existing.notes = notes
                existing.acceptance_criteria = acceptance_criteria or {}
                session.commit()
                return existing.id

            queue_entry = ForwardTestQueue(
                strategy_id=strategy_id,
                priority_score=priority_score,
                priority_tier=priority_tier,
                performance_score=performance_score,
                portfolio_fit_score=portfolio_fit_score,
                novelty_score=novelty_score,
                status='queued',
                notes=notes,
                acceptance_criteria=acceptance_criteria or {},
                queued_at=datetime.utcnow()
            )

            session.add(queue_entry)
            session.commit()
            session.refresh(queue_entry)
            return queue_entry.id
        finally:
            session.close()

    def get_forward_test_queue(
        self,
        status: str = 'queued',
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get strategies from forward test queue.

        Args:
            status: Queue status filter (queued, testing, completed, failed)
            limit: Maximum number of results

        Returns:
            List of queue entries with strategy details
        """
        session = self.get_session()
        try:
            query = session.query(ForwardTestQueue, Strategy).join(
                Strategy, ForwardTestQueue.strategy_id == Strategy.id
            ).filter(ForwardTestQueue.status == status).order_by(
                desc(ForwardTestQueue.priority_score)
            )

            if limit:
                query = query.limit(limit)

            results = []
            for queue_entry, strategy in query.all():
                results.append({
                    'queue_id': queue_entry.id,
                    'strategy_id': strategy.id,
                    'strategy_name': strategy.name,
                    'strategy_type': strategy.strategy_type,
                    'priority_score': queue_entry.priority_score,
                    'priority_tier': queue_entry.priority_tier,
                    'status': queue_entry.status,
                    'queued_at': queue_entry.queued_at,
                    'notes': queue_entry.notes
                })

            return results
        finally:
            session.close()

    # ============================================================================
    # Query Methods
    # ============================================================================

    def get_strategy_by_id(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """Get strategy by ID."""
        session = self.get_session()
        try:
            strategy = session.query(Strategy).get(strategy_id)
            if not strategy:
                return None

            return {
                'id': strategy.id,
                'name': strategy.name,
                'strategy_type': strategy.strategy_type,
                'description': strategy.description,
                'indicators': strategy.indicators,
                'parameters': strategy.parameters,
                'parent_id': strategy.parent_id,
                'generation': strategy.generation,
                'refinement_type': strategy.refinement_type,
                'status': strategy.status,
                'created_at': strategy.created_at,
                'llm_model': strategy.llm_model
            }
        finally:
            session.close()

    def get_strategy_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get strategy by name."""
        session = self.get_session()
        try:
            strategy = session.query(Strategy).filter_by(name=name).first()
            if not strategy:
                return None

            return {
                'id': strategy.id,
                'name': strategy.name,
                'strategy_type': strategy.strategy_type,
                'description': strategy.description,
                'indicators': strategy.indicators,
                'parameters': strategy.parameters,
                'parent_id': strategy.parent_id,
                'generation': strategy.generation,
                'status': strategy.status,
                'created_at': strategy.created_at
            }
        finally:
            session.close()

    def get_recent_strategies(
        self,
        days: int = 7,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recently created strategies.

        Args:
            days: Number of days to look back
            limit: Maximum number of results

        Returns:
            List of strategy dictionaries
        """
        session = self.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            query = session.query(Strategy).filter(
                Strategy.created_at >= cutoff
            ).order_by(desc(Strategy.created_at))

            if limit:
                query = query.limit(limit)

            strategies = []
            for s in query.all():
                strategies.append({
                    'id': s.id,
                    'name': s.name,
                    'strategy_type': s.strategy_type,
                    'description': s.description,
                    'indicators': s.indicators,
                    'status': s.status,
                    'created_at': s.created_at
                })

            return strategies
        finally:
            session.close()

    def get_strategy_genealogy(self, strategy_id: int) -> Dict[str, Any]:
        """
        Get complete genealogy (ancestors and descendants) of a strategy.

        Args:
            strategy_id: Strategy ID

        Returns:
            Dictionary with ancestors and descendants lists
        """
        session = self.get_session()
        try:
            strategy = session.query(Strategy).get(strategy_id)
            if not strategy:
                return {'ancestors': [], 'descendants': [], 'current': None}

            # Get ancestors (walk up parent chain)
            ancestors = []
            current = strategy
            while current.parent_id:
                parent = session.query(Strategy).get(current.parent_id)
                if not parent:
                    break
                ancestors.append({
                    'id': parent.id,
                    'name': parent.name,
                    'strategy_type': parent.strategy_type,
                    'generation': parent.generation,
                    'created_at': parent.created_at
                })
                current = parent

            # Get descendants (all children recursively)
            def get_children(parent_id):
                children = session.query(Strategy).filter_by(parent_id=parent_id).all()
                result = []
                for child in children:
                    result.append({
                        'id': child.id,
                        'name': child.name,
                        'strategy_type': child.strategy_type,
                        'generation': child.generation,
                        'refinement_type': child.refinement_type,
                        'created_at': child.created_at,
                        'status': child.status
                    })
                    # Recursive call for grandchildren
                    result.extend(get_children(child.id))
                return result

            descendants = get_children(strategy_id)

            return {
                'current': {
                    'id': strategy.id,
                    'name': strategy.name,
                    'strategy_type': strategy.strategy_type,
                    'generation': strategy.generation,
                    'created_at': strategy.created_at,
                    'status': strategy.status
                },
                'ancestors': ancestors,
                'descendants': descendants
            }
        finally:
            session.close()

    def get_failure_patterns(
        self,
        days: int = 30,
        min_occurrences: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Analyze historical failure patterns.

        Args:
            days: Number of days to analyze
            min_occurrences: Minimum occurrences to consider a pattern

        Returns:
            List of failure pattern dictionaries
        """
        session = self.get_session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            # Query failed backtest results
            failed_results = session.query(
                BacktestResult, Strategy
            ).join(
                Strategy, BacktestResult.strategy_id == Strategy.id
            ).filter(
                and_(
                    BacktestResult.passed_filters == False,
                    BacktestResult.tested_at >= cutoff
                )
            ).all()

            # Aggregate failure reasons
            failure_counts = {}
            for result, strategy in failed_results:
                for reason in (result.rejection_reasons or []):
                    if reason not in failure_counts:
                        failure_counts[reason] = {
                            'reason': reason,
                            'count': 0,
                            'strategy_types': [],
                            'avg_sharpe': []
                        }
                    failure_counts[reason]['count'] += 1
                    failure_counts[reason]['strategy_types'].append(strategy.strategy_type)
                    if result.sharpe_ratio is not None:
                        failure_counts[reason]['avg_sharpe'].append(result.sharpe_ratio)

            # Filter and format results
            patterns = []
            for reason, data in failure_counts.items():
                if data['count'] >= min_occurrences:
                    patterns.append({
                        'failure_reason': reason,
                        'occurrences': data['count'],
                        'common_strategy_types': list(set(data['strategy_types'])),
                        'avg_sharpe': sum(data['avg_sharpe']) / len(data['avg_sharpe']) if data['avg_sharpe'] else None
                    })

            # Sort by occurrence count
            patterns.sort(key=lambda x: x['occurrences'], reverse=True)
            return patterns
        finally:
            session.close()

    def query_strategies(
        self,
        strategy_type: Optional[str] = None,
        status: Optional[str] = None,
        min_sharpe: Optional[float] = None,
        passed_filters: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Flexible strategy query interface.

        Args:
            strategy_type: Filter by strategy type
            status: Filter by status
            min_sharpe: Minimum Sharpe ratio from latest backtest
            passed_filters: Filter by whether passed filters
            limit: Maximum number of results

        Returns:
            List of strategy dictionaries with latest backtest results
        """
        session = self.get_session()
        try:
            query = session.query(Strategy)

            # Apply filters
            if strategy_type:
                query = query.filter(Strategy.strategy_type == strategy_type)
            if status:
                query = query.filter(Strategy.status == status)

            # Order by creation date
            query = query.order_by(desc(Strategy.created_at))

            if limit:
                query = query.limit(limit)

            results = []
            for strategy in query.all():
                # Get latest backtest result
                latest_backtest = session.query(BacktestResult).filter_by(
                    strategy_id=strategy.id
                ).order_by(desc(BacktestResult.tested_at)).first()

                # Apply backtest filters
                if min_sharpe is not None:
                    if not latest_backtest or latest_backtest.sharpe_ratio is None or latest_backtest.sharpe_ratio < min_sharpe:
                        continue

                if passed_filters is not None:
                    if not latest_backtest or latest_backtest.passed_filters != passed_filters:
                        continue

                result = {
                    'id': strategy.id,
                    'name': strategy.name,
                    'strategy_type': strategy.strategy_type,
                    'description': strategy.description,
                    'indicators': strategy.indicators,
                    'status': strategy.status,
                    'created_at': strategy.created_at,
                    'generation': strategy.generation,
                    'parent_id': strategy.parent_id
                }

                if latest_backtest:
                    result['latest_backtest'] = {
                        'sharpe_ratio': latest_backtest.sharpe_ratio,
                        'total_return_pct': latest_backtest.total_return_pct,
                        'max_drawdown_pct': latest_backtest.max_drawdown_pct,
                        'num_trades': latest_backtest.num_trades,
                        'passed_filters': latest_backtest.passed_filters,
                        'filter_category': latest_backtest.filter_category,
                        'tested_at': latest_backtest.tested_at
                    }

                results.append(result)

            return results
        finally:
            session.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        session = self.get_session()
        try:
            return {
                'total_strategies': session.query(Strategy).count(),
                'total_backtests': session.query(BacktestResult).count(),
                'strategies_by_status': dict(
                    session.query(Strategy.status, func.count(Strategy.id))
                    .group_by(Strategy.status).all()
                ),
                'approved_strategies': session.query(Strategy).filter_by(status='approved').count(),
                'strategies_by_type': dict(
                    session.query(Strategy.strategy_type, func.count(Strategy.id))
                    .group_by(Strategy.strategy_type).all()
                ),
                'avg_generation': session.query(func.avg(Strategy.generation)).scalar() or 0,
                'portfolio_evaluations': session.query(PortfolioEvaluation).count(),
                'forward_test_queue_size': session.query(ForwardTestQueue).filter_by(status='queued').count()
            }
        finally:
            session.close()

    # ============================================================================
    # Agent-Specific Query Methods (Phase 1a)
    # ============================================================================

    def get_top_strategies(
        self,
        metric: str = 'sharpe_ratio',
        limit: int = 10,
        ml_strategy_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top performing strategies by metric.

        Args:
            metric: Metric to sort by ('sharpe_ratio', 'total_return_pct', etc.)
            limit: Number of results
            ml_strategy_type: Filter by ML type (pure_technical, hybrid_ml, pure_ml)

        Returns:
            List of strategy dicts with backtest results
        """
        session = self.get_session()
        try:
            query = session.query(Strategy, BacktestResult).join(
                BacktestResult, Strategy.id == BacktestResult.strategy_id
            ).filter(BacktestResult.passed_filters == True)

            if ml_strategy_type:
                query = query.filter(Strategy.ml_strategy_type == ml_strategy_type)

            # Sort by metric
            if metric == 'sharpe_ratio':
                query = query.order_by(desc(BacktestResult.sharpe_ratio))
            elif metric == 'total_return_pct':
                query = query.order_by(desc(BacktestResult.total_return_pct))
            elif metric == 'win_rate':
                query = query.order_by(desc(BacktestResult.win_rate))

            query = query.limit(limit)

            results = []
            for strategy, backtest in query.all():
                results.append({
                    'id': strategy.id,
                    'name': strategy.name,
                    'strategy_type': strategy.strategy_type,
                    'ml_strategy_type': strategy.ml_strategy_type,
                    'ml_models_used': strategy.ml_models_used,
                    'indicators': strategy.indicators,
                    'created_at': strategy.created_at,
                    'metrics': {
                        'sharpe_ratio': backtest.sharpe_ratio,
                        'total_return_pct': backtest.total_return_pct,
                        'max_drawdown_pct': backtest.max_drawdown_pct,
                        'win_rate': backtest.win_rate,
                        'num_trades': backtest.num_trades
                    }
                })

            return results
        finally:
            session.close()

    def search_by_model(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Search for strategies using a specific ML model.

        Args:
            model_id: Model ID (e.g., 'XGBoost_direction_v3')

        Returns:
            List of strategy dicts with performance metrics
        """
        session = self.get_session()
        try:
            # Find strategies where ml_models_used contains the model_id
            strategies = session.query(Strategy).filter(
                Strategy.ml_models_used.contains([model_id])
            ).all()

            results = []
            for strategy in strategies:
                # Get latest backtest
                backtest = session.query(BacktestResult).filter_by(
                    strategy_id=strategy.id
                ).order_by(desc(BacktestResult.tested_at)).first()

                result = {
                    'id': strategy.id,
                    'name': strategy.name,
                    'strategy_type': strategy.strategy_type,
                    'ml_strategy_type': strategy.ml_strategy_type,
                    'ml_models_used': strategy.ml_models_used,
                    'status': strategy.status,
                    'created_at': strategy.created_at
                }

                if backtest:
                    result['metrics'] = {
                        'sharpe_ratio': backtest.sharpe_ratio,
                        'total_return_pct': backtest.total_return_pct,
                        'passed_filters': backtest.passed_filters
                    }

                results.append(result)

            return results
        finally:
            session.close()

    def get_underexplored_areas(self) -> Dict[str, Any]:
        """
        Identify underexplored areas in the strategy space.

        Returns:
            Dict with counts by strategy type, ML type, indicators, etc.
        """
        session = self.get_session()
        try:
            # Count by strategy logic type
            by_logic_type = dict(
                session.query(Strategy.strategy_type, func.count(Strategy.id))
                .group_by(Strategy.strategy_type).all()
            )

            # Count by ML type
            by_ml_type = dict(
                session.query(Strategy.ml_strategy_type, func.count(Strategy.id))
                .group_by(Strategy.ml_strategy_type).all()
            )

            # Get average metrics by ML type
            ml_performance = {}
            for ml_type in ['pure_technical', 'hybrid_ml', 'pure_ml']:
                avg_sharpe = session.query(func.avg(BacktestResult.sharpe_ratio)).join(
                    Strategy
                ).filter(Strategy.ml_strategy_type == ml_type).scalar()

                ml_performance[ml_type] = {
                    'count': by_ml_type.get(ml_type, 0),
                    'avg_sharpe': avg_sharpe or 0.0
                }

            # Count indicator usage
            indicator_counts = {}
            strategies = session.query(Strategy).all()
            for strategy in strategies:
                for indicator in strategy.indicators:
                    indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

            # Sort indicators by usage (ascending) to find underused ones
            sorted_indicators = sorted(indicator_counts.items(), key=lambda x: x[1])

            return {
                'by_logic_type': by_logic_type,
                'by_ml_type': by_ml_type,
                'ml_performance': ml_performance,
                'total_strategies': len(strategies),
                'underused_indicators': sorted_indicators[:5],  # Bottom 5
                'popular_indicators': sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:5]  # Top 5
            }
        finally:
            session.close()

    def get_genealogy(self, strategy_id: int, depth: int = 2) -> Dict[str, Any]:
        """
        Get genealogy of a strategy (up to specified depth).

        Args:
            strategy_id: Strategy ID
            depth: How many generations to trace back (1 or 2)

        Returns:
            Dict with parent and grandparent info
        """
        session = self.get_session()
        try:
            strategy = session.query(Strategy).get(strategy_id)
            if not strategy:
                return {}

            genealogy = {
                'current': {
                    'id': strategy.id,
                    'name': strategy.name,
                    'strategy_type': strategy.strategy_type,
                    'ml_strategy_type': strategy.ml_strategy_type,
                    'generation': strategy.generation
                }
            }

            # Get parent
            if strategy.parent_id and depth >= 1:
                parent = session.query(Strategy).get(strategy.parent_id)
                if parent:
                    genealogy['parent'] = {
                        'id': parent.id,
                        'name': parent.name,
                        'strategy_type': parent.strategy_type,
                        'ml_strategy_type': parent.ml_strategy_type,
                        'exploration_vector': strategy.exploration_vector
                    }

                    # Get grandparent
                    if parent.parent_id and depth >= 2:
                        grandparent = session.query(Strategy).get(parent.parent_id)
                        if grandparent:
                            genealogy['grandparent'] = {
                                'id': grandparent.id,
                                'name': grandparent.name,
                                'strategy_type': grandparent.strategy_type,
                                'ml_strategy_type': grandparent.ml_strategy_type
                            }

            return genealogy
        finally:
            session.close()
