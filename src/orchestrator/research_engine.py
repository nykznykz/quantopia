"""Core research orchestrator for the autonomous strategy generation flywheel."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import logging
import time

from src.strategy_generation.generator import StrategyGenerator
from src.code_generation.code_generator import CodeGenerator
from src.backtest.batch_tester import BatchTester
from src.critique.filter import StrategyFilter, FilterCriteria
from src.portfolio.evaluator import PortfolioEvaluator
from src.database.manager import StrategyDatabase

logger = logging.getLogger(__name__)


@dataclass
class IterationReport:
    """Report from a single research iteration."""

    # Required fields
    iteration_number: int
    timestamp: datetime
    num_strategies_generated: int
    num_codes_generated: int
    num_backtests_run: int
    num_approved: int
    num_rejected: int
    num_marginal: int
    num_refinements: int
    num_family_variants: int

    # Optional fields with defaults
    generation_failures: List[str] = field(default_factory=list)
    backtest_failures: List[str] = field(default_factory=list)
    approved_strategies: List[str] = field(default_factory=list)
    rejected_strategies: List[str] = field(default_factory=list)
    refinement_results: Dict[str, Any] = field(default_factory=dict)
    portfolio_metrics: Optional[Dict[str, Any]] = None
    forward_test_priorities: List[Dict[str, Any]] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    database_stored: bool = False

    def __str__(self) -> str:
        """Human-readable summary."""
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         RESEARCH ITERATION #{self.iteration_number}                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}                                              ║
║ Duration: {self.total_duration_seconds:.2f}s                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ GENERATION PHASE                                                             ║
║   Strategies Generated: {self.num_strategies_generated:<3d}                                                  ║
║   Code Generated: {self.num_codes_generated:<3d}                                                        ║
║   Generation Failures: {len(self.generation_failures):<3d}                                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ BACKTESTING PHASE                                                            ║
║   Backtests Run: {self.num_backtests_run:<3d}                                                       ║
║   Backtest Failures: {len(self.backtest_failures):<3d}                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ FILTERING PHASE                                                              ║
║   ✓ Approved: {self.num_approved:<3d}                                                            ║
║   ✗ Rejected: {self.num_rejected:<3d}                                                            ║
║   ~ Marginal: {self.num_marginal:<3d}                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ REFINEMENT & EXPANSION                                                       ║
║   Refinements Attempted: {self.num_refinements:<3d}                                                  ║
║   Family Variants Generated: {self.num_family_variants:<3d}                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ PORTFOLIO EVALUATION                                                         ║
║   Forward Test Queue: {len(self.forward_test_priorities):<3d} strategies prioritized                       ║
║   Database Storage: {'✓ Stored' if self.database_stored else '✗ Not stored'}                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """


class ResearchOrchestrator:
    """
    Core orchestrator for the autonomous strategy research flywheel.

    This class connects all 5 agents in a single research iteration:
    1. Strategy Generator (Agent 1)
    2. Code Generator (Agent 2)
    3. Backtester & Critic (Agent 3)
    4. Refinement Agent (Agent 4)
    5. Portfolio Evaluator (Agent 5)

    Each iteration:
    - Generates new strategies
    - Converts to executable code
    - Backtests all strategies
    - Filters by acceptance criteria
    - Evaluates portfolio fit
    - Refines failures and generates families from successes
    - Stores all results to database
    """

    def __init__(
        self,
        strategy_generator: StrategyGenerator,
        code_generator: CodeGenerator,
        batch_tester: BatchTester,
        strategy_filter: StrategyFilter,
        portfolio_evaluator: PortfolioEvaluator,
        database: Optional[StrategyDatabase] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize research orchestrator.

        Args:
            strategy_generator: Strategy generation agent
            code_generator: Code generation agent
            batch_tester: Backtesting engine
            strategy_filter: Strategy filtering/critique agent
            portfolio_evaluator: Portfolio evaluation agent
            database: Optional database for persistent storage
            config: Configuration dictionary
        """
        self.strategy_generator = strategy_generator
        self.code_generator = code_generator
        self.batch_tester = batch_tester
        self.strategy_filter = strategy_filter
        self.portfolio_evaluator = portfolio_evaluator
        self.database = database

        # Configuration with defaults
        self.config = config or {}
        self.num_strategies_per_batch = self.config.get('num_strategies_per_batch', 10)
        self.max_refinement_attempts = self.config.get('max_refinement_attempts', 2)
        self.strategy_family_size = self.config.get('strategy_family_size', 3)
        self.enable_refinement = self.config.get('enable_refinement', True)
        self.enable_family_generation = self.config.get('enable_family_generation', True)
        self.enable_database_storage = self.config.get('enable_database_storage', True)

        # Iteration counter
        self.iteration_count = 0

        logger.info("Initialized ResearchOrchestrator with all 5 agents")
        logger.info(f"Config: {self.config}")

    def run_iteration(
        self,
        data: pd.DataFrame,
        symbol: str,
        num_strategies: Optional[int] = None,
        strategy_type: Optional[str] = None
    ) -> IterationReport:
        """
        Run one complete research cycle.

        Args:
            data: Historical OHLCV data for backtesting
            symbol: Trading symbol (e.g., 'BTC-USD')
            num_strategies: Number of strategies to generate (overrides config)
            strategy_type: Optional strategy type hint

        Returns:
            IterationReport with complete iteration results
        """
        self.iteration_count += 1
        start_time = time.time()

        logger.info(f"Starting Research Iteration #{self.iteration_count}")
        logger.info(f"Symbol: {symbol}, Data shape: {data.shape}")

        report = IterationReport(
            iteration_number=self.iteration_count,
            timestamp=datetime.now(),
            num_strategies_generated=0,
            num_codes_generated=0,
            num_backtests_run=0,
            num_approved=0,
            num_rejected=0,
            num_marginal=0,
            num_refinements=0,
            num_family_variants=0
        )

        # Override batch size if specified
        batch_size = num_strategies or self.num_strategies_per_batch

        # Phase 1: Generate strategies
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 1: Strategy Generation ({batch_size} strategies)")
        logger.info(f"{'='*80}")

        strategies, generation_failures = self._generate_strategies(
            batch_size=batch_size,
            strategy_type=strategy_type
        )

        report.num_strategies_generated = len(strategies)
        report.generation_failures = generation_failures

        # Phase 2: Generate code
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 2: Code Generation ({len(strategies)} strategies)")
        logger.info(f"{'='*80}")

        strategy_codes, code_failures = self._generate_codes(strategies)

        report.num_codes_generated = len(strategy_codes)
        report.generation_failures.extend(code_failures)

        # Phase 3: Backtest all
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 3: Backtesting ({len(strategy_codes)} strategies)")
        logger.info(f"{'='*80}")

        backtest_results, backtest_failures = self._run_backtests(
            strategies=strategies,
            codes=strategy_codes,
            data=data,
            symbol=symbol
        )

        report.num_backtests_run = len(backtest_results)
        report.backtest_failures = backtest_failures

        # Phase 4: Filter strategies
        logger.info(f"\n{'='*80}")
        logger.info(f"PHASE 4: Strategy Filtering ({len(backtest_results)} results)")
        logger.info(f"{'='*80}")

        filter_results = self._filter_strategies(backtest_results)

        approved = [r for r in filter_results if r.classification == 'approved']
        rejected = [r for r in filter_results if r.classification == 'rejected']
        marginal = [r for r in filter_results if r.classification == 'marginal']

        report.num_approved = len(approved)
        report.num_rejected = len(rejected)
        report.num_marginal = len(marginal)
        report.approved_strategies = [r.strategy_name for r in approved]
        report.rejected_strategies = [r.strategy_name for r in rejected]

        # Phase 5: Portfolio evaluation (if we have approved strategies)
        if approved:
            logger.info(f"\n{'='*80}")
            logger.info(f"PHASE 5: Portfolio Evaluation ({len(approved)} approved)")
            logger.info(f"{'='*80}")

            approved_results = [
                r for r in backtest_results
                if r['strategy_name'] in report.approved_strategies
            ]

            portfolio_eval = self._evaluate_portfolio(approved_results, data)
            report.portfolio_metrics = portfolio_eval.get('portfolio_metrics')
            report.forward_test_priorities = portfolio_eval.get('forward_test_priorities', [])

        # Phase 6: Refinement of failures (if enabled)
        if self.enable_refinement and rejected:
            logger.info(f"\n{'='*80}")
            logger.info(f"PHASE 6: Refinement ({len(rejected)} rejected strategies)")
            logger.info(f"{'='*80}")

            refinement_results = self._refine_failures(
                rejected_strategies=rejected,
                backtest_results=backtest_results,
                data=data,
                symbol=symbol
            )

            report.num_refinements = refinement_results['num_attempts']
            report.refinement_results = refinement_results

        # Phase 7: Family generation from successes (if enabled)
        if self.enable_family_generation and approved:
            logger.info(f"\n{'='*80}")
            logger.info(f"PHASE 7: Family Generation ({len(approved)} approved)")
            logger.info(f"{'='*80}")

            family_results = self._generate_families(
                approved_strategies=approved,
                backtest_results=backtest_results,
                data=data,
                symbol=symbol
            )

            report.num_family_variants = family_results['num_variants']

        # Phase 8: Store to database (if enabled)
        if self.database and self.enable_database_storage:
            logger.info(f"\n{'='*80}")
            logger.info(f"PHASE 8: Database Storage")
            logger.info(f"{'='*80}")

            self._store_to_database(
                strategies=strategies,
                codes=strategy_codes,
                backtest_results=backtest_results,
                portfolio_eval=portfolio_eval if approved else None
            )

            report.database_stored = True

        # Finalize report
        report.total_duration_seconds = time.time() - start_time

        logger.info(str(report))

        return report

    def _generate_strategies(
        self,
        batch_size: int,
        strategy_type: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Generate batch of strategies."""
        strategies = []
        failures = []

        # Get recent strategies from database to avoid duplicates
        avoid_similar = []
        if self.database:
            recent = self.database.get_recent_strategies(limit=50)
            avoid_similar = [s.to_dict() for s in recent]

        for i in range(batch_size):
            try:
                logger.info(f"Generating strategy {i+1}/{batch_size}")

                strategy = self.strategy_generator.generate_strategy(
                    strategy_type=strategy_type,
                    avoid_similar_to=avoid_similar
                )

                strategies.append(strategy)
                avoid_similar.append(strategy)  # Avoid within this batch too

                logger.info(f"✓ Generated: {strategy['strategy_name']}")

            except Exception as e:
                error_msg = f"Strategy {i+1} generation failed: {str(e)}"
                logger.error(error_msg)
                failures.append(error_msg)

        logger.info(f"Generated {len(strategies)}/{batch_size} strategies")

        return strategies, failures

    def _generate_codes(
        self,
        strategies: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Generate executable code for strategies."""
        codes = []
        failures = []

        for strategy in strategies:
            try:
                logger.info(f"Generating code for: {strategy['strategy_name']}")

                code = self.code_generator.generate_strategy_class(strategy)
                codes.append(code)

                logger.info(f"✓ Code generated: {len(code)} chars")

            except Exception as e:
                error_msg = f"Code generation failed for {strategy['strategy_name']}: {str(e)}"
                logger.error(error_msg)
                failures.append(error_msg)

        logger.info(f"Generated {len(codes)}/{len(strategies)} code files")

        return codes, failures

    def _run_backtests(
        self,
        strategies: List[Dict[str, Any]],
        codes: List[str],
        data: pd.DataFrame,
        symbol: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Run backtests for all strategies."""
        if not codes:
            return [], []

        strategy_names = [s['strategy_name'] for s in strategies[:len(codes)]]

        try:
            results = self.batch_tester.run_batch(
                strategy_codes=codes,
                strategy_names=strategy_names,
                data=data,
                symbol=symbol,
                parallel=False,  # Sequential for stability
                store_in_db=False  # We'll store later with full context
            )

            logger.info(f"✓ Completed {len(results)} backtests")

            return results, []

        except Exception as e:
            error_msg = f"Batch backtest failed: {str(e)}"
            logger.error(error_msg)
            return [], [error_msg]

    def _filter_strategies(
        self,
        backtest_results: List[Dict[str, Any]]
    ) -> List[Any]:
        """Filter strategies by acceptance criteria."""
        filter_results = []

        for result in backtest_results:
            try:
                filter_result = self.strategy_filter.filter_strategy(result)
                filter_results.append(filter_result)

                status = "✓" if filter_result.passed else "✗"
                logger.info(f"{status} {filter_result.strategy_name}: {filter_result.classification}")

            except Exception as e:
                logger.error(f"Filter error for {result.get('strategy_name')}: {e}")

        return filter_results

    def _evaluate_portfolio(
        self,
        approved_results: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate portfolio-level metrics and forward test priorities."""
        try:
            evaluation = self.portfolio_evaluator.evaluate_portfolio(
                backtest_results=approved_results,
                historical_data=data
            )

            logger.info("✓ Portfolio evaluation completed")

            if 'portfolio_metrics' in evaluation:
                metrics = evaluation['portfolio_metrics']
                logger.info(f"  Portfolio Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  Portfolio Return: {metrics.get('total_return', 0)*100:.2f}%")
                logger.info(f"  Diversification: {metrics.get('diversification_score', 0):.2f}")

            return evaluation

        except Exception as e:
            logger.error(f"Portfolio evaluation failed: {e}")
            return {}

    def _refine_failures(
        self,
        rejected_strategies: List[Any],
        backtest_results: List[Dict[str, Any]],
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """Attempt to refine failed strategies."""
        num_attempts = 0
        num_successes = 0
        refined_strategies = []

        # Limit refinements to max attempts
        strategies_to_refine = rejected_strategies[:self.max_refinement_attempts]

        for filter_result in strategies_to_refine:
            try:
                num_attempts += 1
                strategy_name = filter_result.strategy_name

                logger.info(f"Refining: {strategy_name}")
                logger.info(f"  Reasons: {', '.join(filter_result.rejection_reasons)}")

                # Get original strategy metadata
                original_result = next(
                    (r for r in backtest_results if r['strategy_name'] == strategy_name),
                    None
                )

                if not original_result:
                    logger.warning(f"Could not find original result for {strategy_name}")
                    continue

                # Generate refined version
                refined_strategy = self.strategy_generator.refine_strategy(
                    original_strategy=original_result.get('strategy_metadata', {}),
                    failure_reasons=filter_result.rejection_reasons,
                    backtest_metrics=filter_result.metrics_summary
                )

                # Generate code and test
                refined_code = self.code_generator.generate_strategy_class(refined_strategy)

                refined_results = self.batch_tester.run_batch(
                    strategy_codes=[refined_code],
                    strategy_names=[refined_strategy['strategy_name']],
                    data=data,
                    symbol=symbol,
                    store_in_db=False
                )

                if refined_results:
                    refined_filter = self.strategy_filter.filter_strategy(refined_results[0])

                    if refined_filter.passed:
                        num_successes += 1
                        refined_strategies.append(refined_results[0])
                        logger.info(f"✓ Refinement succeeded: {refined_strategy['strategy_name']}")
                    else:
                        logger.info(f"✗ Refinement still rejected: {refined_strategy['strategy_name']}")

            except Exception as e:
                logger.error(f"Refinement failed: {e}")

        logger.info(f"Refinement: {num_successes}/{num_attempts} successes")

        return {
            'num_attempts': num_attempts,
            'num_successes': num_successes,
            'refined_strategies': refined_strategies
        }

    def _generate_families(
        self,
        approved_strategies: List[Any],
        backtest_results: List[Dict[str, Any]],
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """Generate strategy families from successful strategies."""
        num_variants = 0
        successful_variants = []

        # Pick top performers for family generation (limit to 2-3 parents)
        parents = approved_strategies[:min(2, len(approved_strategies))]

        for filter_result in parents:
            try:
                strategy_name = filter_result.strategy_name
                logger.info(f"Generating family from: {strategy_name}")

                # Get original strategy metadata
                original_result = next(
                    (r for r in backtest_results if r['strategy_name'] == strategy_name),
                    None
                )

                if not original_result:
                    continue

                # Generate family variants
                family = self.strategy_generator.generate_strategy_family(
                    parent_strategy=original_result.get('strategy_metadata', {}),
                    num_variants=self.strategy_family_size
                )

                # Test each variant
                for variant in family:
                    try:
                        variant_code = self.code_generator.generate_strategy_class(variant)

                        variant_results = self.batch_tester.run_batch(
                            strategy_codes=[variant_code],
                            strategy_names=[variant['strategy_name']],
                            data=data,
                            symbol=symbol,
                            store_in_db=False
                        )

                        if variant_results:
                            num_variants += 1
                            variant_filter = self.strategy_filter.filter_strategy(variant_results[0])

                            if variant_filter.passed:
                                successful_variants.append(variant_results[0])
                                logger.info(f"✓ Family variant approved: {variant['strategy_name']}")
                            else:
                                logger.info(f"~ Family variant rejected: {variant['strategy_name']}")

                    except Exception as e:
                        logger.error(f"Family variant failed: {e}")

            except Exception as e:
                logger.error(f"Family generation failed: {e}")

        logger.info(f"Generated {num_variants} family variants ({len(successful_variants)} approved)")

        return {
            'num_variants': num_variants,
            'num_approved': len(successful_variants),
            'successful_variants': successful_variants
        }

    def _store_to_database(
        self,
        strategies: List[Dict[str, Any]],
        codes: List[str],
        backtest_results: List[Dict[str, Any]],
        portfolio_eval: Optional[Dict[str, Any]]
    ):
        """Store all iteration results to database."""
        if not self.database:
            return

        try:
            # Store strategies and results
            for i, strategy in enumerate(strategies):
                if i < len(codes) and i < len(backtest_results):
                    result = backtest_results[i]

                    # Store strategy
                    strategy_id = self.database.store_strategy(
                        strategy_metadata=strategy,
                        code=codes[i]
                    )

                    # Store backtest result
                    self.database.store_backtest_results(
                        strategy_id=strategy_id,
                        results=result
                    )

            # Store portfolio evaluation
            if portfolio_eval:
                self.database.store_portfolio_evaluation(portfolio_eval)

            logger.info(f"✓ Stored {len(strategies)} strategies to database")

        except Exception as e:
            logger.error(f"Database storage failed: {e}")
