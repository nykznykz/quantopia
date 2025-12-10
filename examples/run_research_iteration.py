#!/usr/bin/env python3
"""
CLI interface for running research iterations.

This script runs a single iteration of the autonomous strategy research flywheel,
connecting all 5 agents to generate, test, evaluate, and refine trading strategies.

Usage:
    python examples/run_research_iteration.py \\
        --symbol BTC-USD \\
        --num-strategies 10 \\
        --exchange binance \\
        --timeframe 1h

    # With custom config:
    python examples/run_research_iteration.py \\
        --symbol ETH-USD \\
        --config config/orchestrator_config.yaml \\
        --num-strategies 5

    # With specific strategy type:
    python examples/run_research_iteration.py \\
        --symbol BTC-USD \\
        --strategy-type momentum \\
        --num-strategies 8
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import yaml
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.orchestrator.research_engine import ResearchOrchestrator
from src.strategy_generation.generator import StrategyGenerator
from src.strategy_generation.llm_client import LLMClient
from src.code_generation.code_generator import CodeGenerator
from src.backtest.batch_tester import BatchTester
from src.critique.filter import StrategyFilter, FilterCriteria
from src.portfolio.evaluator import PortfolioEvaluator
from src.database.manager import StrategyDatabase
from src.indicators import INDICATOR_REGISTRY


def setup_logging(config: dict):
    """Setup logging based on config."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create logs directory if needed
    if log_config.get('log_to_file', False):
        log_file = log_config.get('log_file_path', 'logs/orchestrator.log')
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=log_level, format=log_format)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_market_data(symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
    """
    Load historical market data.

    For now, this loads from the data directory. In production, this would
    connect to exchange APIs or a data warehouse.
    """
    # Try to load from data directory
    data_path = project_root / "data" / f"{exchange}_{symbol.replace('-', '_')}_{timeframe}.csv"

    if not data_path.exists():
        # Try alternative naming
        data_path = project_root / "data" / f"{symbol.replace('-', '_')}_{timeframe}.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Market data not found at {data_path}. "
            f"Please download data first or specify correct path."
        )

    logging.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Ensure proper column names
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {required_cols}")

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    logging.info(f"Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def initialize_components(config: dict):
    """Initialize all components based on configuration."""
    logging.info("Initializing components...")

    # Database
    db_config = config.get('database', {})
    database = None
    if db_config.get('enable_storage', True):
        database = StrategyDatabase(
            db_path=db_config.get('path', 'data/strategies.db'),
            echo=db_config.get('echo_sql', False)
        )
        logging.info(f"✓ Database initialized: {db_config.get('path')}")

    # LLM Client for strategy generation
    strategy_config = config.get('strategy_generation', {})
    llm_client = LLMClient(
        provider=strategy_config.get('llm_provider', 'openai'),
        model=strategy_config.get('llm_model', 'gpt-4'),
        temperature=strategy_config.get('generation_temperature', 0.8)
    )
    logging.info(f"✓ LLM client initialized: {llm_client.provider}/{llm_client.model}")

    # Strategy Generator
    indicator_config = config.get('indicators', {})
    available_indicators = indicator_config.get('available_indicators', list(INDICATOR_REGISTRY.keys()))

    strategy_generator = StrategyGenerator(
        llm_client=llm_client,
        available_indicators=available_indicators,
        max_retries=strategy_config.get('max_generation_retries', 3),
        db=database
    )
    logging.info(f"✓ Strategy generator initialized ({len(available_indicators)} indicators)")

    # Code Generator
    code_generator = CodeGenerator(
        llm_client=llm_client,
        temperature=strategy_config.get('code_generation_temperature', 0.2),
        max_retries=strategy_config.get('max_generation_retries', 3)
    )
    logging.info("✓ Code generator initialized")

    # Batch Tester
    backtest_config = config.get('backtesting', {})
    batch_tester = BatchTester(
        initial_capital=backtest_config.get('initial_capital', 10000.0),
        slippage_bps=backtest_config.get('slippage_bps', 5.0),
        maker_fee=backtest_config.get('maker_fee', 0.0),
        taker_fee=backtest_config.get('taker_fee', 0.00025),
        db=database
    )
    logging.info("✓ Batch tester initialized")

    # Strategy Filter
    filter_config = config.get('filtering', {})
    filter_criteria = FilterCriteria(
        min_total_return=filter_config.get('min_total_return', 0.05),
        min_sharpe_ratio=filter_config.get('min_sharpe_ratio', 0.5),
        max_drawdown=filter_config.get('max_drawdown', 0.30),
        max_consecutive_losses=filter_config.get('max_consecutive_losses', 10),
        min_num_trades=filter_config.get('min_num_trades', 10),
        max_num_trades=filter_config.get('max_num_trades'),
        min_win_rate=filter_config.get('min_win_rate', 0.30),
        min_profit_factor=filter_config.get('min_profit_factor', 1.0),
        min_sample_size=filter_config.get('min_sample_size', 20)
    )
    strategy_filter = StrategyFilter(criteria=filter_criteria)
    logging.info("✓ Strategy filter initialized")

    # Portfolio Evaluator
    portfolio_evaluator = PortfolioEvaluator()
    logging.info("✓ Portfolio evaluator initialized")

    # Research Orchestrator
    orchestrator_config = config.get('research_iteration', {})
    orchestrator = ResearchOrchestrator(
        strategy_generator=strategy_generator,
        code_generator=code_generator,
        batch_tester=batch_tester,
        strategy_filter=strategy_filter,
        portfolio_evaluator=portfolio_evaluator,
        database=database,
        config=orchestrator_config
    )
    logging.info("✓ Research orchestrator initialized")

    return orchestrator


def save_iteration_report(report, config: dict):
    """Save iteration report to disk."""
    perf_config = config.get('performance_tracking', {})

    if not perf_config.get('save_reports', True):
        return

    reports_dir = Path(perf_config.get('reports_dir', 'reports/iterations'))
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f"iteration_{report.iteration_number}_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write(str(report))

    logging.info(f"✓ Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run a single research iteration of the autonomous strategy research flywheel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading symbol (e.g., BTC-USD, ETH-USD)'
    )

    # Optional arguments
    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange name (default: binance)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d) (default: 1h)'
    )

    parser.add_argument(
        '--num-strategies',
        type=int,
        help='Number of strategies to generate (overrides config)'
    )

    parser.add_argument(
        '--strategy-type',
        type=str,
        choices=['momentum', 'trend', 'mean_reversion', 'volatility', 'breakout'],
        help='Strategy type hint (optional)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/orchestrator_config.yaml',
        help='Path to configuration file (default: config/orchestrator_config.yaml)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to CSV data file (overrides default data loading)'
    )

    args = parser.parse_args()

    # Load configuration
    config_path = project_root / args.config
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Setup logging
    setup_logging(config)

    logging.info("="*80)
    logging.info("AUTONOMOUS RESEARCH ITERATION STARTING")
    logging.info("="*80)
    logging.info(f"Symbol: {args.symbol}")
    logging.info(f"Exchange: {args.exchange}")
    logging.info(f"Timeframe: {args.timeframe}")
    logging.info(f"Config: {args.config}")

    try:
        # Load market data
        if args.data_path:
            logging.info(f"Loading data from: {args.data_path}")
            data = pd.read_csv(args.data_path)
        else:
            data = load_market_data(args.symbol, args.exchange, args.timeframe)

        # Initialize components
        orchestrator = initialize_components(config)

        # Run iteration
        logging.info("\n" + "="*80)
        logging.info("STARTING RESEARCH ITERATION")
        logging.info("="*80 + "\n")

        report = orchestrator.run_iteration(
            data=data,
            symbol=args.symbol,
            num_strategies=args.num_strategies,
            strategy_type=args.strategy_type
        )

        # Print report
        print(report)

        # Save report
        save_iteration_report(report, config)

        # Print summary
        print("\n" + "="*80)
        print("ITERATION COMPLETE")
        print("="*80)
        print(f"Strategies Generated: {report.num_strategies_generated}")
        print(f"Backtests Run: {report.num_backtests_run}")
        print(f"Approved: {report.num_approved}")
        print(f"Rejected: {report.num_rejected}")
        print(f"Forward Test Queue: {len(report.forward_test_priorities)}")
        print(f"Duration: {report.total_duration_seconds:.2f}s")
        print("="*80)

        if report.approved_strategies:
            print("\n✓ APPROVED STRATEGIES:")
            for name in report.approved_strategies:
                print(f"  - {name}")

        if report.forward_test_priorities:
            print(f"\n⭐ TOP FORWARD TEST PRIORITIES:")
            for i, priority in enumerate(report.forward_test_priorities[:5], 1):
                print(f"  {i}. {priority.get('strategy_name', 'Unknown')} "
                      f"(score: {priority.get('priority_score', 0):.2f})")

        logging.info("\nResearch iteration completed successfully!")

    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during research iteration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
