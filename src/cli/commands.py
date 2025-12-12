"""
CLI command implementations for Quantopia.
"""

import os
import sys
import json
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.agents.strategy_agent import StrategyAgent
from src.agents.ml_quant_agent import MLQuantAgent
from src.orchestrator.agent_router import AgentRouter
from src.orchestrator.research_engine import ResearchOrchestrator
from src.database.manager import StrategyDatabase
from src.ml.model_registry import ModelRegistry
from src.code_generation.code_generator import CodeGenerator
from src.strategy_generation.llm_client import LLMClient
from src.strategy_generation.generator import StrategyGenerator
from src.backtest.batch_tester import BatchTester
from src.critique.filter import StrategyFilter, FilterCriteria
from src.portfolio.evaluator import PortfolioEvaluator
from src.indicators import INDICATOR_REGISTRY

# Add path for simulated_exchange imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "simulated_exchange" / "src"))
from simulated_exchange import download_data

logger = logging.getLogger(__name__)


def run_research_session(
    config: Dict[str, Any],
    num_strategies: int,
    symbol: str,
    days: int,
    exploration_rate: float,
    enable_ml: bool
):
    """Run a COMPLETE autonomous research session with full flywheel."""
    logger.info("="*80)
    logger.info("STARTING COMPLETE RESEARCH SESSION (FULL FLYWHEEL)")
    logger.info("="*80)
    logger.info(f"Strategies: {num_strategies}")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Days: {days}")
    logger.info(f"Exploration rate: {exploration_rate}")
    logger.info(f"ML enabled: {enable_ml}")
    logger.info("")
    logger.info("Phases: Generation → Code → Backtest → Critique → Storage")
    logger.info("="*80)
    logger.info("")

    # Initialize infrastructure
    logger.info("[1/4] Initializing infrastructure...")

    db_path = config.get('database', {}).get('path', 'data/strategies.db')
    registry_dir = config.get('ml', {}).get('registry_dir', 'models/registry')

    database = StrategyDatabase(db_path=db_path)
    model_registry = ModelRegistry(registry_dir=registry_dir)

    # Get LLM configuration from config or environment
    llm_config = config.get('llm', {})
    provider = os.getenv('LLM_PROVIDER', llm_config.get('provider', 'openai'))

    # Initialize LLM client based on provider
    logger.info(f"Initializing LLM client with provider: {provider}")

    if provider == 'azure':
        if not os.getenv('AZURE_OPENAI_API_KEY'):
            logger.error("AZURE_OPENAI_API_KEY not found in environment")
            raise ValueError("AZURE_OPENAI_API_KEY not found. Please check your .env file.")

        logger.info("✓ Azure OpenAI credentials loaded")
        llm_client = LLMClient(
            provider='azure',
            model=os.getenv('AZURE_OPENAI_MODEL', llm_config.get('model', 'gpt-4')),
            temperature=llm_config.get('temperature', 0.7),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )
    else:
        if not os.getenv('OPENAI_API_KEY'):
            logger.error("OPENAI_API_KEY not found in environment")
            raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

        logger.info("✓ OpenAI API key loaded")
        llm_client = LLMClient(
            provider='openai',
            model=llm_config.get('model', 'gpt-4'),
            temperature=llm_config.get('temperature', 0.7)
        )

    logger.info("✓ Infrastructure ready")

    # Initialize all 5 agents
    logger.info("\n[2/4] Initializing agents...")

    # Strategy Generator
    available_indicators = list(INDICATOR_REGISTRY.keys())
    strategy_generator = StrategyGenerator(
        llm_client=llm_client,
        available_indicators=available_indicators
    )

    # Code Generator
    code_generator = CodeGenerator(
        llm_client=llm_client,
        temperature=0.2
    )

    # Batch Tester
    batch_tester = BatchTester(db=database)

    # Strategy Filter
    filter_criteria = FilterCriteria(
        min_sharpe_ratio=config.get('filtering', {}).get('min_sharpe_ratio', 0.5),
        min_total_return=config.get('filtering', {}).get('min_total_return', 0.05),
        max_drawdown=config.get('filtering', {}).get('max_drawdown', 0.30),
        min_num_trades=config.get('filtering', {}).get('min_num_trades', 10)
    )
    strategy_filter = StrategyFilter(criteria=filter_criteria)

    # Portfolio Evaluator
    portfolio_evaluator = PortfolioEvaluator()

    logger.info("✓ All 5 agents initialized")

    # Initialize Research Orchestrator (the FULL flywheel!)
    logger.info("\n[3/4] Initializing Research Orchestrator...")

    orchestrator_config = {
        'num_strategies_per_batch': num_strategies,
        'max_refinement_attempts': config.get('agents', {}).get('max_refinement_attempts', 2),
        'strategy_family_size': config.get('agents', {}).get('strategy_family_size', 3),
        'enable_refinement': config.get('research', {}).get('enable_refinement', True),
        'enable_family_generation': config.get('research', {}).get('enable_family_generation', True),
        'enable_database_storage': True
    }

    orchestrator = ResearchOrchestrator(
        strategy_generator=strategy_generator,
        code_generator=code_generator,
        batch_tester=batch_tester,
        strategy_filter=strategy_filter,
        portfolio_evaluator=portfolio_evaluator,
        database=database,
        config=orchestrator_config
    )

    logger.info("✓ Research Orchestrator ready")

    # Load real market data from Binance
    logger.info(f"\n[4/4] Loading {days} days of real market data from Binance...")
    training_data = _load_or_download_data(
        symbol=symbol,
        days=days,
        cache_dir='data/historical'
    )
    logger.info(f"✓ Data ready: {len(training_data)} candles from {training_data['timestamp'].min()} to {training_data['timestamp'].max()}")

    # Run COMPLETE iteration (all 8 phases!)
    logger.info("\n" + "="*80)
    logger.info("STARTING COMPLETE RESEARCH ITERATION")
    logger.info("="*80)
    logger.info("")

    try:
        report = orchestrator.run_iteration(
            data=training_data,
            symbol=symbol,
            num_strategies=num_strategies
        )

        # Print complete report
        logger.info("\n" + str(report))

        # Additional summary
        logger.info("\n" + "="*80)
        logger.info("FLYWHEEL COMPLETE!")
        logger.info("="*80)
        logger.info(f"✓ Generated: {report.num_strategies_generated} strategies")
        logger.info(f"✓ Backtested: {report.num_backtests_run} strategies")
        logger.info(f"✓ Approved: {report.num_approved} strategies")
        logger.info(f"✓ Rejected: {report.num_rejected} strategies")
        logger.info(f"✓ Refined: {report.num_refinements} attempts")
        logger.info(f"✓ Family variants: {report.num_family_variants} generated")
        logger.info(f"✓ Stored: {'Yes' if report.database_stored else 'No'}")
        logger.info(f"✓ Duration: {report.total_duration_seconds:.1f}s")
        logger.info("")

        if report.approved_strategies:
            logger.info("Top approved strategies:")
            for strat_name in report.approved_strategies[:5]:
                logger.info(f"  ✓ {strat_name}")

        logger.info("="*80)

    except Exception as e:
        logger.error(f"Research iteration failed: {e}")
        raise


def start_daemon(
    config: Dict[str, Any],
    daemon: bool,
    continuous: bool,
    interval: int
):
    """Start Quantopia as a daemon."""
    logger.info("Starting Quantopia daemon...")

    if daemon:
        _daemonize()

    # Write PID file
    pid_file = Path(config.get('daemon', {}).get('pid_file', 'data/quantopia.pid'))
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    logger.info(f"Daemon started with PID: {os.getpid()}")

    if continuous:
        logger.info(f"Running continuous sessions every {interval}s...")
        # TODO: Implement continuous loop
        logger.warning("Continuous mode not yet implemented")
    else:
        logger.info("Daemon running (no continuous mode)")


def stop_daemon(config: Dict[str, Any]):
    """Stop Quantopia daemon."""
    pid_file = Path(config.get('daemon', {}).get('pid_file', 'data/quantopia.pid'))

    if not pid_file.exists():
        logger.error("Daemon is not running (PID file not found)")
        return

    pid = int(pid_file.read_text())

    try:
        os.kill(pid, signal.SIGTERM)
        logger.info(f"Stopped daemon (PID: {pid})")
        pid_file.unlink()
    except ProcessLookupError:
        logger.error(f"Process {pid} not found")
        pid_file.unlink()


def status_command(config: Dict[str, Any], detailed: bool):
    """Show system status."""
    print("\n" + "="*80)
    print("QUANTOPIA SYSTEM STATUS")
    print("="*80)

    # Daemon status
    pid_file = Path(config.get('daemon', {}).get('pid_file', 'data/quantopia.pid'))
    if pid_file.exists():
        pid = int(pid_file.read_text())
        print(f"Daemon: RUNNING (PID: {pid})")
    else:
        print("Daemon: NOT RUNNING")

    # Database status
    db_path = config.get('database', {}).get('path', 'data/strategies.db')
    database = StrategyDatabase(db_path=db_path)

    stats = database.get_statistics()

    print(f"\nDatabase: {db_path}")
    print(f"Total strategies: {stats['total_strategies']}")
    print(f"By status: {stats['strategies_by_status']}")

    if detailed:
        print(f"\nDetailed Statistics:")
        underexplored = database.get_underexplored_areas()
        print(f"Underexplored ML types: {underexplored['by_ml_type']}")
        print(f"Underexplored logic types: {underexplored['by_logic_type']}")

    print("="*80 + "\n")


def list_strategies(
    config: Dict[str, Any],
    top: Optional[int],
    recent: Optional[int],
    filter_status: str,
    ml_type: Optional[str]
):
    """List strategies."""
    db_path = config.get('database', {}).get('path', 'data/strategies.db')
    database = StrategyDatabase(db_path=db_path)

    if top:
        strategies = database.get_top_strategies(metric='sharpe_ratio', limit=top)
        title = f"TOP {top} STRATEGIES (by Sharpe Ratio)"
    elif recent:
        strategies = database.get_recent_strategies(limit=recent)
        title = f"{recent} MOST RECENT STRATEGIES"
    else:
        # TODO: Implement general listing
        strategies = database.get_recent_strategies(limit=20)
        title = "RECENT STRATEGIES"

    print("\n" + "="*80)
    print(title)
    print("="*80)

    for i, strat in enumerate(strategies, 1):
        strat_dict = strat.to_dict()
        print(f"\n{i}. {strat_dict['name']}")
        print(f"   ID: {strat_dict['id']}")
        print(f"   Type: {strat_dict['strategy_type']}")
        print(f"   Created: {strat_dict['created_at']}")

    print("="*80 + "\n")


def show_strategy(
    config: Dict[str, Any],
    strategy_id: str,
    show_code: bool,
    show_metrics: bool
):
    """Show strategy details."""
    db_path = config.get('database', {}).get('path', 'data/strategies.db')
    database = StrategyDatabase(db_path=db_path)

    # TODO: Implement get_strategy_by_id
    print(f"Strategy {strategy_id} details:")
    print("(Not yet implemented)")


def export_strategy(
    config: Dict[str, Any],
    strategy_id: str,
    output_path: Optional[str]
):
    """Export strategy code."""
    db_path = config.get('database', {}).get('path', 'data/strategies.db')
    database = StrategyDatabase(db_path=db_path)

    # TODO: Implement export
    print(f"Exporting strategy {strategy_id}...")
    print("(Not yet implemented)")


def query_database(
    config: Dict[str, Any],
    underexplored: bool,
    statistics: bool,
    models: bool
):
    """Query database."""
    db_path = config.get('database', {}).get('path', 'data/strategies.db')
    database = StrategyDatabase(db_path=db_path)

    if underexplored:
        areas = database.get_underexplored_areas()
        print("\nUnderexplored Areas:")
        print(json.dumps(areas, indent=2))

    if statistics:
        stats = database.get_statistics()
        print("\nDatabase Statistics:")
        print(json.dumps(stats, indent=2))

    if models:
        registry_dir = config.get('ml', {}).get('registry_dir', 'models/registry')
        model_registry = ModelRegistry(registry_dir=registry_dir)
        print("\nML Models Registry:")
        print("(Not yet implemented)")


# Helper functions

def _load_or_download_data(
    symbol: str,
    days: int,
    cache_dir: str = 'data/historical',
    timeframe: str = '1h'
) -> pd.DataFrame:
    """Load or download real market data from Binance.

    Args:
        symbol: Trading symbol (e.g., 'BTC-USD')
        days: Number of days of historical data
        cache_dir: Directory to cache downloaded data
        timeframe: Candle timeframe (default: '1h')

    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
    """
    from datetime import timedelta

    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Cache filename
    cache_file = cache_path / f"binance_{symbol.replace('-', '_')}_{timeframe}_{days}d.csv"

    # Try to load from cache
    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Check if cache is recent enough (within last 24 hours)
        latest_timestamp = df['timestamp'].max()
        if datetime.now() - latest_timestamp < timedelta(hours=24):
            logger.info(f"Cache is fresh (latest: {latest_timestamp})")
            return df
        else:
            logger.info(f"Cache is stale (latest: {latest_timestamp}), re-downloading...")

    # Download from Binance
    logger.info(f"Downloading {days} days of {symbol} data from Binance...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        df = download_data(
            symbols=[symbol],
            exchange='binance',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            output_file=None  # Don't save yet
        )

        # Save to cache
        df.to_csv(cache_file, index=False)
        logger.info(f"Saved data to cache: {cache_file}")

        return df

    except Exception as e:
        logger.error(f"Failed to download data from Binance: {e}")

        # Fallback to cached data if available (even if stale)
        if cache_file.exists():
            logger.warning("Falling back to stale cached data")
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df

        # Last resort: raise error
        raise RuntimeError(
            f"Failed to download data and no cache available. "
            f"Error: {e}"
        )


def _print_session_summary(results: list):
    """Print session summary."""
    logger.info("\n" + "="*80)
    logger.info("SESSION SUMMARY")
    logger.info("="*80)

    successful = [r for r in results if r['strategy_id'] is not None]

    logger.info(f"Total strategies: {len(results)}")
    logger.info(f"Successful: {len(successful)}")

    # By type
    types = {}
    for r in results:
        t = r['strategy_type']
        types[t] = types.get(t, 0) + 1

    logger.info(f"\nBy type:")
    for t, count in types.items():
        logger.info(f"  {t}: {count}")

    # Top performers
    sorted_results = sorted(results, key=lambda x: x.get('mock_sharpe', 0), reverse=True)

    logger.info(f"\nTop 3 performers:")
    for i, r in enumerate(sorted_results[:3], 1):
        logger.info(f"  {i}. {r['strategy_name']} - Sharpe: {r.get('mock_sharpe', 0):.2f}")

    logger.info("="*80)


def _daemonize():
    """Daemonize the current process."""
    if os.fork():
        sys.exit(0)

    os.chdir('/')
    os.setsid()
    os.umask(0)

    if os.fork():
        sys.exit(0)

    sys.stdout.flush()
    sys.stderr.flush()

    with open('/dev/null', 'r') as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open('/dev/null', 'a+') as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
    with open('/dev/null', 'a+') as f:
        os.dup2(f.fileno(), sys.stderr.fileno())
