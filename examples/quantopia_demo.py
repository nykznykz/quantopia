#!/usr/bin/env python3
"""
Quantopia Full System Demonstration

This script demonstrates the complete autonomous quant research flywheel:
- Multiple research iterations
- Portfolio evolution tracking
- Strategy genealogy visualization
- Database pattern analysis
- Performance analytics

Usage:
    python examples/quantopia_demo.py --iterations 3 --strategies-per-batch 5
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.orchestrator.research_engine import ResearchOrchestrator
from src.strategy_generation.generator import StrategyGenerator
from src.code_generation.code_generator import CodeGenerator
from src.backtest.batch_tester import BatchTester
from src.critique.filter import StrategyFilter, FilterCriteria
from src.portfolio.evaluator import PortfolioEvaluator
from src.database.manager import StrategyDatabase


def print_banner(text: str):
    """Print a formatted banner."""
    width = 80
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def create_sample_data(days: int = 30, freq: str = '1H') -> pd.DataFrame:
    """
    Create sample OHLCV data for demonstration.

    Args:
        days: Number of days of data
        freq: Frequency ('1H', '4H', '1D')

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Creating {days} days of sample data at {freq} frequency...")

    # Calculate number of periods
    periods_per_day = {'1H': 24, '4H': 6, '1D': 1}
    periods = days * periods_per_day.get(freq, 24)

    # Generate realistic price data
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=periods, freq=freq)
    np.random.seed(42)

    # Simulate price movement with trends and mean reversion
    base_price = 50000
    trend = np.linspace(0, 0.2, periods)  # 20% upward trend over period
    noise = np.random.randn(periods) * 0.015  # 1.5% volatility
    mean_reversion = -0.3 * (np.cumsum(noise))  # Mean reversion component

    price_multiplier = 1 + trend + noise + mean_reversion * 0.1
    close_prices = base_price * price_multiplier

    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices * (1 + np.random.randn(periods) * 0.002),
        'high': close_prices * (1 + np.random.uniform(0, 0.01, periods)),
        'low': close_prices * (1 - np.random.uniform(0, 0.01, periods)),
        'close': close_prices,
        'volume': np.random.uniform(100, 1000, periods)
    })

    # Ensure OHLC validity
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    print(f"✓ Created {len(data)} candles from {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
    print(f"  Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"  Total return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100:.2f}%\n")

    return data


def initialize_orchestrator(
    database_path: str,
    api_key: str,
    model: str = "gpt-4",
    config: Dict[str, Any] = None
) -> ResearchOrchestrator:
    """
    Initialize the research orchestrator with all components.

    Args:
        database_path: Path to SQLite database
        api_key: OpenAI/Anthropic API key
        model: LLM model to use
        config: Configuration dictionary

    Returns:
        Configured ResearchOrchestrator
    """
    print_banner("Initializing Research Orchestrator")

    # Initialize database
    print(f"Database: {database_path}")
    database = StrategyDatabase(database_path)

    # Initialize Agent 1: Strategy Generator
    print("Agent 1: Strategy Generator (GPT-4)")
    from src.strategy_generation.llm_client import LLMClient
    llm_client = LLMClient(provider="openai", api_key=api_key, model=model, temperature=0.7)

    # Get available indicators from the indicators module
    from src.indicators.indicator_library import AVAILABLE_INDICATORS
    available_indicators = list(AVAILABLE_INDICATORS.keys())

    strategy_generator = StrategyGenerator(
        llm_client=llm_client,
        available_indicators=available_indicators
    )

    # Initialize Agent 2: Code Generator
    print("Agent 2: Code Generator (GPT-4)")
    code_llm_client = LLMClient(provider="openai", api_key=api_key, model=model, temperature=0.0)
    code_generator = CodeGenerator(
        llm_client=code_llm_client
    )

    # Initialize Agent 3: Backtester
    print("Agent 3: Backtester & Critic")
    batch_tester = BatchTester(
        initial_capital=10000,
        slippage_bps=5.0,
        db=database
    )

    # Initialize filtering criteria
    filter_criteria = FilterCriteria(
        min_sharpe_ratio=0.5,
        min_return=0.05,
        max_drawdown=0.35,
        min_trades=5,
        min_win_rate=0.35,
        min_profit_factor=1.0
    )

    strategy_filter = StrategyFilter(criteria=filter_criteria)

    # Initialize Agent 5: Portfolio Evaluator
    print("Agent 5: Portfolio Evaluator")
    portfolio_evaluator = PortfolioEvaluator()

    # Default config
    default_config = {
        'num_strategies_per_batch': 10,
        'max_refinement_attempts': 2,
        'strategy_family_size': 3,
        'enable_refinement': True,
        'enable_family_generation': True,
        'enable_database_storage': True
    }

    if config:
        default_config.update(config)

    # Create orchestrator
    orchestrator = ResearchOrchestrator(
        strategy_generator=strategy_generator,
        code_generator=code_generator,
        batch_tester=batch_tester,
        strategy_filter=strategy_filter,
        portfolio_evaluator=portfolio_evaluator,
        database=database,
        config=default_config
    )

    print("\n✓ All agents initialized and connected\n")

    return orchestrator


def run_demo_iterations(
    orchestrator: ResearchOrchestrator,
    data: pd.DataFrame,
    symbol: str,
    num_iterations: int = 3
) -> List[Any]:
    """
    Run multiple research iterations and track evolution.

    Args:
        orchestrator: Initialized orchestrator
        data: Historical OHLCV data
        symbol: Trading symbol
        num_iterations: Number of iterations to run

    Returns:
        List of iteration reports
    """
    print_banner(f"Running {num_iterations} Research Iterations")

    reports = []

    for i in range(num_iterations):
        print(f"\n{'─' * 80}")
        print(f"ITERATION {i + 1}/{num_iterations}")
        print(f"{'─' * 80}\n")

        try:
            report = orchestrator.run_iteration(
                data=data,
                symbol=symbol
            )

            reports.append(report)

            # Print concise summary
            print(f"\n✓ Iteration {i + 1} completed in {report.total_duration_seconds:.1f}s")
            print(f"  Generated: {report.num_strategies_generated} | "
                  f"Approved: {report.num_approved} | "
                  f"Rejected: {report.num_rejected}")

            if report.portfolio_metrics:
                print(f"  Portfolio Sharpe: {report.portfolio_metrics.get('sharpe_ratio', 0):.2f} | "
                      f"Return: {report.portfolio_metrics.get('total_return', 0) * 100:.1f}%")

        except Exception as e:
            print(f"\n✗ Iteration {i + 1} failed: {e}")
            import traceback
            traceback.print_exc()

    return reports


def analyze_database_patterns(database: StrategyDatabase):
    """
    Analyze patterns in the strategy database.

    Args:
        database: Strategy database
    """
    print_banner("Database Pattern Analysis")

    # Get all strategies
    all_strategies = database.get_recent_strategies(limit=1000)

    if not all_strategies:
        print("No strategies found in database.\n")
        return

    print(f"Total strategies in database: {len(all_strategies)}\n")

    # Analyze by type
    type_counts = {}
    for strategy in all_strategies:
        strategy_type = strategy.strategy_type or 'unknown'
        type_counts[strategy_type] = type_counts.get(strategy_type, 0) + 1

    print("Strategy Types:")
    for strategy_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_strategies)) * 100
        print(f"  {strategy_type:20s}: {count:3d} ({percentage:5.1f}%)")

    # Analyze genealogy
    with_parents = [s for s in all_strategies if s.parent_id is not None]
    print(f"\nGenealogy:")
    print(f"  Strategies with parents: {len(with_parents)} ({len(with_parents)/len(all_strategies)*100:.1f}%)")
    print(f"  Root strategies: {len(all_strategies) - len(with_parents)}")

    # Get backtest results for performance analysis
    try:
        # Query for strategies with backtest results
        strategies_with_results = []
        for strategy in all_strategies:
            results = database.get_backtest_results(strategy.id)
            if results:
                strategies_with_results.append((strategy, results[-1]))  # Most recent result

        if strategies_with_results:
            print(f"\nPerformance Statistics ({len(strategies_with_results)} strategies):")

            sharpe_ratios = [r.sharpe_ratio for s, r in strategies_with_results if r.sharpe_ratio is not None]
            returns = [r.total_return for s, r in strategies_with_results if r.total_return is not None]

            if sharpe_ratios:
                print(f"  Sharpe Ratio - Mean: {np.mean(sharpe_ratios):.2f}, "
                      f"Median: {np.median(sharpe_ratios):.2f}, "
                      f"Max: {np.max(sharpe_ratios):.2f}")

            if returns:
                print(f"  Total Return - Mean: {np.mean(returns)*100:.1f}%, "
                      f"Median: {np.median(returns)*100:.1f}%, "
                      f"Max: {np.max(returns)*100:.1f}%")

    except Exception as e:
        print(f"\n  (Could not analyze performance: {e})")

    print()


def visualize_genealogy_tree(database: StrategyDatabase, max_depth: int = 3):
    """
    Visualize strategy genealogy as a tree.

    Args:
        database: Strategy database
        max_depth: Maximum depth to display
    """
    print_banner("Strategy Genealogy Tree")

    all_strategies = database.get_recent_strategies(limit=1000)

    if not all_strategies:
        print("No strategies found.\n")
        return

    # Build parent-child map
    children_map = {}
    root_strategies = []

    for strategy in all_strategies:
        if strategy.parent_id is None:
            root_strategies.append(strategy)
        else:
            if strategy.parent_id not in children_map:
                children_map[strategy.parent_id] = []
            children_map[strategy.parent_id].append(strategy)

    if not root_strategies:
        print("No root strategies found.\n")
        return

    print(f"Found {len(root_strategies)} root strategies\n")

    # Display first few trees
    for idx, root in enumerate(root_strategies[:5]):  # Show first 5 trees
        print(f"Tree {idx + 1}: {root.strategy_name}")
        _print_tree_recursive(root, children_map, depth=0, max_depth=max_depth, prefix="")

    if len(root_strategies) > 5:
        print(f"\n... and {len(root_strategies) - 5} more genealogy trees\n")


def _print_tree_recursive(
    strategy,
    children_map: Dict,
    depth: int,
    max_depth: int,
    prefix: str
):
    """Recursively print strategy tree."""
    if depth >= max_depth:
        return

    children = children_map.get(strategy.id, [])

    for idx, child in enumerate(children):
        is_last = idx == len(children) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{child.strategy_name} ({child.strategy_type})")

        extension = "    " if is_last else "│   "
        _print_tree_recursive(child, children_map, depth + 1, max_depth, prefix + extension)


def print_summary(reports: List[Any]):
    """
    Print summary of all iterations.

    Args:
        reports: List of iteration reports
    """
    print_banner("Overall Summary")

    if not reports:
        print("No iterations completed.\n")
        return

    total_generated = sum(r.num_strategies_generated for r in reports)
    total_approved = sum(r.num_approved for r in reports)
    total_rejected = sum(r.num_rejected for r in reports)
    total_refinements = sum(r.num_refinements for r in reports)
    total_family_variants = sum(r.num_family_variants for r in reports)
    total_duration = sum(r.total_duration_seconds for r in reports)

    print(f"Iterations completed: {len(reports)}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"\nStrategy Generation:")
    print(f"  Total generated: {total_generated}")
    print(f"  Total approved: {total_approved} ({total_approved/total_generated*100:.1f}%)")
    print(f"  Total rejected: {total_rejected} ({total_rejected/total_generated*100:.1f}%)")
    print(f"  Refinements: {total_refinements}")
    print(f"  Family variants: {total_family_variants}")

    # Portfolio evolution
    print(f"\nPortfolio Evolution:")
    for i, report in enumerate(reports):
        if report.portfolio_metrics:
            metrics = report.portfolio_metrics
            print(f"  Iteration {i+1}: "
                  f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                  f"Return={metrics.get('total_return', 0)*100:.1f}%, "
                  f"Strategies={report.num_approved}")
        else:
            print(f"  Iteration {i+1}: No portfolio formed (no approved strategies)")

    print()


def export_results(reports: List[Any], output_path: str):
    """
    Export iteration results to JSON.

    Args:
        reports: List of iteration reports
        output_path: Path to output file
    """
    print(f"Exporting results to {output_path}...")

    results = []
    for report in reports:
        result_dict = {
            'iteration_number': report.iteration_number,
            'timestamp': report.timestamp.isoformat(),
            'num_strategies_generated': report.num_strategies_generated,
            'num_approved': report.num_approved,
            'num_rejected': report.num_rejected,
            'num_refinements': report.num_refinements,
            'num_family_variants': report.num_family_variants,
            'portfolio_metrics': report.portfolio_metrics,
            'forward_test_priorities': report.forward_test_priorities,
            'total_duration_seconds': report.total_duration_seconds
        }
        results.append(result_dict)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results exported\n")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description='Quantopia Full System Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of research iterations to run (default: 3)')
    parser.add_argument('--strategies-per-batch', type=int, default=5,
                       help='Number of strategies per iteration (default: 5)')
    parser.add_argument('--symbol', type=str, default='BTC-USD',
                       help='Trading symbol (default: BTC-USD)')
    parser.add_argument('--data-days', type=int, default=30,
                       help='Days of historical data (default: 30)')
    parser.add_argument('--data-freq', type=str, default='1H',
                       choices=['1H', '4H', '1D'],
                       help='Data frequency (default: 1H)')
    parser.add_argument('--database', type=str, default='data/quantopia_demo.db',
                       help='Database path (default: data/quantopia_demo.db)')
    parser.add_argument('--api-key', type=str, required=True,
                       help='OpenAI API key')
    parser.add_argument('--model', type=str, default='gpt-4',
                       help='LLM model to use (default: gpt-4)')
    parser.add_argument('--export', type=str,
                       help='Export results to JSON file')
    parser.add_argument('--skip-refinement', action='store_true',
                       help='Skip refinement phase')
    parser.add_argument('--skip-family-generation', action='store_true',
                       help='Skip family generation phase')

    args = parser.parse_args()

    # Print demo header
    print("\n" + "=" * 80)
    print("  QUANTOPIA AUTONOMOUS QUANT RESEARCH SYSTEM - FULL DEMONSTRATION")
    print("=" * 80)
    print(f"\n  Symbol: {args.symbol}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Strategies per batch: {args.strategies_per_batch}")
    print(f"  Model: {args.model}")
    print(f"  Database: {args.database}")
    print()

    # Create sample data
    data = create_sample_data(days=args.data_days, freq=args.data_freq)

    # Initialize orchestrator
    config = {
        'num_strategies_per_batch': args.strategies_per_batch,
        'enable_refinement': not args.skip_refinement,
        'enable_family_generation': not args.skip_family_generation
    }

    orchestrator = initialize_orchestrator(
        database_path=args.database,
        api_key=args.api_key,
        model=args.model,
        config=config
    )

    # Run iterations
    reports = run_demo_iterations(
        orchestrator=orchestrator,
        data=data,
        symbol=args.symbol,
        num_iterations=args.iterations
    )

    # Analyze results
    analyze_database_patterns(orchestrator.database)
    visualize_genealogy_tree(orchestrator.database, max_depth=3)
    print_summary(reports)

    # Export if requested
    if args.export:
        export_results(reports, args.export)

    # Final message
    print_banner("Demo Complete")
    print("Next steps:")
    print("  1. Review the iteration reports above")
    print("  2. Query the database for specific strategies")
    print(f"  3. Examine the SQLite database at: {args.database}")
    print("  4. Run forward tests on prioritized strategies")
    print("  5. Deploy top performers to paper trading")
    print()


if __name__ == '__main__':
    main()
