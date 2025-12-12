#!/usr/bin/env python3
"""
Quantopia - Autonomous Quant Research Platform

Main entry point for the Quantopia system.
Provides CLI interface and service management.
"""

import sys
import argparse
import logging
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli.commands import (
    run_research_session,
    start_daemon,
    stop_daemon,
    status_command,
    query_database,
    list_strategies,
    show_strategy,
    export_strategy
)
from src.cli.config_manager import load_config, init_config
from src.utils.logging_config import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Quantopia - Autonomous Quant Research Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize configuration
  quantopia init

  # Run a research session
  quantopia research --num-strategies 50 --symbol BTC-USD

  # Start as background daemon
  quantopia start --daemon

  # Check status
  quantopia status

  # List top strategies
  quantopia list --top 10

  # Show strategy details
  quantopia show <strategy_id>

  # Export strategy code
  quantopia export <strategy_id> --output strategy.py
"""
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/quantopia.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize Quantopia configuration')
    init_parser.add_argument('--force', action='store_true', help='Overwrite existing config')

    # Research command
    research_parser = subparsers.add_parser('research', help='Run research session')
    research_parser.add_argument(
        '--num-strategies',
        type=int,
        default=10,
        help='Number of strategies to generate'
    )
    research_parser.add_argument(
        '--symbol',
        type=str,
        default='BTC-USD',
        help='Trading symbol'
    )
    research_parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Days of historical data'
    )
    research_parser.add_argument(
        '--exploration-rate',
        type=float,
        default=0.3,
        help='Exploration vs exploitation rate (0-1)'
    )
    research_parser.add_argument(
        '--enable-ml',
        action='store_true',
        default=True,
        help='Enable ML strategies'
    )

    # Start daemon
    start_parser = subparsers.add_parser('start', help='Start Quantopia daemon')
    start_parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run in background as daemon'
    )
    start_parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuous research sessions'
    )
    start_parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Interval between sessions (seconds)'
    )

    # Stop daemon
    stop_parser = subparsers.add_parser('stop', help='Stop Quantopia daemon')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed statistics'
    )

    # List strategies
    list_parser = subparsers.add_parser('list', help='List strategies')
    list_parser.add_argument(
        '--top',
        type=int,
        help='Show top N strategies by Sharpe'
    )
    list_parser.add_argument(
        '--recent',
        type=int,
        help='Show N most recent strategies'
    )
    list_parser.add_argument(
        '--filter',
        type=str,
        choices=['all', 'approved', 'rejected', 'marginal'],
        default='all',
        help='Filter by status'
    )
    list_parser.add_argument(
        '--ml-type',
        type=str,
        choices=['pure_technical', 'hybrid_ml', 'pure_ml'],
        help='Filter by ML type'
    )

    # Show strategy
    show_parser = subparsers.add_parser('show', help='Show strategy details')
    show_parser.add_argument('strategy_id', type=str, help='Strategy ID or name')
    show_parser.add_argument('--code', action='store_true', help='Show code')
    show_parser.add_argument('--metrics', action='store_true', help='Show metrics')

    # Export strategy
    export_parser = subparsers.add_parser('export', help='Export strategy code')
    export_parser.add_argument('strategy_id', type=str, help='Strategy ID or name')
    export_parser.add_argument('--output', type=str, help='Output file path')

    # Query database
    query_parser = subparsers.add_parser('query', help='Query database')
    query_parser.add_argument(
        '--underexplored',
        action='store_true',
        help='Show underexplored areas'
    )
    query_parser.add_argument(
        '--statistics',
        action='store_true',
        help='Show database statistics'
    )
    query_parser.add_argument(
        '--models',
        action='store_true',
        help='Show ML models registry'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0

    try:
        # Load configuration
        if args.command != 'init':
            config = load_config(args.config)
        else:
            config = None

        # Execute command
        if args.command == 'init':
            init_config(args.config, force=args.force)

        elif args.command == 'research':
            run_research_session(
                config=config,
                num_strategies=args.num_strategies,
                symbol=args.symbol,
                days=args.days,
                exploration_rate=args.exploration_rate,
                enable_ml=args.enable_ml
            )

        elif args.command == 'start':
            start_daemon(
                config=config,
                daemon=args.daemon,
                continuous=args.continuous,
                interval=args.interval
            )

        elif args.command == 'stop':
            stop_daemon(config=config)

        elif args.command == 'status':
            status_command(config=config, detailed=args.detailed)

        elif args.command == 'list':
            list_strategies(
                config=config,
                top=args.top,
                recent=args.recent,
                filter_status=args.filter,
                ml_type=args.ml_type
            )

        elif args.command == 'show':
            show_strategy(
                config=config,
                strategy_id=args.strategy_id,
                show_code=args.code,
                show_metrics=args.metrics
            )

        elif args.command == 'export':
            export_strategy(
                config=config,
                strategy_id=args.strategy_id,
                output_path=args.output
            )

        elif args.command == 'query':
            query_database(
                config=config,
                underexplored=args.underexplored,
                statistics=args.statistics,
                models=args.models
            )

        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
