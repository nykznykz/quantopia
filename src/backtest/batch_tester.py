"""Batch testing for multiple strategies."""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.backtest.runner import BacktestRunner

logger = logging.getLogger(__name__)


class BatchTester:
    """Batch testing for multiple strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        slippage_bps: float = 5.0,
        maker_fee: float = 0.0,
        taker_fee: float = 0.00025,
        max_workers: Optional[int] = None,
        db: Optional[Any] = None
    ):
        """Initialize batch tester.

        Args:
            initial_capital: Starting capital for each backtest
            slippage_bps: Slippage in basis points
            maker_fee: Maker fee percentage
            taker_fee: Taker fee percentage
            max_workers: Maximum parallel workers (default: CPU count)
            db: Optional StrategyDatabase instance for persistent storage
        """
        self.runner = BacktestRunner(
            initial_capital=initial_capital,
            slippage_bps=slippage_bps,
            maker_fee=maker_fee,
            taker_fee=taker_fee
        )
        self.max_workers = max_workers
        self.db = db
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        logger.info(f"Initialized BatchTester with {max_workers or 'auto'} workers{' (with database)' if db else ''}")

    def run_batch(
        self,
        strategy_codes: List[str],
        strategy_names: List[str],
        data: pd.DataFrame,
        symbol: str,
        parallel: bool = False,
        strategy_ids: Optional[List[int]] = None,
        store_in_db: bool = True
    ) -> List[Dict[str, Any]]:
        """Run backtests for multiple strategies.

        Args:
            strategy_codes: List of Python code strings
            strategy_names: List of strategy names
            data: Historical OHLCV data (shared across all strategies)
            symbol: Trading symbol
            parallel: Whether to run in parallel (experimental)
            strategy_ids: Optional list of strategy database IDs
            store_in_db: Whether to store results in database (if db available)

        Returns:
            List of backtest results
        """
        if len(strategy_codes) != len(strategy_names):
            raise ValueError("strategy_codes and strategy_names must have same length")

        logger.info(f"Running batch backtest for {len(strategy_codes)} strategies")

        # Create scratchpad directory for debugging
        scratchpad_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scratchpad/strategies'))
        os.makedirs(scratchpad_dir, exist_ok=True)
        logger.info(f"Saving strategy codes to: {scratchpad_dir}")

        results = []

        if parallel:
            # Parallel execution (experimental - may have issues with shared state)
            logger.warning("Parallel execution is experimental")
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.runner.run_from_code,
                        code,
                        data,
                        symbol,
                        name
                    ): (name, idx)
                    for idx, (code, name) in enumerate(zip(strategy_codes, strategy_names))
                }

                for future in as_completed(futures):
                    strategy_name, idx = futures[future]
                    try:
                        result = future.result()

                        # Add strategy ID if available
                        if strategy_ids and idx < len(strategy_ids):
                            result['strategy_id'] = strategy_ids[idx]

                        results.append(result)
                        logger.info(f"✓ Completed: {strategy_name}")
                    except Exception as e:
                        logger.error(f"✗ Failed: {strategy_name} - {e}")
                        results.append({
                            'strategy_name': strategy_name,
                            'status': 'failed',
                            'error': str(e)
                        })
        else:
            # Sequential execution (recommended)
            for idx, (code, name) in enumerate(zip(strategy_codes, strategy_names)):
                try:
                    logger.info(f"Starting backtest {idx+1}/{len(strategy_codes)}: {name}")

                    # Save strategy code to scratchpad for debugging
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    code_filename = f"{timestamp}_{idx+1}_{safe_name}.py"
                    code_filepath = os.path.join(scratchpad_dir, code_filename)

                    with open(code_filepath, 'w') as f:
                        f.write(f"# Strategy: {name}\n")
                        f.write(f"# Timestamp: {timestamp}\n")
                        f.write(f"# Index: {idx+1}/{len(strategy_codes)}\n")
                        f.write(f"# Symbol: {symbol}\n")
                        f.write("#" + "="*70 + "\n\n")
                        f.write(code)

                    logger.info(f"  Saved strategy code to: {code_filepath}")

                    result = self.runner.run_from_code(
                        strategy_code=code,
                        data=data,
                        symbol=symbol,
                        strategy_name=name
                    )

                    # Save result status to the code file as a comment
                    with open(code_filepath, 'a') as f:
                        f.write(f"\n\n# {'='*70}\n")
                        f.write(f"# BACKTEST RESULT: {result.get('status', 'unknown')}\n")
                        if result.get('status') == 'failed':
                            f.write(f"# ERROR: {result.get('error', 'Unknown error')}\n")
                        elif result.get('status') == 'success':
                            metrics = result.get('metrics', {})
                            f.write(f"# Sharpe: {metrics.get('sharpe_ratio', 0):.2f}\n")
                            f.write(f"# Return: {metrics.get('total_return', 0):.2f}%\n")
                            f.write(f"# Max DD: {metrics.get('max_drawdown', 0):.2f}%\n")
                            f.write(f"# Trades: {metrics.get('total_trades', 0)}\n")

                    # Add strategy ID if available
                    if strategy_ids and idx < len(strategy_ids):
                        result['strategy_id'] = strategy_ids[idx]

                    results.append(result)
                    logger.info(f"✓ Completed: {name}")
                except Exception as e:
                    logger.error(f"✗ Failed: {name} - {e}")
                    # Log full error details for debugging
                    import traceback
                    logger.debug(f"Full traceback:\n{traceback.format_exc()}")

                    results.append({
                        'strategy_name': name,
                        'status': 'failed',
                        'error': str(e),
                        'metrics': {}
                    })
                    # Continue with next strategy instead of stopping

        # Store results in database if enabled
        if self.db and store_in_db:
            self._store_results_in_db(results, data, symbol)

        # Summary statistics
        successful = sum(1 for r in results if r.get('status') == 'success')
        logger.info(
            f"✓ Batch complete: {successful}/{len(strategy_codes)} strategies succeeded"
        )

        return results

    def _store_results_in_db(
        self,
        results: List[Dict[str, Any]],
        data: pd.DataFrame,
        symbol: str
    ):
        """Store backtest results in database.

        Args:
            results: List of backtest results
            data: Historical data used
            symbol: Trading symbol
        """
        for result in results:
            if result.get('status') != 'success':
                continue

            strategy_id = result.get('strategy_id')
            if not strategy_id:
                logger.warning(f"No strategy_id for {result.get('strategy_name')}, skipping DB storage")
                continue

            try:
                metrics = result.get('metrics', {})

                # Extract equity curve if available
                equity_curve = None
                if 'equity_curve' in result:
                    equity_curve = [
                        {'timestamp': str(ts), 'equity': equity}
                        for ts, equity in result['equity_curve'].items()
                    ]

                # Prepare backtest config
                backtest_config = {
                    'slippage_bps': self.slippage_bps,
                    'maker_fee': self.maker_fee,
                    'taker_fee': self.taker_fee
                }

                # Get filter result if available
                filter_result = result.get('filter_result', {})

                # Store in database
                backtest_id = self.db.store_backtest_results(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    timeframe='unknown',  # TODO: Extract from data if available
                    start_date=data.index[0] if not data.empty else datetime.now(),
                    end_date=data.index[-1] if not data.empty else datetime.now(),
                    initial_capital=self.initial_capital,
                    metrics={
                        'total_return_pct': metrics.get('total_return', 0) * 100,
                        'sharpe_ratio': metrics.get('sharpe_ratio'),
                        'max_drawdown_pct': abs(metrics.get('max_drawdown', 0)) * 100,
                        'win_rate': metrics.get('win_rate'),
                        'profit_factor': metrics.get('profit_factor'),
                        'num_trades': metrics.get('num_trades'),
                        'avg_trade_return_pct': metrics.get('avg_trade', 0) * 100,
                        'max_consecutive_wins': metrics.get('max_consecutive_wins'),
                        'max_consecutive_losses': metrics.get('max_consecutive_losses')
                    },
                    equity_curve=equity_curve,
                    passed_filters=filter_result.get('passed', False),
                    filter_category=filter_result.get('category'),
                    rejection_reasons=filter_result.get('reasons', []),
                    backtest_config=backtest_config
                )

                logger.info(
                    f"Stored backtest result in database: strategy_id={strategy_id}, "
                    f"backtest_id={backtest_id}"
                )

            except Exception as e:
                logger.error(f"Failed to store backtest result in database: {e}")

    def summarize_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create summary DataFrame from batch results.

        Args:
            results: List of backtest results

        Returns:
            DataFrame with key metrics for each strategy
        """
        summary_data = []

        for result in results:
            metrics = result.get('metrics', {})

            summary_data.append({
                'strategy_name': result.get('strategy_name', 'Unknown'),
                'status': result.get('status', 'unknown'),
                'final_capital': result.get('final_capital', 0),
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'num_trades': metrics.get('num_trades', 0),
                'avg_trade': metrics.get('avg_trade', 0),
                'total_fees': metrics.get('total_fees', 0),
                'error': result.get('error', '')
            })

        df = pd.DataFrame(summary_data)

        # Sort by total return (descending)
        if 'total_return' in df.columns:
            df = df.sort_values('total_return', ascending=False)

        return df

    def save_batch_results(
        self,
        results: List[Dict[str, Any]],
        output_dir: str = 'results/backtests/batch'
    ) -> str:
        """Save batch results to disk.

        Args:
            results: List of backtest results
            output_dir: Output directory

        Returns:
            Path to saved summary
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_name = f"batch_{timestamp}"

        # Save individual strategy results
        for result in results:
            if result.get('status') == 'success':
                self.runner.save_results(result, output_dir)

        # Save batch summary
        summary_df = self.summarize_results(results)
        summary_path = os.path.join(output_dir, f"{batch_name}_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        logger.info(f"Saved batch summary to: {summary_path}")

        return summary_path
