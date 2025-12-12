"""Backtest runner that orchestrates SimulatedExchange with generated strategies."""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, Type
import pandas as pd
import logging
import multiprocessing as mp
from functools import partial

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../simulated_exchange/src"))

from simulated_exchange import SimulatedExchange, HistoricalPriceFeed, download_data
from src.code_generation.strategy_base import BaseStrategy

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when a backtest times out."""
    pass


def _run_strategy_in_process(
    strategy_code: str,
    strategy_name: str,
    data: pd.DataFrame,
    symbol: str,
    initial_capital: float,
    slippage_bps: float,
    maker_fee: float,
    taker_fee: float,
    result_queue: mp.Queue
):
    """Helper function to run backtest in separate process.

    Args:
        strategy_code: Python code string containing strategy class definition
        strategy_name: Name of the strategy (for error reporting)
        data: Historical data
        symbol: Trading symbol
        initial_capital: Starting capital
        slippage_bps: Slippage in basis points
        maker_fee: Maker fee
        taker_fee: Taker fee
        result_queue: Queue to put results
    """
    try:
        import importlib.util
        import importlib
        import tempfile

        # CRITICAL: Disable bytecode caching to ensure fresh imports
        sys.dont_write_bytecode = True

        # CRITICAL: Aggressive cache busting - remove ALL strategy-related modules from cache
        # This ensures subprocess uses latest code, not cached .pyc files
        modules_to_remove = [k for k in list(sys.modules.keys()) if 'strategy' in k.lower() or 'code_generation' in k]
        for mod in modules_to_remove:
            try:
                del sys.modules[mod]
            except:
                pass

        # Dynamically create the strategy class from code in this process
        # Generate a unique module name
        module_name = f'strategy_{id(strategy_code)}'

        # Create a temporary file for the strategy code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(strategy_code)
            temp_file_path = temp_file.name

        try:
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Failed to create module spec for strategy")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find strategy class
            strategy_class = None
            import logging
            logger = logging.getLogger(__name__)

            logger.info(f"[SUBPROCESS] Searching for BaseStrategy subclass in module...")
            logger.info(f"[SUBPROCESS] Module members: {[name for name in dir(module) if not name.startswith('_')]}")

            for name in dir(module):
                obj = getattr(module, name)
                if not isinstance(obj, type):
                    continue

                # Check if this is a BaseStrategy subclass by checking base class names
                # This avoids issues with module reloading and different BaseStrategy instances
                try:
                    base_names = [base.__name__ for base in obj.__mro__]
                    if 'BaseStrategy' in base_names and obj.__name__ != 'BaseStrategy':
                        strategy_class = obj
                        logger.info(f"[SUBPROCESS] Found strategy class: {strategy_class}")
                        break
                except Exception as e:
                    continue

            if strategy_class is None:
                logger.error(f"[SUBPROCESS] No BaseStrategy subclass found!")
                logger.error(f"[SUBPROCESS] Available classes: {[name for name in dir(module) if isinstance(getattr(module, name), type)]}")
                raise ValueError("No BaseStrategy subclass found in generated code")

            # Initialize SimulatedExchange with config dictionaries
            slippage_config = {
                'type': 'fixed',
                'fixed_bps': slippage_bps
            }
            fee_config = {
                'type': 'tiered',
                'maker_fee': maker_fee,
                'taker_fee': taker_fee
            }

            price_feed = HistoricalPriceFeed(data, symbols=[symbol])

            exchange = SimulatedExchange(
                price_feed=price_feed,
                initial_capital=initial_capital,
                slippage_config=slippage_config,
                fee_config=fee_config
            )

            # Initialize strategy
            strategy = strategy_class(
                exchange=exchange,
                price_feed=price_feed,
                symbol=symbol,
                initial_capital=initial_capital
            )

            # Run backtest
            results = strategy.run_backtest(data)

            # Get performance metrics from SimulatedExchange
            metrics = exchange.get_performance_metrics()

            # Get trade history
            trades = exchange.get_trade_history()

            # Get equity curve
            equity_curve = pd.DataFrame([
                {
                    'timestamp': point.timestamp,
                    'equity': point.portfolio_value,
                    'cash': point.cash,
                    'position_value': point.portfolio_value - point.cash
                }
                for point in exchange.equity_curve
            ])

            # Compile full results
            full_results = {
                'metrics': metrics,
                'trade_history': trades,
                'equity_curve': equity_curve,
                'final_portfolio_value': exchange.get_portfolio_value(),
                'final_cash': exchange.cash
            }

            # Put results in queue
            result_queue.put(('success', full_results))

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

    except Exception as e:
        import traceback
        result_queue.put(('error', f"{str(e)}\n{traceback.format_exc()}"))


class BacktestRunner:
    """Orchestrates backtesting using SimulatedExchange."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        slippage_bps: float = 5.0,
        maker_fee: float = 0.0,
        taker_fee: float = 0.00025,
        backtest_timeout_seconds: int = 60
    ):
        """Initialize backtest runner.

        Args:
            initial_capital: Starting capital in USD
            slippage_bps: Slippage in basis points (default: 5 bps)
            maker_fee: Maker fee percentage (default: 0% - Hyperliquid)
            taker_fee: Taker fee percentage (default: 0.025% - Hyperliquid)
            backtest_timeout_seconds: Maximum time for a single backtest (default: 60s)
        """
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.backtest_timeout_seconds = backtest_timeout_seconds

        logger.info(
            f"Initialized BacktestRunner with capital=${initial_capital}, "
            f"slippage={slippage_bps}bps, fees={maker_fee}/{taker_fee}, "
            f"timeout={backtest_timeout_seconds}s"
        )

    def load_data(
        self,
        symbol: str,
        exchange: str = 'binance',
        timeframe: str = '1h',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cache_dir: str = 'data/historical'
    ) -> pd.DataFrame:
        """Load historical OHLCV data.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            exchange: Exchange name (default: 'binance')
            timeframe: Candle timeframe (default: '1h')
            start_date: Start date (default: 2 years ago)
            end_date: End date (default: now)
            cache_dir: Data cache directory

        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = datetime.now().replace(year=datetime.now().year - 2)
        if end_date is None:
            end_date = datetime.now()

        logger.info(
            f"Loading data: {symbol} from {exchange} "
            f"({start_date.date()} to {end_date.date()}, {timeframe})"
        )

        try:
            df = download_data(
                symbols=[symbol],
                exchange=exchange,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir
            )

            logger.info(f"Loaded {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def run_backtest(
        self,
        strategy_class: Type[BaseStrategy],
        data: pd.DataFrame,
        symbol: str,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run backtest for a strategy class (in-process, no multiprocessing).

        NOTE: This method runs in-process and is suitable for pre-defined strategy
        classes. For dynamically generated strategies, use run_from_code() instead
        which uses multiprocessing with proper code serialization.

        Args:
            strategy_class: Strategy class (must inherit from BaseStrategy)
            data: Historical OHLCV data
            symbol: Trading symbol
            strategy_name: Optional strategy name for logging

        Returns:
            Dict with backtest results and metrics
        """
        strategy_name = strategy_name or strategy_class.__name__

        logger.info(f"Starting backtest for: {strategy_name}")
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]} ({len(data)} candles)")

        try:
            backtest_start = datetime.now()

            # Initialize SimulatedExchange with config dictionaries
            slippage_config = {
                'type': 'fixed',
                'fixed_bps': self.slippage_bps
            }
            fee_config = {
                'type': 'tiered',
                'maker_fee': self.maker_fee,
                'taker_fee': self.taker_fee
            }

            price_feed = HistoricalPriceFeed(data, symbols=[symbol])

            exchange = SimulatedExchange(
                price_feed=price_feed,
                initial_capital=self.initial_capital,
                slippage_config=slippage_config,
                fee_config=fee_config
            )

            # Initialize strategy
            strategy = strategy_class(
                exchange=exchange,
                price_feed=price_feed,
                symbol=symbol,
                initial_capital=self.initial_capital
            )

            # Run backtest
            results = strategy.run_backtest(data)

            # Get performance metrics from SimulatedExchange
            metrics = exchange.get_performance_metrics()

            # Get trade history
            trades = exchange.get_trade_history()

            # Get equity curve
            equity_curve = pd.DataFrame([
                {
                    'timestamp': point.timestamp,
                    'equity': point.portfolio_value,
                    'cash': point.cash,
                    'position_value': point.portfolio_value - point.cash
                }
                for point in exchange.equity_curve
            ])

            backtest_duration = (datetime.now() - backtest_start).total_seconds()

            # Compile results
            # Use timestamp column if available, otherwise fall back to index
            if 'timestamp' in data.columns:
                start_date = data['timestamp'].iloc[0] if hasattr(data['timestamp'].iloc[0], 'to_pydatetime') else data['timestamp'].iloc[0]
                end_date = data['timestamp'].iloc[-1] if hasattr(data['timestamp'].iloc[-1], 'to_pydatetime') else data['timestamp'].iloc[-1]
            else:
                start_date = data.index[0] if isinstance(data.index[0], (datetime, pd.Timestamp)) else datetime.fromtimestamp(0)
                end_date = data.index[-1] if isinstance(data.index[-1], (datetime, pd.Timestamp)) else datetime.now()

            backtest_results = {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'num_candles': len(data),
                'initial_capital': self.initial_capital,
                'final_capital': results.get('final_portfolio_value', self.initial_capital),
                'metrics': metrics,
                'trades': trades,
                'equity_curve': equity_curve,
                'backtest_duration_seconds': backtest_duration,
                'status': 'success'
            }

            logger.info(f"✓ Backtest completed for {strategy_name}")
            logger.info(
                f"  Final Capital: ${results.get('final_portfolio_value', self.initial_capital):.2f} "
                f"({metrics.get('total_return', 0):.2f}%)"
            )
            logger.info(
                f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                f"Max DD: {metrics.get('max_drawdown', 0):.2f}%, "
                f"Win Rate: {metrics.get('win_rate', 0):.2f}%"
            )
            logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")

            return backtest_results

        except Exception as e:
            logger.error(f"Backtest failed for {strategy_name}: {e}")
            return {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'metrics': {}
            }

    def run_from_code(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        symbol: str,
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run backtest from generated strategy code (IN-PROCESS for speed).

        NOTE: With O(n) optimized indicator pre-calculation, backtests complete in
        ~0.5s for 8760 candles, so multiprocessing overhead isn't worth it.

        Args:
            strategy_code: Python code string (generated by CodeGenerator)
            data: Historical OHLCV data
            symbol: Trading symbol
            strategy_name: Optional strategy name

        Returns:
            Dict with backtest results
        """
        import importlib.util
        import tempfile

        # Infer strategy name from code if not provided
        if not strategy_name:
            strategy_name = f'Strategy_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'

        logger.info(f"Starting backtest for: {strategy_name}")
        logger.info(f"Data range: {data.index[0]} to {data.index[-1]} ({len(data)} candles)")

        temp_file_path = None

        try:
            backtest_start = datetime.now()

            # Create temporary file with strategy code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(strategy_code)
                temp_file_path = temp_file.name

            # Import the strategy module
            module_name = f'strategy_{datetime.now().strftime("%Y%m%d%H%M%S%f")}'
            spec = importlib.util.spec_from_file_location(module_name, temp_file_path)

            if spec is None or spec.loader is None:
                raise ValueError("Failed to create module spec for strategy")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find strategy class by checking base class names
            strategy_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if not isinstance(obj, type):
                    continue

                try:
                    base_names = [base.__name__ for base in obj.__mro__]
                    if 'BaseStrategy' in base_names and obj.__name__ != 'BaseStrategy':
                        strategy_class = obj
                        break
                except:
                    continue

            if strategy_class is None:
                raise ValueError("No BaseStrategy subclass found in generated code")

            # Initialize exchange components
            slippage_config = {'type': 'fixed', 'fixed_bps': self.slippage_bps}
            fee_config = {'type': 'tiered', 'maker_fee': self.maker_fee, 'taker_fee': self.taker_fee}

            price_feed = HistoricalPriceFeed(data, symbols=[symbol])
            exchange = SimulatedExchange(
                price_feed=price_feed,
                initial_capital=self.initial_capital,
                slippage_config=slippage_config,
                fee_config=fee_config
            )

            # Initialize and run strategy
            strategy = strategy_class(
                exchange=exchange,
                price_feed=price_feed,
                symbol=symbol,
                initial_capital=self.initial_capital
            )

            # Run backtest
            strategy.run_backtest(data)

            # Get results
            metrics = exchange.get_performance_metrics()
            trades = exchange.get_trade_history()
            equity_curve = pd.DataFrame([
                {
                    'timestamp': point.timestamp,
                    'equity': point.portfolio_value,
                    'cash': point.cash,
                    'position_value': point.portfolio_value - point.cash
                }
                for point in exchange.equity_curve
            ])

            backtest_duration = (datetime.now() - backtest_start).total_seconds()

            # Compile results
            if 'timestamp' in data.columns:
                start_date = data['timestamp'].iloc[0] if hasattr(data['timestamp'].iloc[0], 'to_pydatetime') else data['timestamp'].iloc[0]
                end_date = data['timestamp'].iloc[-1] if hasattr(data['timestamp'].iloc[-1], 'to_pydatetime') else data['timestamp'].iloc[-1]
            else:
                start_date = data.index[0] if isinstance(data.index[0], (datetime, pd.Timestamp)) else datetime.fromtimestamp(0)
                end_date = data.index[-1] if isinstance(data.index[-1], (datetime, pd.Timestamp)) else datetime.now()

            backtest_results = {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'num_candles': len(data),
                'initial_capital': self.initial_capital,
                'final_capital': exchange.get_portfolio_value(),
                'metrics': metrics,
                'trades': trades,
                'equity_curve': equity_curve,
                'backtest_duration_seconds': backtest_duration,
                'status': 'success'
            }

            logger.info(f"✓ Backtest completed for {strategy_name}")
            logger.info(
                f"  Final Capital: ${exchange.get_portfolio_value():.2f} "
                f"({metrics.get('total_return', 0):.2f}%)"
            )
            logger.info(
                f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                f"Max DD: {metrics.get('max_drawdown', 0):.2f}%, "
                f"Win Rate: {metrics.get('win_rate', 0):.2f}%"
            )
            logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")

            return backtest_results

        except Exception as e:
            logger.error(f"Backtest failed for {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'metrics': {}
            }
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str = 'results/backtests'
    ) -> str:
        """Save backtest results to disk.

        Args:
            results: Backtest results dict
            output_dir: Output directory

        Returns:
            Path to saved results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strategy_name = results['strategy_name'].replace(' ', '_')
        base_filename = f"{strategy_name}_{timestamp}"

        # Save trades as CSV
        if 'trades' in results and isinstance(results['trades'], pd.DataFrame):
            trades_path = os.path.join(output_dir, f"{base_filename}_trades.csv")
            results['trades'].to_csv(trades_path, index=False)
            logger.info(f"Saved trades to: {trades_path}")

        # Save equity curve as CSV
        if 'equity_curve' in results and isinstance(results['equity_curve'], pd.DataFrame):
            equity_path = os.path.join(output_dir, f"{base_filename}_equity.csv")
            results['equity_curve'].to_csv(equity_path, index=False)
            logger.info(f"Saved equity curve to: {equity_path}")

        # Save summary as JSON
        import json
        summary = {
            'strategy_name': results['strategy_name'],
            'symbol': results.get('symbol'),
            'start_date': str(results.get('start_date')),
            'end_date': str(results.get('end_date')),
            'initial_capital': results.get('initial_capital'),
            'final_capital': results.get('final_capital'),
            'metrics': results.get('metrics', {}),
            'status': results.get('status'),
            'error': results.get('error'),
            'backtest_duration_seconds': results.get('backtest_duration_seconds')
        }

        summary_path = os.path.join(output_dir, f"{base_filename}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved summary to: {summary_path}")

        return summary_path
