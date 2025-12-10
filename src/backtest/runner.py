"""Backtest runner that orchestrates SimulatedExchange with generated strategies."""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, Type
import pandas as pd
import logging

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../simulated_exchange/src"))

from simulated_exchange import SimulatedExchange, HistoricalPriceFeed, download_data
from simulated_exchange.slippage import FixedSlippageModel
from simulated_exchange.fees import TieredFeeModel
from src.code_generation.strategy_base import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Orchestrates backtesting using SimulatedExchange."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        slippage_bps: float = 5.0,
        maker_fee: float = 0.0,
        taker_fee: float = 0.00025
    ):
        """Initialize backtest runner.

        Args:
            initial_capital: Starting capital in USD
            slippage_bps: Slippage in basis points (default: 5 bps)
            maker_fee: Maker fee percentage (default: 0% - Hyperliquid)
            taker_fee: Taker fee percentage (default: 0.025% - Hyperliquid)
        """
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        logger.info(
            f"Initialized BacktestRunner with capital=${initial_capital}, "
            f"slippage={slippage_bps}bps, fees={maker_fee}/{taker_fee}"
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
        """Run backtest for a strategy class.

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
            # Initialize SimulatedExchange components
            slippage_model = FixedSlippageModel(fixed_bps=self.slippage_bps)
            fee_model = TieredFeeModel(
                maker_fee=self.maker_fee,
                taker_fee=self.taker_fee
            )

            price_feed = HistoricalPriceFeed(data, symbol=symbol)

            exchange = SimulatedExchange(
                price_feed=price_feed,
                initial_capital=self.initial_capital,
                slippage_model=slippage_model,
                fee_model=fee_model
            )

            # Initialize strategy
            strategy = strategy_class(
                exchange=exchange,
                price_feed=price_feed,
                symbol=symbol,
                initial_capital=self.initial_capital
            )

            # Run backtest (strategy handles the main loop)
            backtest_start = datetime.now()
            results = strategy.run_backtest(data)
            backtest_duration = (datetime.now() - backtest_start).total_seconds()

            # Get performance metrics from SimulatedExchange
            metrics = exchange.get_performance_metrics()

            # Get trade history
            trades = exchange.get_trade_history()

            # Get equity curve
            equity_curve = pd.DataFrame([
                {
                    'timestamp': point.timestamp,
                    'equity': point.equity,
                    'cash': point.cash,
                    'position_value': point.position_value
                }
                for point in exchange.equity_curve
            ])

            # Compile results
            backtest_results = {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'start_date': data.index[0],
                'end_date': data.index[-1],
                'num_candles': len(data),
                'initial_capital': self.initial_capital,
                'final_capital': exchange.get_total_equity(),
                'metrics': metrics,
                'trades': trades,
                'equity_curve': equity_curve,
                'backtest_duration_seconds': backtest_duration,
                'status': 'success'
            }

            logger.info(f"✓ Backtest completed for {strategy_name}")
            logger.info(
                f"  Final Capital: ${exchange.get_total_equity():.2f} "
                f"({metrics.get('total_return', 0)*100:.2f}%)"
            )
            logger.info(
                f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                f"Max DD: {metrics.get('max_drawdown', 0)*100:.2f}%, "
                f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%"
            )
            logger.info(f"  Total Trades: {metrics.get('num_trades', 0)}")

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
        """Run backtest from generated strategy code.

        Args:
            strategy_code: Python code string (generated by CodeGenerator)
            data: Historical OHLCV data
            symbol: Trading symbol
            strategy_name: Optional strategy name

        Returns:
            Dict with backtest results
        """
        try:
            # Execute code to define strategy class
            namespace = {}
            exec(strategy_code, namespace)

            # Find strategy class (should inherit from BaseStrategy)
            strategy_class = None
            for name, obj in namespace.items():
                if (isinstance(obj, type) and
                    issubclass(obj, BaseStrategy) and
                    obj != BaseStrategy):
                    strategy_class = obj
                    break

            if strategy_class is None:
                raise ValueError("No BaseStrategy subclass found in generated code")

            # Run backtest with the class
            return self.run_backtest(
                strategy_class=strategy_class,
                data=data,
                symbol=symbol,
                strategy_name=strategy_name or strategy_class.__name__
            )

        except Exception as e:
            logger.error(f"Failed to run backtest from code: {e}")
            return {
                'strategy_name': strategy_name or 'Unknown',
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'metrics': {}
            }

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
