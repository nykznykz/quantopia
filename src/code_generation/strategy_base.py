"""Base strategy class for OmniAlpha generated strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import sys
import os

# Add simulated_exchange to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../simulated_exchange/src'))

from simulated_exchange import SimulatedExchange, HistoricalPriceFeed


class BaseStrategy(ABC):
    """Abstract base class for all generated strategies.

    This class provides the interface between LLM-generated strategies
    and the SimulatedExchange backtesting engine.
    """

    def __init__(
        self,
        exchange: SimulatedExchange,
        price_feed: HistoricalPriceFeed,
        symbol: str,
        initial_capital: float = 10000.0
    ):
        """Initialize strategy.

        Args:
            exchange: SimulatedExchange instance
            price_feed: Price feed for accessing market data
            symbol: Trading symbol (e.g., 'BTC-USD')
            initial_capital: Starting capital
        """
        self.exchange = exchange
        self.price_feed = price_feed
        self.symbol = symbol
        self.initial_capital = initial_capital

        # Strategy metadata (populated by subclasses)
        self.strategy_name = self.__class__.__name__
        self.strategy_type = "unknown"  # e.g., "mean_reversion", "trend_following"
        self.description = ""
        self.indicators = {}

        # ML model support (optional)
        self.ml_predictor = None
        self.ml_predictions = {}

        # Internal state
        self._current_bar_data = None
        self._position = None

    def load_ml_model(self, model_path: str, pipeline_path: Optional[str] = None) -> None:
        """Load ML model for predictions.

        Args:
            model_path: Path to saved ML model
            pipeline_path: Path to feature pipeline (optional)
        """
        from src.ml.prediction import ModelPredictor

        try:
            self.ml_predictor = ModelPredictor.from_files(
                model_path=model_path,
                pipeline_path=pipeline_path
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ML model: {e}")

    def get_ml_prediction(self, data: pd.DataFrame) -> float:
        """Get ML model prediction for current data.

        Args:
            data: Historical OHLCV DataFrame

        Returns:
            Model prediction (interpretation depends on model type)
        """
        if self.ml_predictor is None:
            raise RuntimeError("ML model not loaded. Call load_ml_model() first.")

        # Get prediction for latest data point
        prediction = self.ml_predictor.predict(data)

        # Return last prediction
        return float(prediction[-1]) if len(prediction) > 0 else 0.0

    def get_ml_signal(self, data: pd.DataFrame, threshold: float = 0.5) -> int:
        """Get trading signal from ML model.

        Args:
            data: Historical OHLCV DataFrame
            threshold: Confidence threshold

        Returns:
            Signal: 1 (buy), -1 (sell), 0 (hold)
        """
        if self.ml_predictor is None:
            return 0

        signals = self.ml_predictor.get_signals(data, threshold=threshold)
        return int(signals.iloc[-1]) if len(signals) > 0 else 0

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all indicators required by the strategy.

        This method should be implemented by generated strategies to compute
        all technical indicators needed for entry/exit decisions.

        Args:
            data: Historical OHLCV DataFrame up to current point

        Returns:
            Dict of indicator name -> current value
        """
        pass

    @abstractmethod
    def should_enter(self) -> bool:
        """Determine if strategy should enter a position.

        This method should use self.indicators to make entry decisions.

        Returns:
            True if should enter, False otherwise
        """
        pass

    @abstractmethod
    def should_exit(self) -> bool:
        """Determine if strategy should exit current position.

        This method should use self.indicators to make exit decisions.

        Returns:
            True if should exit, False otherwise
        """
        pass

    def position_size(self, capital: float, price: float) -> float:
        """Calculate position size for entry.

        Default implementation uses fixed percentage of capital.
        Can be overridden by generated strategies.

        Args:
            capital: Available capital
            price: Current asset price

        Returns:
            Position size (number of units to buy)
        """
        # Default: Use 90% of available capital
        allocation = capital * 0.9
        size = allocation / price
        return size

    def stop_loss(self, entry_price: float) -> Optional[float]:
        """Calculate stop loss price.

        Default implementation: No stop loss.
        Can be overridden by generated strategies.

        Args:
            entry_price: Entry price of position

        Returns:
            Stop loss price, or None if no stop loss
        """
        return None

    def take_profit(self, entry_price: float) -> Optional[float]:
        """Calculate take profit price.

        Default implementation: No take profit.
        Can be overridden by generated strategies.

        Args:
            entry_price: Entry price of position

        Returns:
            Take profit price, or None if no take profit
        """
        return None

    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest for this strategy.

        Main execution loop that coordinates with SimulatedExchange.

        Args:
            data: Full historical OHLCV DataFrame

        Returns:
            Dict with backtest results and performance metrics
        """
        # Reset exchange state
        self.exchange.reset()

        bar_count = 0
        trades_executed = 0

        # Main backtest loop
        while self.price_feed.has_next():
            # Get current market data up to this point
            current_idx = self.price_feed._current_index
            data_up_to_now = data.iloc[:current_idx + 1]

            # Calculate indicators
            self.indicators = self.calculate_indicators(data_up_to_now)

            # Get current position
            self._position = self.exchange.get_position(self.symbol)

            # Get current price
            current_price = self.price_feed.get_current_price(self.symbol)

            # Trading logic
            if self._position is None:
                # No position - check for entry
                if self.should_enter():
                    # Calculate position size
                    size = self.position_size(self.exchange.cash, current_price)

                    # Place buy order
                    result = self.exchange.place_order(
                        symbol=self.symbol,
                        side='buy',
                        size=size,
                        order_type='market'
                    )

                    if result['status'] == 'filled':
                        trades_executed += 1
            else:
                # Have position - check for exit
                should_exit_signal = self.should_exit()

                # Check stop loss
                stop_loss_price = self.stop_loss(self._position['avg_entry_price'])
                stop_loss_hit = (stop_loss_price is not None and
                                current_price <= stop_loss_price)

                # Check take profit
                take_profit_price = self.take_profit(self._position['avg_entry_price'])
                take_profit_hit = (take_profit_price is not None and
                                  current_price >= take_profit_price)

                if should_exit_signal or stop_loss_hit or take_profit_hit:
                    # Place sell order
                    result = self.exchange.place_order(
                        symbol=self.symbol,
                        side='sell',
                        size=self._position['size'],
                        order_type='market'
                    )

                    if result['status'] == 'filled':
                        trades_executed += 1

            # Update exchange state
            self.exchange.update()

            # Advance to next bar
            self.price_feed.next_bar()
            bar_count += 1

        # Close any remaining positions
        final_position = self.exchange.get_position(self.symbol)
        if final_position and final_position['size'] > 0:
            self.exchange.place_order(
                symbol=self.symbol,
                side='sell',
                size=final_position['size'],
                order_type='market'
            )

        # Get performance metrics
        metrics = self.exchange.get_performance_metrics()
        metrics['bars_processed'] = bar_count
        metrics['trades_executed'] = trades_executed

        # Get trade history and equity curve
        trade_history = self.exchange.get_trade_history()
        equity_curve = self.exchange.get_equity_curve()

        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'metrics': metrics,
            'trade_history': trade_history,
            'equity_curve': equity_curve,
            'final_portfolio_value': self.exchange.get_portfolio_value(),
            'final_cash': self.exchange.cash
        }

    def get_strategy_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata for logging/database.

        Returns:
            Dict with strategy information
        """
        return {
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type,
            'description': self.description,
            'symbol': self.symbol,
            'initial_capital': self.initial_capital
        }

    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.strategy_name}(symbol={self.symbol}, type={self.strategy_type})"
