"""SimulatedExchange - Core exchange implementation."""

from datetime import datetime
from typing import Dict, List, Optional
import uuid
import logging
import pandas as pd

from .models import (
    Order, Trade, Position, EquityCurvePoint, OHLCV,
    OrderSide, OrderType, OrderStatus
)
from .price_feed import PriceFeed
from .slippage import SlippageModel, create_slippage_model
from .fees import FeeModel, create_fee_model
from .performance import calculate_performance_metrics, get_equity_curve_df, get_trade_history_df
from .exceptions import (
    InsufficientFundsError, InvalidOrderError,
    OrderNotFoundError, PositionNotFoundError
)

logger = logging.getLogger(__name__)


class SimulatedExchange:
    """Simulated cryptocurrency exchange for backtesting and paper trading."""

    def __init__(
        self,
        price_feed: PriceFeed,
        initial_capital: float = 10000.0,
        mode: str = 'backtest',
        slippage_config: Optional[Dict] = None,
        fee_config: Optional[Dict] = None
    ):
        """Initialize simulated exchange.

        Args:
            price_feed: Historical or live price feed
            initial_capital: Starting capital in USD
            mode: Operating mode ('backtest' or 'live')
            slippage_config: Slippage model configuration
            fee_config: Fee model configuration
        """
        self.price_feed = price_feed
        self.initial_capital = initial_capital
        self.mode = mode

        # Initialize models
        self.slippage_model: SlippageModel = create_slippage_model(slippage_config)
        self.fee_model: FeeModel = create_fee_model(fee_config)

        # State variables
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.open_orders: List[Order] = []
        self.trade_history: List[Trade] = []
        self.equity_curve: List[EquityCurvePoint] = []
        self.realized_pnl = 0.0

        # Track order/trade IDs
        self._order_counter = 0
        self._trade_counter = 0

        logger.info(
            f"Initialized SimulatedExchange: mode={mode}, "
            f"capital=${initial_capital:.2f}"
        )

        # Record initial equity curve point
        self._record_equity_curve_point()

    def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = 'market',
        limit_price: Optional[float] = None
    ) -> dict:
        """Place a new order.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            side: Order side ('buy' or 'sell')
            size: Order size
            order_type: Order type ('market' or 'limit')
            limit_price: Limit price (required for limit orders)

        Returns:
            Order result dict with status, filled size, fees, etc.

        Raises:
            InvalidOrderError: If order parameters are invalid
            InsufficientFundsError: If insufficient funds/position
        """
        # Convert strings to enums
        try:
            side_enum = OrderSide(side.lower())
            type_enum = OrderType(order_type.lower())
        except ValueError as e:
            raise InvalidOrderError(f"Invalid order parameter: {e}")

        # Validate order
        self._validate_order(symbol, side_enum, size, type_enum, limit_price)

        # Create order
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side_enum,
            order_type=type_enum,
            size=size,
            timestamp=self.price_feed.get_current_timestamp(),
            limit_price=limit_price
        )

        logger.info(
            f"Placing order: {order.order_id} {side} {size} {symbol} "
            f"@ {order_type}" + (f" {limit_price}" if limit_price else "")
        )

        # Try to execute or queue order
        if type_enum == OrderType.MARKET:
            return self._execute_market_order(order)
        else:  # LIMIT
            return self._handle_limit_order(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found or already filled
        """
        for order in self.open_orders:
            if order.order_id == order_id:
                if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
                    order.status = OrderStatus.CANCELLED
                    self.open_orders.remove(order)
                    logger.info(f"Cancelled order: {order_id}")
                    return True
                else:
                    logger.warning(
                        f"Cannot cancel order {order_id}: status={order.status.value}"
                    )
                    return False

        logger.warning(f"Order not found: {order_id}")
        return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open order dicts
        """
        orders = self.open_orders
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return [o.to_dict() for o in orders]

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get current position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position dict or None if no position
        """
        position = self.positions.get(symbol)
        if position is None:
            return None

        try:
            current_price = self.price_feed.get_current_price(symbol)
            return position.to_dict(current_price)
        except Exception as e:
            logger.error(f"Error getting position for {symbol}: {e}")
            return position.to_dict()

    def get_all_positions(self) -> Dict[str, dict]:
        """Get all open positions.

        Returns:
            Dict mapping symbol to position info
        """
        result = {}
        for symbol, position in self.positions.items():
            result[symbol] = self.get_position(symbol)
        return result

    def get_portfolio_value(self) -> float:
        """Get current total portfolio value.

        Returns:
            Total portfolio value (cash + position values)
        """
        portfolio_value = self.cash

        for symbol, position in self.positions.items():
            try:
                current_price = self.price_feed.get_current_price(symbol)
                position_value = position.size * current_price
                portfolio_value += position_value
            except Exception as e:
                logger.error(f"Error calculating position value for {symbol}: {e}")

        return portfolio_value

    def get_pnl(self) -> dict:
        """Get profit and loss summary.

        Returns:
            PnL dict with realized, unrealized, total, percentage, and fees
        """
        unrealized_pnl = 0.0

        for symbol, position in self.positions.items():
            try:
                current_price = self.price_feed.get_current_price(symbol)
                unrealized_pnl += (current_price - position.avg_entry_price) * position.size
            except Exception as e:
                logger.error(f"Error calculating unrealized PnL for {symbol}: {e}")

        total_pnl = self.realized_pnl + unrealized_pnl
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        total_fees = sum(trade.fee for trade in self.trade_history)

        return {
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'total_fees': total_fees
        }

    def get_account_info(self) -> dict:
        """Get current account summary.

        Returns:
            Account info dict
        """
        return {
            'cash': self.cash,
            'portfolio_value': self.get_portfolio_value(),
            'buying_power': self.cash,  # No leverage for now
            'open_positions': len(self.positions),
            'open_orders': len(self.open_orders)
        }

    def reset(self):
        """Reset exchange to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.open_orders = []
        self.trade_history = []
        self.equity_curve = []
        self.realized_pnl = 0.0
        self._order_counter = 0
        self._trade_counter = 0
        self._record_equity_curve_point()
        logger.info("Exchange reset to initial state")

    def update(self):
        """Update exchange state (check limit orders, record equity).

        Should be called after price feed updates.
        """
        self._check_limit_orders()
        self._record_equity_curve_point()

    def get_performance_metrics(self) -> dict:
        """Calculate comprehensive performance metrics.

        Returns:
            Dict with all performance metrics including:
            - total_return, sharpe_ratio, max_drawdown
            - win_rate, profit_factor, avg_win, avg_loss
            - total_trades, total_fees, etc.
        """
        return calculate_performance_metrics(
            trade_history=self.trade_history,
            equity_curve=self.equity_curve,
            initial_capital=self.initial_capital
        )

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as pandas DataFrame.

        Returns:
            DataFrame with columns: timestamp, portfolio_value, cash,
                                   unrealized_pnl, realized_pnl
        """
        return get_equity_curve_df(self.equity_curve)

    def get_trade_history(self) -> pd.DataFrame:
        """Get all trades as pandas DataFrame.

        Returns:
            DataFrame with all trade details
        """
        return get_trade_history_df(self.trade_history)

    # Private methods

    def _validate_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        order_type: OrderType,
        limit_price: Optional[float]
    ):
        """Validate order parameters."""
        if size <= 0:
            raise InvalidOrderError("Order size must be positive")

        if order_type == OrderType.LIMIT and limit_price is None:
            raise InvalidOrderError("Limit price required for limit orders")

        if limit_price is not None and limit_price <= 0:
            raise InvalidOrderError("Limit price must be positive")

        # Check funds for buy orders
        if side == OrderSide.BUY:
            try:
                current_price = self.price_feed.get_current_price(symbol)
                estimated_cost = size * (limit_price if limit_price else current_price)
                # Add buffer for slippage and fees
                estimated_cost *= 1.01

                if estimated_cost > self.cash:
                    raise InsufficientFundsError(
                        f"Insufficient funds: need ${estimated_cost:.2f}, "
                        f"have ${self.cash:.2f}"
                    )
            except Exception as e:
                logger.error(f"Error validating buy order: {e}")
                raise

        # Check position for sell orders
        elif side == OrderSide.SELL:
            position = self.positions.get(symbol)
            if position is None or position.size < size:
                current_size = position.size if position else 0
                raise InsufficientFundsError(
                    f"Insufficient position: trying to sell {size}, "
                    f"have {current_size}"
                )

    def _execute_market_order(self, order: Order) -> dict:
        """Execute a market order immediately."""
        try:
            # Get current price and volume
            ohlcv = self.price_feed.get_ohlcv(order.symbol)
            current_price = ohlcv.close
            volume = ohlcv.volume

            # Calculate slipped price
            fill_price = self.slippage_model.calculate_slippage(
                price=current_price,
                size=order.size,
                side=order.side,
                volume=volume
            )

            # Calculate trade value and fee
            trade_value = order.size * fill_price
            fee = self.fee_model.calculate_fee(trade_value, order.order_type)

            # Calculate slippage amount
            slippage = abs(fill_price - current_price)

            # Update order
            order.status = OrderStatus.FILLED
            order.filled_size = order.size
            order.avg_fill_price = fill_price

            # Create trade
            trade = Trade(
                trade_id=self._generate_trade_id(),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                fill_price=fill_price,
                slippage=slippage,
                fee=fee,
                timestamp=order.timestamp
            )

            # Update positions and cash
            self._update_position(trade)

            # Record trade
            self.trade_history.append(trade)

            logger.info(
                f"Executed market order: {order.order_id} {order.side.value} "
                f"{order.size} {order.symbol} @ ${fill_price:.2f} "
                f"(fee: ${fee:.4f})"
            )

            return {
                'order_id': order.order_id,
                'status': 'filled',
                'filled_size': order.size,
                'avg_fill_price': fill_price,
                'fee': fee,
                'message': 'Order filled successfully'
            }

        except Exception as e:
            logger.error(f"Failed to execute market order: {e}")
            order.status = OrderStatus.REJECTED
            return {
                'order_id': order.order_id,
                'status': 'rejected',
                'filled_size': 0.0,
                'avg_fill_price': 0.0,
                'fee': 0.0,
                'message': f'Order rejected: {str(e)}'
            }

    def _handle_limit_order(self, order: Order) -> dict:
        """Handle a limit order (try to execute or queue)."""
        # Try immediate execution
        try:
            current_price = self.price_feed.get_current_price(order.symbol)

            should_fill = False
            if order.side == OrderSide.BUY and current_price <= order.limit_price:
                should_fill = True
            elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                should_fill = True

            if should_fill:
                # Execute at limit price (or better)
                return self._execute_limit_order(order, current_price)
            else:
                # Queue order
                self.open_orders.append(order)
                logger.info(f"Queued limit order: {order.order_id}")
                return {
                    'order_id': order.order_id,
                    'status': 'pending',
                    'filled_size': 0.0,
                    'avg_fill_price': 0.0,
                    'fee': 0.0,
                    'message': 'Order queued'
                }

        except Exception as e:
            logger.error(f"Failed to handle limit order: {e}")
            order.status = OrderStatus.REJECTED
            return {
                'order_id': order.order_id,
                'status': 'rejected',
                'filled_size': 0.0,
                'avg_fill_price': 0.0,
                'fee': 0.0,
                'message': f'Order rejected: {str(e)}'
            }

    def _execute_limit_order(self, order: Order, current_price: float) -> dict:
        """Execute a limit order."""
        try:
            # Use limit price as fill price (limit orders get their price or better)
            fill_price = order.limit_price

            # Get volume for slippage calculation
            try:
                ohlcv = self.price_feed.get_ohlcv(order.symbol)
                volume = ohlcv.volume
            except:
                volume = None

            # Apply minimal slippage even for limit orders
            fill_price = self.slippage_model.calculate_slippage(
                price=fill_price,
                size=order.size,
                side=order.side,
                volume=volume
            )

            # Calculate trade value and fee
            trade_value = order.size * fill_price
            fee = self.fee_model.calculate_fee(trade_value, order.order_type)

            # Calculate slippage
            slippage = abs(fill_price - order.limit_price)

            # Update order
            order.status = OrderStatus.FILLED
            order.filled_size = order.size
            order.avg_fill_price = fill_price

            # Remove from open orders if present
            if order in self.open_orders:
                self.open_orders.remove(order)

            # Create trade
            trade = Trade(
                trade_id=self._generate_trade_id(),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                fill_price=fill_price,
                slippage=slippage,
                fee=fee,
                timestamp=self.price_feed.get_current_timestamp()
            )

            # Update positions and cash
            self._update_position(trade)

            # Record trade
            self.trade_history.append(trade)

            logger.info(
                f"Executed limit order: {order.order_id} {order.side.value} "
                f"{order.size} {order.symbol} @ ${fill_price:.2f} "
                f"(fee: ${fee:.4f})"
            )

            return {
                'order_id': order.order_id,
                'status': 'filled',
                'filled_size': order.size,
                'avg_fill_price': fill_price,
                'fee': fee,
                'message': 'Limit order filled'
            }

        except Exception as e:
            logger.error(f"Failed to execute limit order: {e}")
            order.status = OrderStatus.REJECTED
            if order in self.open_orders:
                self.open_orders.remove(order)
            return {
                'order_id': order.order_id,
                'status': 'rejected',
                'filled_size': 0.0,
                'avg_fill_price': 0.0,
                'fee': 0.0,
                'message': f'Order rejected: {str(e)}'
            }

    def _check_limit_orders(self):
        """Check and execute any limit orders that can be filled."""
        orders_to_remove = []

        for order in self.open_orders[:]:  # Copy list to avoid modification during iteration
            try:
                current_price = self.price_feed.get_current_price(order.symbol)

                should_fill = False
                if order.side == OrderSide.BUY and current_price <= order.limit_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                    should_fill = True

                if should_fill:
                    result = self._execute_limit_order(order, current_price)
                    if result['status'] == 'filled':
                        orders_to_remove.append(order)

            except Exception as e:
                logger.error(f"Error checking limit order {order.order_id}: {e}")

        # Remove filled orders
        for order in orders_to_remove:
            if order in self.open_orders:
                self.open_orders.remove(order)

    def _update_position(self, trade: Trade):
        """Update position and cash based on trade."""
        symbol = trade.symbol
        position = self.positions.get(symbol)

        if trade.side == OrderSide.BUY:
            # Deduct cash (trade value + fee)
            total_cost = (trade.size * trade.fill_price) + trade.fee
            self.cash -= total_cost

            # Update or create position
            if position is None:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    size=trade.size,
                    avg_entry_price=trade.fill_price,
                    entry_timestamp=trade.timestamp,
                    last_update=trade.timestamp
                )
            else:
                # Add to existing position
                total_cost_existing = position.size * position.avg_entry_price
                total_cost_new = trade.size * trade.fill_price
                new_size = position.size + trade.size
                position.avg_entry_price = (total_cost_existing + total_cost_new) / new_size
                position.size = new_size
                position.last_update = trade.timestamp

        else:  # SELL
            # Add cash (trade value - fee)
            proceeds = (trade.size * trade.fill_price) - trade.fee
            self.cash += proceeds

            if position is None:
                logger.error(f"No position to sell for {symbol}")
                raise InsufficientFundsError(f"No position to sell for {symbol}")

            # Calculate realized PnL
            pnl = (trade.fill_price - position.avg_entry_price) * trade.size
            trade.pnl = pnl
            self.realized_pnl += pnl

            # Update position
            position.size -= trade.size
            position.last_update = trade.timestamp

            # Remove position if fully closed
            if position.size <= 0.001:  # Small epsilon for floating point
                del self.positions[symbol]

    def _record_equity_curve_point(self):
        """Record current equity curve point."""
        try:
            point = EquityCurvePoint(
                timestamp=self.price_feed.get_current_timestamp(),
                portfolio_value=self.get_portfolio_value(),
                cash=self.cash,
                unrealized_pnl=self.get_pnl()['unrealized_pnl'],
                realized_pnl=self.realized_pnl
            )
            self.equity_curve.append(point)
        except Exception as e:
            logger.error(f"Error recording equity curve point: {e}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD-{self._order_counter:06d}"

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return f"TRD-{self._trade_counter:06d}"
