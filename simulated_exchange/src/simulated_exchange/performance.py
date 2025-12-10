"""Performance metrics calculation for trading strategies."""

from typing import List, Dict
import pandas as pd
import numpy as np
from datetime import timedelta
import logging

from .models import Trade, EquityCurvePoint

logger = logging.getLogger(__name__)


def calculate_performance_metrics(
    trade_history: List[Trade],
    equity_curve: List[EquityCurvePoint],
    initial_capital: float
) -> Dict:
    """Calculate comprehensive performance metrics.

    Args:
        trade_history: List of completed trades
        equity_curve: List of equity curve points
        initial_capital: Starting capital

    Returns:
        Dict with performance metrics
    """
    if not equity_curve:
        logger.warning("Empty equity curve, returning default metrics")
        return _default_metrics()

    # Convert to DataFrames for easier calculation
    equity_df = pd.DataFrame([e.to_dict() for e in equity_curve])
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

    if not trade_history:
        logger.warning("No trades executed")
        return _calculate_metrics_no_trades(equity_df, initial_capital)

    trades_df = pd.DataFrame([t.to_dict() for t in trade_history])
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

    # Calculate metrics
    metrics = {}

    # Basic returns
    final_value = equity_df['portfolio_value'].iloc[-1]
    total_return_usd = final_value - initial_capital
    total_return_pct = (total_return_usd / initial_capital) * 100

    metrics['total_return'] = total_return_pct
    metrics['total_return_usd'] = total_return_usd
    metrics['final_portfolio_value'] = final_value

    # Sharpe ratio
    metrics['sharpe_ratio'] = _calculate_sharpe_ratio(equity_df)

    # Drawdown metrics
    dd_metrics = _calculate_drawdown_metrics(equity_df)
    metrics.update(dd_metrics)

    # Trade metrics
    trade_metrics = _calculate_trade_metrics(trades_df)
    metrics.update(trade_metrics)

    # Fees
    metrics['total_fees'] = trades_df['fee'].sum()

    # Time metrics
    time_metrics = _calculate_time_metrics(equity_df, trades_df)
    metrics.update(time_metrics)

    return metrics


def get_equity_curve_df(equity_curve: List[EquityCurvePoint]) -> pd.DataFrame:
    """Convert equity curve to DataFrame.

    Args:
        equity_curve: List of equity curve points

    Returns:
        DataFrame with equity curve data
    """
    if not equity_curve:
        return pd.DataFrame()

    df = pd.DataFrame([e.to_dict() for e in equity_curve])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_trade_history_df(trade_history: List[Trade]) -> pd.DataFrame:
    """Convert trade history to DataFrame.

    Args:
        trade_history: List of trades

    Returns:
        DataFrame with trade data
    """
    if not trade_history:
        return pd.DataFrame()

    df = pd.DataFrame([t.to_dict() for t in trade_history])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# Private helper functions

def _default_metrics() -> Dict:
    """Return default metrics when no data available."""
    return {
        'total_return': 0.0,
        'total_return_usd': 0.0,
        'final_portfolio_value': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'max_drawdown_duration': 0,
        'win_rate': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'profit_factor': 0.0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_fees': 0.0,
        'avg_holding_period': 0.0,
        'total_days': 0
    }


def _calculate_metrics_no_trades(equity_df: pd.DataFrame, initial_capital: float) -> Dict:
    """Calculate metrics when no trades were executed."""
    metrics = _default_metrics()

    final_value = equity_df['portfolio_value'].iloc[-1]
    total_return_usd = final_value - initial_capital
    total_return_pct = (total_return_usd / initial_capital) * 100

    metrics['total_return'] = total_return_pct
    metrics['total_return_usd'] = total_return_usd
    metrics['final_portfolio_value'] = final_value
    metrics['sharpe_ratio'] = _calculate_sharpe_ratio(equity_df)

    dd_metrics = _calculate_drawdown_metrics(equity_df)
    metrics.update(dd_metrics)

    # Calculate time period
    if len(equity_df) > 1:
        time_delta = equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]
        metrics['total_days'] = time_delta.days

    return metrics


def _calculate_sharpe_ratio(equity_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio.

    Args:
        equity_df: Equity curve DataFrame
        risk_free_rate: Annual risk-free rate (default 0%)

    Returns:
        Sharpe ratio (annualized)
    """
    if len(equity_df) < 2:
        return 0.0

    # Calculate returns
    returns = equity_df['portfolio_value'].pct_change().dropna()

    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Calculate average return and std dev
    avg_return = returns.mean()
    std_return = returns.std()

    # Annualize (assuming daily data, adjust if needed)
    # For backtesting, we might have hourly data, so we need to estimate the period
    time_delta = equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]
    total_days = max(time_delta.days, 1)
    periods_per_day = len(equity_df) / total_days if total_days > 0 else 1

    # Annualization factor
    periods_per_year = 252 * periods_per_day  # 252 trading days per year

    sharpe = (avg_return * np.sqrt(periods_per_year)) / std_return

    return float(sharpe)


def _calculate_drawdown_metrics(equity_df: pd.DataFrame) -> Dict:
    """Calculate drawdown metrics.

    Args:
        equity_df: Equity curve DataFrame

    Returns:
        Dict with max_drawdown and max_drawdown_duration
    """
    if len(equity_df) < 2:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_duration': 0
        }

    # Calculate running maximum
    equity_df['running_max'] = equity_df['portfolio_value'].expanding().max()

    # Calculate drawdown
    equity_df['drawdown'] = (
        (equity_df['portfolio_value'] - equity_df['running_max']) /
        equity_df['running_max'] * 100
    )

    max_drawdown = abs(equity_df['drawdown'].min())

    # Calculate drawdown duration
    equity_df['is_drawdown'] = equity_df['drawdown'] < 0
    equity_df['drawdown_group'] = (
        equity_df['is_drawdown'] != equity_df['is_drawdown'].shift()
    ).cumsum()

    drawdown_periods = equity_df[equity_df['is_drawdown']].groupby('drawdown_group')

    max_duration = 0
    for _, group in drawdown_periods:
        if len(group) > 0:
            duration = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).days
            max_duration = max(max_duration, duration)

    return {
        'max_drawdown': float(max_drawdown),
        'max_drawdown_duration': int(max_duration)
    }


def _calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict:
    """Calculate trade-based metrics.

    Args:
        trades_df: Trade history DataFrame

    Returns:
        Dict with trade metrics
    """
    total_trades = len(trades_df)

    if total_trades == 0:
        return {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }

    # Filter trades with PnL (closing trades)
    closed_trades = trades_df[trades_df['pnl'] != 0].copy()

    if len(closed_trades) == 0:
        return {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'total_trades': total_trades,
            'winning_trades': 0,
            'losing_trades': 0
        }

    # Separate wins and losses
    winning_trades = closed_trades[closed_trades['pnl'] > 0]
    losing_trades = closed_trades[closed_trades['pnl'] < 0]

    num_wins = len(winning_trades)
    num_losses = len(losing_trades)
    total_closed = len(closed_trades)

    # Win rate
    win_rate = (num_wins / total_closed * 100) if total_closed > 0 else 0.0

    # Average win/loss
    avg_win = winning_trades['pnl'].mean() if num_wins > 0 else 0.0
    avg_loss = abs(losing_trades['pnl'].mean()) if num_losses > 0 else 0.0

    # Profit factor
    gross_profit = winning_trades['pnl'].sum() if num_wins > 0 else 0.0
    gross_loss = abs(losing_trades['pnl'].sum()) if num_losses > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return {
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'total_trades': int(total_trades),
        'winning_trades': int(num_wins),
        'losing_trades': int(num_losses)
    }


def _calculate_time_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
    """Calculate time-based metrics.

    Args:
        equity_df: Equity curve DataFrame
        trades_df: Trade history DataFrame

    Returns:
        Dict with time metrics
    """
    metrics = {}

    # Total time period
    if len(equity_df) > 1:
        time_delta = equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]
        metrics['total_days'] = time_delta.days
    else:
        metrics['total_days'] = 0

    # Average holding period (rough estimate based on trade frequency)
    if len(trades_df) > 1:
        # Group trades into pairs (buy/sell)
        # For simplicity, estimate holding period from trade frequency
        time_span = trades_df['timestamp'].iloc[-1] - trades_df['timestamp'].iloc[0]
        num_trade_pairs = len(trades_df) / 2  # Rough estimate

        if num_trade_pairs > 0:
            avg_holding = time_span / num_trade_pairs
            metrics['avg_holding_period'] = avg_holding.total_seconds() / 3600  # Hours
        else:
            metrics['avg_holding_period'] = 0.0
    else:
        metrics['avg_holding_period'] = 0.0

    return metrics
