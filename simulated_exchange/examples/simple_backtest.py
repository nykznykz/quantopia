"""Simple backtest example - Buy and hold strategy."""

import sys
sys.path.insert(0, '../src')

from datetime import datetime
from simulated_exchange import (
    SimulatedExchange,
    HistoricalPriceFeed,
    download_data,
    setup_logging
)

# Setup logging
setup_logging()


def main():
    """Run a simple buy-and-hold backtest."""
    print("=" * 60)
    print("Simple Backtest Example - Buy and Hold Strategy")
    print("=" * 60)

    # Step 1: Download historical data
    print("\n1. Downloading historical data...")
    df = download_data(
        symbols=['BTC-USD'],
        exchange='binance',
        timeframe='1h',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        output_file='../data/btc_sample.csv'
    )
    print(f"   Downloaded {len(df)} bars")

    # Step 2: Create price feed
    print("\n2. Initializing price feed...")
    feed = HistoricalPriceFeed(
        data_source='../data/btc_sample.csv',
        symbols=['BTC-USD'],
        timeframe='1h'
    )
    start_date, end_date = feed.get_date_range()
    print(f"   Data range: {start_date} to {end_date}")

    # Step 3: Create exchange
    print("\n3. Initializing simulated exchange...")
    exchange = SimulatedExchange(
        price_feed=feed,
        initial_capital=10000.0,
        mode='backtest'
    )
    print(f"   Initial capital: ${exchange.initial_capital:,.2f}")

    # Step 4: Run backtest
    print("\n4. Running backtest...")
    position_opened = False
    bar_count = 0

    while feed.has_next():
        # Get current price
        current_price = feed.get_current_price('BTC-USD')
        bar_count += 1

        # Simple strategy: Buy on first bar and hold
        if not position_opened:
            # Calculate position size (invest 90% of capital)
            invest_amount = exchange.cash * 0.9
            size = invest_amount / current_price

            # Place market buy order
            result = exchange.place_order(
                symbol='BTC-USD',
                side='buy',
                size=size,
                order_type='market'
            )

            if result['status'] == 'filled':
                print(f"   📈 Bought {size:.4f} BTC @ ${result['avg_fill_price']:,.2f}")
                print(f"      Fee: ${result['fee']:.4f}")
                position_opened = True

        # Update exchange state
        exchange.update()

        # Advance to next bar
        feed.next_bar()

    print(f"   Processed {bar_count} bars")

    # Step 5: Close position (sell everything)
    print("\n5. Closing position...")
    position = exchange.get_position('BTC-USD')
    if position:
        result = exchange.place_order(
            symbol='BTC-USD',
            side='sell',
            size=position['size'],
            order_type='market'
        )
        if result['status'] == 'filled':
            print(f"   📉 Sold {position['size']:.4f} BTC @ ${result['avg_fill_price']:,.2f}")
            print(f"      Fee: ${result['fee']:.4f}")

    # Step 6: Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    # Account info
    account = exchange.get_account_info()
    print(f"\nAccount Summary:")
    print(f"  Final Portfolio Value: ${account['portfolio_value']:,.2f}")
    print(f"  Final Cash:           ${account['cash']:,.2f}")

    # PnL
    pnl = exchange.get_pnl()
    print(f"\nProfit & Loss:")
    print(f"  Realized PnL:     ${pnl['realized_pnl']:,.2f}")
    print(f"  Total PnL:        ${pnl['total_pnl']:,.2f}")
    print(f"  Total Return:     {pnl['total_pnl_pct']:.2f}%")
    print(f"  Total Fees:       ${pnl['total_fees']:.4f}")

    # Performance metrics
    metrics = exchange.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total Return:     {metrics['total_return']:.2f}%")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:.2f}%")
    print(f"  Total Trades:     {metrics['total_trades']}")
    print(f"  Win Rate:         {metrics['win_rate']:.2f}%")

    # Step 7: Export data
    print("\n6. Exporting results...")
    equity_df = exchange.get_equity_curve()
    equity_df.to_csv('../data/equity_curve.csv', index=False)
    print("   ✓ Saved equity curve to ../data/equity_curve.csv")

    trades_df = exchange.get_trade_history()
    if not trades_df.empty:
        trades_df.to_csv('../data/trade_history.csv', index=False)
        print("   ✓ Saved trade history to ../data/trade_history.csv")

    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
