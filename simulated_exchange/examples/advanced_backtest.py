"""Advanced backtest example - Moving average crossover strategy."""

import sys
sys.path.insert(0, '../src')

from datetime import datetime
import pandas as pd
from simulated_exchange import (
    SimulatedExchange,
    HistoricalPriceFeed,
    download_data,
    Config,
    setup_logging
)

# Setup logging
setup_logging()


def calculate_moving_averages(prices, short_window=20, long_window=50):
    """Calculate short and long moving averages."""
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    return short_ma, long_ma


def main():
    """Run an advanced backtest with moving average crossover strategy."""
    print("=" * 60)
    print("Advanced Backtest - Moving Average Crossover")
    print("=" * 60)

    # Configuration
    config = Config.from_dict({
        'exchange': {
            'initial_capital': 10000.0,
            'mode': 'backtest'
        },
        'slippage': {
            'model': 'fixed',
            'fixed_bps': 5
        },
        'fees': {
            'model': 'tiered',
            'maker_fee': 0.0000,
            'taker_fee': 0.00025
        }
    })

    # Step 1: Download data
    print("\n1. Downloading historical data...")
    df = download_data(
        symbols=['BTC-USD', 'ETH-USD'],
        exchange='binance',
        timeframe='1h',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
        output_file='../data/crypto_sample.csv'
    )
    print(f"   Downloaded {len(df)} bars for {df['symbol'].nunique()} symbols")

    # Step 2: Create price feed and exchange
    print("\n2. Initializing exchange...")
    feed = HistoricalPriceFeed(
        data_source='../data/crypto_sample.csv',
        symbols=['BTC-USD', 'ETH-USD'],
        timeframe='1h'
    )

    exchange = SimulatedExchange(
        price_feed=feed,
        initial_capital=config.get('exchange.initial_capital'),
        mode='backtest',
        slippage_config=config.get('slippage'),
        fee_config=config.get('fees')
    )

    # Step 3: Prepare price history for indicators
    print("\n3. Preparing price data...")
    btc_prices = []
    eth_prices = []

    # Step 4: Run backtest
    print("\n4. Running backtest...")
    bar_count = 0
    signals = []

    while feed.has_next():
        # Get current prices
        btc_price = feed.get_current_price('BTC-USD')
        eth_price = feed.get_current_price('ETH-USD')

        btc_prices.append(btc_price)
        eth_prices.append(eth_price)

        # Need enough data for long MA
        if len(btc_prices) >= 50:
            # Calculate MAs for BTC
            btc_series = pd.Series(btc_prices)
            btc_short_ma, btc_long_ma = calculate_moving_averages(btc_series, 20, 50)

            # Get latest MA values
            btc_short = btc_short_ma.iloc[-1]
            btc_long = btc_long_ma.iloc[-1]
            btc_short_prev = btc_short_ma.iloc[-2] if len(btc_short_ma) > 1 else btc_short
            btc_long_prev = btc_long_ma.iloc[-2] if len(btc_long_ma) > 1 else btc_long

            # Check for crossover signals
            btc_position = exchange.get_position('BTC-USD')

            # Bullish crossover (short MA crosses above long MA)
            if btc_short > btc_long and btc_short_prev <= btc_long_prev:
                if not btc_position:
                    # Buy signal
                    size = (exchange.cash * 0.45) / btc_price  # Invest 45% of cash
                    result = exchange.place_order('BTC-USD', 'buy', size, 'market')
                    if result['status'] == 'filled':
                        signal = f"BUY BTC @ ${result['avg_fill_price']:,.2f}"
                        signals.append(signal)
                        if len(signals) <= 5:  # Print first 5 signals
                            print(f"   📈 {signal}")

            # Bearish crossover (short MA crosses below long MA)
            elif btc_short < btc_long and btc_short_prev >= btc_long_prev:
                if btc_position:
                    # Sell signal
                    result = exchange.place_order('BTC-USD', 'sell', btc_position['size'], 'market')
                    if result['status'] == 'filled':
                        signal = f"SELL BTC @ ${result['avg_fill_price']:,.2f}"
                        signals.append(signal)
                        if len(signals) <= 5:
                            print(f"   📉 {signal}")

        # Update exchange and advance
        exchange.update()
        feed.next_bar()
        bar_count += 1

    print(f"   Processed {bar_count} bars, generated {len(signals)} signals")

    # Step 5: Close all positions
    print("\n5. Closing all positions...")
    for symbol in ['BTC-USD', 'ETH-USD']:
        position = exchange.get_position(symbol)
        if position:
            exchange.place_order(symbol, 'sell', position['size'], 'market')
            print(f"   Closed {symbol} position")

    # Step 6: Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    account = exchange.get_account_info()
    pnl = exchange.get_pnl()
    metrics = exchange.get_performance_metrics()

    print(f"\n💰 Final Portfolio Value: ${account['portfolio_value']:,.2f}")
    print(f"   Initial Capital:       ${exchange.initial_capital:,.2f}")
    print(f"   Total Return:          {metrics['total_return']:.2f}%")
    print(f"   Total PnL:             ${pnl['total_pnl']:,.2f}")

    print(f"\n📊 Performance Metrics:")
    print(f"   Sharpe Ratio:          {metrics['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:          {metrics['max_drawdown']:.2f}%")
    print(f"   Total Trades:          {metrics['total_trades']}")
    print(f"   Win Rate:              {metrics['win_rate']:.2f}%")
    print(f"   Profit Factor:         {metrics['profit_factor']:.2f}")
    print(f"   Avg Win:               ${metrics['avg_win']:.2f}")
    print(f"   Avg Loss:              ${metrics['avg_loss']:.2f}")
    print(f"   Total Fees:            ${metrics['total_fees']:.4f}")

    # Step 7: Export results
    print("\n6. Exporting results...")
    exchange.get_equity_curve().to_csv('../data/equity_curve_advanced.csv', index=False)
    trades_df = exchange.get_trade_history()
    if not trades_df.empty:
        trades_df.to_csv('../data/trade_history_advanced.csv', index=False)
    print("   ✓ Results exported")

    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
