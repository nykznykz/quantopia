"""Paper trading example - Live trading with simulated execution."""

import sys
sys.path.insert(0, '../src')

import time
from datetime import datetime
from simulated_exchange import (
    SimulatedExchange,
    LivePriceFeed,
    setup_logging
)

# Setup logging
setup_logging()


def simple_trading_logic(exchange, symbol):
    """Simple trading logic - example for demonstration.

    In real use, you would integrate your LLM or other trading strategy here.
    """
    position = exchange.get_position(symbol)
    current_price = exchange.price_feed.get_current_price(symbol)

    # Example: If no position and price is "low" (simplified), buy
    # In reality, you'd use your LLM or other signals
    if not position:
        # Check if we have enough cash
        if exchange.cash > current_price * 0.01:  # Can buy at least 0.01 BTC
            size = (exchange.cash * 0.5) / current_price  # Invest 50% of cash
            result = exchange.place_order(symbol, 'buy', size, 'market')
            if result['status'] == 'filled':
                print(f"   ✅ BOUGHT {size:.6f} {symbol} @ ${result['avg_fill_price']:,.2f}")
                return True

    # Example: If we have a position and made 2% profit, sell
    elif position:
        unrealized_pnl_pct = position['unrealized_pnl_pct']
        if unrealized_pnl_pct >= 2.0:  # 2% profit target
            result = exchange.place_order(symbol, 'sell', position['size'], 'market')
            if result['status'] == 'filled':
                print(f"   ✅ SOLD {position['size']:.6f} {symbol} @ ${result['avg_fill_price']:,.2f}")
                print(f"      💰 Profit: ${position['unrealized_pnl']:.2f} ({unrealized_pnl_pct:.2f}%)")
                return True

    return False


def main():
    """Run paper trading with live price feed."""
    print("=" * 60)
    print("Paper Trading Example")
    print("=" * 60)
    print("\n⚠️  This is PAPER TRADING (simulated) - No real money involved!")
    print("    Using LIVE price data for realistic simulation")
    print("    Press Ctrl+C to stop\n")

    # Step 1: Create live price feed
    print("1. Connecting to live price feed...")
    try:
        feed = LivePriceFeed(
            exchange='binance',
            symbols=['BTC-USD'],
            testnet=False  # Using production WebSocket for read-only price data
        )
        feed.connect()
        time.sleep(3)  # Wait for initial connection

        if not feed.is_connected():
            print("   ❌ Failed to connect to price feed")
            return

        print("   ✓ Connected to Binance (live price data)")

    except Exception as e:
        print(f"   ❌ Error connecting: {e}")
        print("\n   NOTE: Live trading requires an active internet connection")
        print("   and the exchange's WebSocket API to be accessible.")
        return

    # Step 2: Create exchange
    print("\n2. Initializing paper trading exchange...")
    exchange = SimulatedExchange(
        price_feed=feed,
        initial_capital=10000.0,
        mode='live'
    )
    print(f"   Initial capital: ${exchange.initial_capital:,.2f}")

    # Step 3: Run trading loop
    print("\n3. Starting paper trading...")
    print("=" * 60)

    try:
        iteration = 0
        while True:
            iteration += 1

            # Get current market data
            try:
                current_price = feed.get_current_price('BTC-USD')
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Display status every 10 iterations
                if iteration % 10 == 0:
                    account = exchange.get_account_info()
                    pnl = exchange.get_pnl()

                    print(f"\n[{timestamp}] Status Update:")
                    print(f"   BTC Price:        ${current_price:,.2f}")
                    print(f"   Portfolio Value:  ${account['portfolio_value']:,.2f}")
                    print(f"   Cash:             ${account['cash']:,.2f}")
                    print(f"   Total PnL:        ${pnl['total_pnl']:,.2f} ({pnl['total_pnl_pct']:.2f}%)")
                    print(f"   Open Positions:   {account['open_positions']}")

                # Execute trading logic
                signal = simple_trading_logic(exchange, 'BTC-USD')

                # Update exchange state
                exchange.update()

            except Exception as e:
                print(f"   ⚠️ Error in trading loop: {e}")

            # Wait before next iteration (check every 30 seconds)
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Stopping paper trading...")
        print("=" * 60)

        # Close all positions
        print("\nClosing all positions...")
        position = exchange.get_position('BTC-USD')
        if position:
            exchange.place_order('BTC-USD', 'sell', position['size'], 'market')
            print("   ✓ Closed BTC position")

        # Display final results
        print("\n" + "=" * 60)
        print("PAPER TRADING RESULTS")
        print("=" * 60)

        account = exchange.get_account_info()
        pnl = exchange.get_pnl()
        metrics = exchange.get_performance_metrics()

        print(f"\n💰 Final Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"   Initial Capital:       ${exchange.initial_capital:,.2f}")
        print(f"   Total Return:          {metrics['total_return']:.2f}%")
        print(f"   Total PnL:             ${pnl['total_pnl']:,.2f}")

        print(f"\n📊 Trading Stats:")
        print(f"   Total Trades:          {metrics['total_trades']}")
        print(f"   Win Rate:              {metrics['win_rate']:.2f}%")
        print(f"   Total Fees:            ${metrics['total_fees']:.4f}")

        # Disconnect
        feed.disconnect()
        print("\n✓ Disconnected from exchange")

        print("\n" + "=" * 60)
        print("Paper Trading Session Complete!")
        print("=" * 60)


if __name__ == '__main__':
    main()
