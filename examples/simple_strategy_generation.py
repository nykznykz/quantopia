"""Simple example: Generate a single strategy and backtest it."""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indicators import INDICATOR_REGISTRY
from src.strategy_generation import StrategyGenerator, LLMClient
from src.code_generation.code_generator import CodeGenerator, save_strategy_code
from src.backtest import BacktestRunner
from src.critique import StrategyFilter, FilterCriteria


def main():
    """Generate and backtest a single strategy."""

    print("=" * 80)
    print("OmniAlpha - Simple Strategy Generation Example")
    print("Using LLM for both strategy generation AND code generation")
    print("=" * 80)

    # Step 1: Initialize LLM client
    print("\n[1/6] Initializing LLM client...")
    llm_client = LLMClient(
        provider="openai",  # Options: "openai", "azure", "deepseek", "anthropic"
        model="gpt-4",  # For code generation: "gpt-4", "claude-sonnet-4", or "deepseek-coder"
        temperature=0.8  # For strategy generation (will use 0.2 for code generation)
    )
    print("✓ LLM client initialized")
    print("  Note: Same LLM will be used for both strategy design and code generation")

    # Step 2: Generate strategy
    print("\n[2/6] Generating strategy with LLM...")
    available_indicators = list(INDICATOR_REGISTRY.keys())
    generator = StrategyGenerator(
        llm_client=llm_client,
        available_indicators=available_indicators,
        max_retries=3
    )

    strategy_metadata = generator.generate_strategy(
        strategy_type="mean_reversion",  # Or None for random type
        temperature=0.8
    )

    print(f"✓ Generated: {strategy_metadata['strategy_name']}")
    print(f"  Type: {strategy_metadata['strategy_type']}")
    print(f"  Indicators: {[ind['name'] for ind in strategy_metadata['indicators']]}")

    # Step 3: Convert to code using LLM
    print("\n[3/6] Converting to Python code with LLM...")
    code_gen = CodeGenerator(
        llm_client=llm_client,  # Reuse the same client
        temperature=0.2  # Lower temperature for more deterministic code
    )
    strategy_code = code_gen.generate_strategy_class(strategy_metadata)

    # Save code
    os.makedirs("strategies/generated", exist_ok=True)
    code_filepath = f"strategies/generated/{strategy_metadata['strategy_name'].replace(' ', '_')}.py"
    save_strategy_code(strategy_code, code_filepath)
    print(f"✓ Code saved to: {code_filepath}")

    # Step 4: Load historical data
    print("\n[4/6] Loading historical data...")
    runner = BacktestRunner(initial_capital=10000.0)

    data = runner.load_data(
        symbol="BTC-USD",
        exchange="binance",
        timeframe="1h",
        start_date=datetime.now() - timedelta(days=180),  # 6 months
        end_date=datetime.now()
    )
    print(f"✓ Loaded {len(data)} candles")

    # Step 5: Run backtest
    print("\n[5/6] Running backtest...")
    results = runner.run_from_code(
        strategy_code=strategy_code,
        data=data,
        symbol="BTC-USD",
        strategy_name=strategy_metadata['strategy_name']
    )

    if results['status'] == 'success':
        metrics = results['metrics']
        print(f"✓ Backtest completed")
        print(f"  Return: {metrics['total_return']*100:.2f}%")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Trades: {metrics['num_trades']}")
    else:
        print(f"✗ Backtest failed: {results.get('error')}")
        return

    # Step 6: Filter strategy
    print("\n[6/6] Applying filter...")
    strategy_filter = StrategyFilter(
        criteria=FilterCriteria(
            min_total_return=0.05,  # 5%
            min_sharpe_ratio=0.5,
            max_drawdown=0.30,  # 30%
            min_num_trades=10
        )
    )

    filter_result = strategy_filter.filter_strategy(results)

    print(f"\n{filter_result}")

    if filter_result.passed:
        print(f"\n✓✓✓ STRATEGY APPROVED ✓✓✓")

        # Save approved strategy
        os.makedirs("strategies/approved", exist_ok=True)
        approved_path = f"strategies/approved/{strategy_metadata['strategy_name'].replace(' ', '_')}.py"
        save_strategy_code(strategy_code, approved_path)
        print(f"Saved to: {approved_path}")
    else:
        print(f"\n✗✗✗ STRATEGY REJECTED ✗✗✗")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
