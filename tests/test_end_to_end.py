"""End-to-end test for strategy generation, backtesting, and filtering pipeline."""

import sys
import os
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indicators import INDICATOR_REGISTRY
from src.strategy_generation import StrategyGenerator, LLMClient
from src.code_generation.code_generator import CodeGenerator, save_strategy_code
from src.backtest import BacktestRunner
from src.critique import StrategyFilter, FilterCriteria

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_full_pipeline():
    """Test the complete pipeline: Generate → Code → Backtest → Filter."""

    logger.info("=" * 80)
    logger.info("Starting End-to-End Pipeline Test")
    logger.info("=" * 80)

    # Configuration
    symbol = "BTC-USD"
    exchange = "binance"
    timeframe = "1h"
    start_date = datetime.now() - timedelta(days=180)  # 6 months of data
    end_date = datetime.now()

    # Step 1: Initialize LLM Strategy Generator
    logger.info("\n[STEP 1] Initializing LLM Strategy Generator...")

    try:
        llm_client = LLMClient(
            provider="openai",  # Or "azure", "deepseek"
            model="gpt-4",
            temperature=0.8
        )

        available_indicators = list(INDICATOR_REGISTRY.keys())
        generator = StrategyGenerator(
            llm_client=llm_client,
            available_indicators=available_indicators,
            max_retries=3
        )

        logger.info(f"✓ Generator initialized with {len(available_indicators)} indicators")

    except Exception as e:
        logger.error(f"✗ Failed to initialize LLM client: {e}")
        logger.info("Skipping LLM generation steps (API key may not be configured)")
        return

    # Step 2: Generate Strategy with LLM
    logger.info("\n[STEP 2] Generating strategy with LLM...")

    try:
        strategy_metadata = generator.generate_strategy(
            strategy_type="mean_reversion",  # Can be None for random
            temperature=0.8
        )

        logger.info(f"✓ Generated strategy: {strategy_metadata['strategy_name']}")
        logger.info(f"  Type: {strategy_metadata['strategy_type']}")
        logger.info(f"  Hypothesis: {strategy_metadata['hypothesis'][:100]}...")
        logger.info(f"  Indicators: {[ind['name'] for ind in strategy_metadata['indicators']]}")

    except Exception as e:
        logger.error(f"✗ Strategy generation failed: {e}")
        return

    # Step 3: Convert to Executable Code
    logger.info("\n[STEP 3] Converting strategy to Python code...")

    try:
        code_gen = CodeGenerator()
        strategy_code = code_gen.generate_strategy_class(strategy_metadata)

        # Save generated code
        os.makedirs("strategies/generated", exist_ok=True)
        code_filepath = f"strategies/generated/{strategy_metadata['strategy_name'].replace(' ', '_')}.py"
        save_strategy_code(strategy_code, code_filepath)

        logger.info(f"✓ Generated {len(strategy_code.split(chr(10)))} lines of code")
        logger.info(f"  Saved to: {code_filepath}")

    except Exception as e:
        logger.error(f"✗ Code generation failed: {e}")
        return

    # Step 4: Load Historical Data
    logger.info("\n[STEP 4] Loading historical data...")

    try:
        runner = BacktestRunner(
            initial_capital=10000.0,
            slippage_bps=5.0,
            maker_fee=0.0,
            taker_fee=0.00025
        )

        data = runner.load_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        logger.info(f"✓ Loaded {len(data)} candles ({data.index[0]} to {data.index[-1]})")

    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        logger.info("This may occur if simulated_exchange is not properly installed")
        return

    # Step 5: Run Backtest
    logger.info("\n[STEP 5] Running backtest...")

    try:
        backtest_results = runner.run_from_code(
            strategy_code=strategy_code,
            data=data,
            symbol=symbol,
            strategy_name=strategy_metadata['strategy_name']
        )

        if backtest_results.get('status') == 'success':
            metrics = backtest_results['metrics']
            logger.info(f"✓ Backtest completed successfully")
            logger.info(f"  Final Capital: ${backtest_results['final_capital']:.2f}")
            logger.info(f"  Total Return: {metrics['total_return']*100:.2f}%")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            logger.info(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
            logger.info(f"  Total Trades: {metrics['num_trades']}")
        else:
            logger.error(f"✗ Backtest failed: {backtest_results.get('error')}")
            return

    except Exception as e:
        logger.error(f"✗ Backtest execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # Step 6: Apply Strategy Filter
    logger.info("\n[STEP 6] Applying strategy filter...")

    try:
        criteria = FilterCriteria(
            min_total_return=0.05,  # 5%
            min_sharpe_ratio=0.5,
            max_drawdown=0.30,  # 30%
            min_num_trades=10,
            min_win_rate=0.30  # 30%
        )

        strategy_filter = StrategyFilter(criteria=criteria)
        filter_result = strategy_filter.filter_strategy(backtest_results)

        logger.info(f"\n{filter_result}")

        if filter_result.passed:
            logger.info(f"\n✓✓✓ STRATEGY APPROVED ({filter_result.classification}) ✓✓✓")

            # Save approved strategy
            os.makedirs("strategies/approved", exist_ok=True)
            approved_path = f"strategies/approved/{strategy_metadata['strategy_name'].replace(' ', '_')}.py"
            save_strategy_code(strategy_code, approved_path)
            logger.info(f"Saved approved strategy to: {approved_path}")

            # Save backtest results
            runner.save_results(backtest_results, output_dir="results/backtests")

        else:
            logger.info(f"\n✗✗✗ STRATEGY REJECTED ✗✗✗")
            logger.info(f"Rejection reasons:")
            for reason in filter_result.rejection_reasons:
                logger.info(f"  - {reason}")

    except Exception as e:
        logger.error(f"✗ Filtering failed: {e}")
        return

    # Step 7: Summary
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Test Complete")
    logger.info("=" * 80)
    logger.info(f"Strategy: {strategy_metadata['strategy_name']}")
    logger.info(f"Status: {'APPROVED' if filter_result.passed else 'REJECTED'}")
    logger.info(f"Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max DD: {metrics['max_drawdown']*100:.2f}%")
    logger.info("=" * 80)


def test_batch_pipeline():
    """Test batch processing: Generate multiple strategies → Backtest → Filter."""

    logger.info("=" * 80)
    logger.info("Starting Batch Pipeline Test")
    logger.info("=" * 80)

    num_strategies = 3  # Generate 3 strategies for testing

    # Configuration
    symbol = "BTC-USD"
    exchange = "binance"
    timeframe = "1h"
    start_date = datetime.now() - timedelta(days=90)  # 3 months
    end_date = datetime.now()

    # Step 1: Generate Multiple Strategies
    logger.info(f"\n[STEP 1] Generating {num_strategies} strategies...")

    try:
        llm_client = LLMClient(provider="openai", model="gpt-4", temperature=0.8)
        available_indicators = list(INDICATOR_REGISTRY.keys())
        generator = StrategyGenerator(llm_client, available_indicators, max_retries=3)

        strategies = generator.generate_batch(
            num_strategies=num_strategies,
            temperature=0.8
        )

        logger.info(f"✓ Generated {len(strategies)} strategies")

    except Exception as e:
        logger.error(f"✗ Batch generation failed: {e}")
        return

    # Step 2: Convert All to Code
    logger.info("\n[STEP 2] Converting strategies to code...")

    code_gen = CodeGenerator()
    strategy_codes = []

    for strategy_metadata in strategies:
        try:
            code = code_gen.generate_strategy_class(strategy_metadata)
            strategy_codes.append(code)
            logger.info(f"✓ Generated code for: {strategy_metadata['strategy_name']}")
        except Exception as e:
            logger.error(f"✗ Code generation failed for {strategy_metadata['strategy_name']}: {e}")

    # Step 3: Load Data (once for all strategies)
    logger.info("\n[STEP 3] Loading historical data...")

    try:
        runner = BacktestRunner(initial_capital=10000.0)
        data = runner.load_data(symbol, exchange, timeframe, start_date, end_date)
        logger.info(f"✓ Loaded {len(data)} candles")
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return

    # Step 4: Batch Backtest
    logger.info("\n[STEP 4] Running batch backtest...")

    from src.backtest import BatchTester

    batch_tester = BatchTester(initial_capital=10000.0)

    try:
        batch_results = batch_tester.run_batch(
            strategy_codes=strategy_codes,
            strategy_names=[s['strategy_name'] for s in strategies],
            data=data,
            symbol=symbol,
            parallel=False  # Sequential for testing
        )

        logger.info(f"✓ Completed backtests for {len(batch_results)} strategies")

    except Exception as e:
        logger.error(f"✗ Batch backtest failed: {e}")
        return

    # Step 5: Batch Filter
    logger.info("\n[STEP 5] Filtering batch results...")

    strategy_filter = StrategyFilter()

    try:
        filter_results = strategy_filter.filter_batch(batch_results)

        approved = [r for r in filter_results if r.passed]
        logger.info(f"\n✓ {len(approved)}/{len(filter_results)} strategies approved")

        # Show all results
        for result in filter_results:
            logger.info(f"\n{result}")

        # Analyze rejection patterns
        analysis = strategy_filter.analyze_rejection_patterns(filter_results)
        logger.info(f"\nRejection Analysis:")
        logger.info(f"  Rejection rate: {analysis['rejection_rate']*100:.1f}%")
        logger.info(f"  Most common failure: {analysis['most_common_failure']}")

    except Exception as e:
        logger.error(f"✗ Filtering failed: {e}")
        return

    logger.info("\n" + "=" * 80)
    logger.info("Batch Pipeline Test Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Run single strategy test
    test_full_pipeline()

    # Uncomment to run batch test
    # test_batch_pipeline()
