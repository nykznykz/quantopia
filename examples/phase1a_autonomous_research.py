"""
Phase 1a: Autonomous Research Session Example

Demonstrates the agent-first architecture where:
1. Strategy Agent autonomously decides what to explore
2. ML Quant Agent provides models on demand
3. Agent Router coordinates the pipeline
4. Code Generator creates executable strategies
5. System learns and improves over time

This is a ZERO HUMAN INTERVENTION research loop!
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import agent components
from src.agents.strategy_agent import StrategyAgent
from src.agents.ml_quant_agent import MLQuantAgent
from src.orchestrator.agent_router import AgentRouter

# Import infrastructure
from src.database.manager import StrategyDatabase
from src.ml.model_registry import ModelRegistry
from src.code_generation.code_generator import CodeGenerator
from src.strategy_generation.llm_client import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(days: int = 365) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    logger.info(f"Generating {days} days of sample data...")

    dates = pd.date_range(end=datetime.now(), periods=days * 24, freq='1H')

    # Generate realistic price data with trend + noise
    trend = np.linspace(30000, 40000, len(dates))
    noise = np.random.normal(0, 500, len(dates))
    close_prices = trend + noise + np.cumsum(np.random.normal(0, 100, len(dates)))

    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'high': close_prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'low': close_prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, len(dates))
    })

    data.set_index('timestamp', inplace=True)

    logger.info(f"Generated data shape: {data.shape}, price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    return data


def run_phase1a_session(num_strategies: int = 10):
    """
    Run Phase 1a autonomous research session.

    Args:
        num_strategies: Number of strategies to explore
    """
    logger.info("="*80)
    logger.info("PHASE 1a: AUTONOMOUS RESEARCH SESSION")
    logger.info("="*80)

    # 1. Initialize infrastructure
    logger.info("\n[1/6] Initializing infrastructure...")

    database = StrategyDatabase(db_path="data/strategies_phase1a.db")
    model_registry = ModelRegistry(registry_dir="models/registry")

    llm_client = LLMClient(
        provider="openai",
        model="gpt-4",
        temperature=0.7
    )

    logger.info("✓ Infrastructure ready")

    # 2. Initialize agents
    logger.info("\n[2/6] Initializing agents...")

    strategy_agent = StrategyAgent(
        database=database,
        model_registry=model_registry,
        llm_client=llm_client,
        exploration_rate=0.3
    )

    ml_quant_agent = MLQuantAgent(
        model_registry=model_registry,
        model_trainer=None,  # Mock for now
        llm_client=llm_client
    )

    code_generator = CodeGenerator(
        llm_client=llm_client,
        temperature=0.2  # More deterministic for code
    )

    agent_router = AgentRouter(
        code_generator=code_generator,
        ml_quant_agent=ml_quant_agent
    )

    logger.info("✓ All agents initialized")
    logger.info("  - StrategyAgent: Autonomous explorer")
    logger.info("  - MLQuantAgent: On-demand ML expert")
    logger.info("  - AgentRouter: Coordination layer")
    logger.info("  - CodeGenerator: Python code synthesis")

    # 3. Generate training data
    logger.info("\n[3/6] Preparing training data...")

    training_data = generate_sample_data(days=365)

    logger.info("✓ Training data ready")

    # 4. Run autonomous exploration loop
    logger.info(f"\n[4/6] Starting autonomous exploration ({num_strategies} strategies)...")
    logger.info("")

    results = []

    for i in range(num_strategies):
        logger.info("="*80)
        logger.info(f"ITERATION {i+1}/{num_strategies}")
        logger.info("="*80)

        try:
            # STEP 1: Strategy Agent decides what to explore
            logger.info("\nSTEP 1: Strategy Agent decision...")
            decision = strategy_agent.explore_next_strategy()

            logger.info(f"✓ Decision made:")
            logger.info(f"  Type: {decision['strategy_type']}")
            logger.info(f"  Logic: {decision['logic_type']}")
            logger.info(f"  Indicators: {decision['indicators']}")
            logger.info(f"  Rationale: {decision['rationale']}")
            logger.info(f"  Mode: {decision['exploration_mode']}")

            # STEP 2: Route through appropriate agents
            logger.info("\nSTEP 2: Routing through agents...")
            route_result = agent_router.route(
                decision=decision,
                training_data=training_data
            )

            strategy_metadata = route_result['strategy_metadata']
            code = route_result['code']
            model_id = route_result.get('model_id')

            logger.info(f"✓ Strategy generated:")
            logger.info(f"  Name: {strategy_metadata['strategy_name']}")
            logger.info(f"  Code length: {len(code)} chars, {len(code.splitlines())} lines")

            if model_id:
                logger.info(f"  ML Model: {model_id}")

            # STEP 3: Store in database (mock backtest for now)
            logger.info("\nSTEP 3: Storing results...")

            # Mock metrics
            mock_sharpe = np.random.uniform(0.5, 2.0)
            mock_return = np.random.uniform(0.05, 0.30)

            try:
                strategy_id = database.store_strategy(
                    name=strategy_metadata['strategy_name'],
                    strategy_type=strategy_metadata['strategy_type'],
                    indicators=[ind['name'] for ind in strategy_metadata['indicators']],
                    description=strategy_metadata.get('hypothesis', ''),
                    ml_strategy_type=strategy_metadata.get('ml_strategy_type', 'pure_technical'),
                    ml_models_used=strategy_metadata.get('ml_models_used', []),
                    exploration_vector={
                        'exploration_mode': decision['exploration_mode'],
                        'decided_at': decision['decided_at']
                    }
                )

                logger.info(f"✓ Stored as strategy ID: {strategy_id}")

            except Exception as e:
                logger.warning(f"Database storage failed: {e}")
                strategy_id = None

            results.append({
                'iteration': i + 1,
                'strategy_id': strategy_id,
                'strategy_name': strategy_metadata['strategy_name'],
                'strategy_type': decision['strategy_type'],
                'logic_type': decision['logic_type'],
                'model_id': model_id,
                'mock_sharpe': mock_sharpe,
                'mock_return': mock_return,
                'code_length': len(code)
            })

            logger.info(f"✓ Iteration {i+1} complete\n")

        except Exception as e:
            logger.error(f"✗ Iteration {i+1} failed: {e}")
            import traceback
            traceback.print_exc()

    # 5. Session summary
    logger.info("\n[5/6] Session Summary...")
    logger.info("="*80)

    successful = [r for r in results if r['strategy_id'] is not None]

    logger.info(f"Total strategies explored: {len(results)}")
    logger.info(f"Successfully stored: {len(successful)}")

    # Count by type
    pure_tech = len([r for r in results if r['strategy_type'] == 'pure_technical'])
    hybrid_ml = len([r for r in results if r['strategy_type'] == 'hybrid_ml'])
    pure_ml = len([r for r in results if r['strategy_type'] == 'pure_ml'])

    logger.info(f"\nBy strategy type:")
    logger.info(f"  Pure Technical: {pure_tech}")
    logger.info(f"  Hybrid ML: {hybrid_ml}")
    logger.info(f"  Pure ML: {pure_ml}")

    # Count by logic type
    logic_types = {}
    for r in results:
        lt = r['logic_type']
        logic_types[lt] = logic_types.get(lt, 0) + 1

    logger.info(f"\nBy logic type:")
    for lt, count in logic_types.items():
        logger.info(f"  {lt}: {count}")

    # ML models used
    ml_models_used = set([r['model_id'] for r in results if r['model_id']])
    if ml_models_used:
        logger.info(f"\nML models used: {len(ml_models_used)}")
        for model in ml_models_used:
            logger.info(f"  - {model}")

    # Top performers (mock)
    sorted_results = sorted(results, key=lambda x: x['mock_sharpe'], reverse=True)

    logger.info(f"\nTop 3 performers (mock):")
    for i, r in enumerate(sorted_results[:3], 1):
        logger.info(f"  {i}. {r['strategy_name']} - Sharpe: {r['mock_sharpe']:.2f}, Return: {r['mock_return']:.1%}")

    # 6. Database statistics
    logger.info("\n[6/6] Database Statistics...")

    try:
        stats = database.get_statistics()

        logger.info(f"Total strategies in DB: {stats['total_strategies']}")
        logger.info(f"By status: {stats['strategies_by_status']}")

        underexplored = database.get_underexplored_areas()

        logger.info(f"\nUnderexplored areas:")
        logger.info(f"  By ML type: {underexplored['by_ml_type']}")
        logger.info(f"  By logic type: {underexplored['by_logic_type']}")

    except Exception as e:
        logger.warning(f"Could not get database stats: {e}")

    logger.info("\n" + "="*80)
    logger.info("PHASE 1a SESSION COMPLETE")
    logger.info("="*80)

    logger.info("""
✓ Strategy Agent autonomously explored strategy space
✓ ML Quant Agent provided models on demand
✓ Agent Router coordinated the pipeline
✓ Code Generator created executable strategies
✓ Results stored for future learning

Next steps:
- Run actual backtests on generated strategies
- Implement Phase 1b: Background ML improvement
- Add walk-forward testing (Phase 2)
""")


def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PHASE 1a: AUTONOMOUS RESEARCH SESSION                    ║
║                                                                              ║
║  This script demonstrates the agent-first architecture where AI agents      ║
║  autonomously explore the strategy space with ZERO human intervention.      ║
║                                                                              ║
║  The system will:                                                           ║
║  1. Decide what strategies to explore (Strategy Agent)                      ║
║  2. Provide ML models on demand (ML Quant Agent)                            ║
║  3. Generate executable Python code (Code Generator)                        ║
║  4. Learn from results and adapt                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  WARNING: OPENAI_API_KEY not set in environment")
        print("   The LLM agents may not work without it.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        print("")

        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Run session
    try:
        run_phase1a_session(num_strategies=10)

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user")

    except Exception as e:
        print(f"\n\n✗ Session failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
