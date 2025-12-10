#!/usr/bin/env python3
"""Quick test of strategy generation with real API."""

import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
# Load .env from project root
load_dotenv(project_root / '.env')

from src.strategy_generation.llm_client import LLMClient
from src.strategy_generation.generator import StrategyGenerator
from src.indicators import INDICATOR_REGISTRY


def main():
    import os
    print("Initializing LLM client...")

    # Get provider from environment
    provider = os.getenv("LLM_PROVIDER", "openai")
    print(f"Using LLM provider: {provider}")

    # Initialize LLM client based on provider
    if provider == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            raise ValueError("AZURE_OPENAI_API_KEY not found. Please check your .env file.")
        print("✓ Azure OpenAI credentials loaded")
        llm_client = LLMClient(
            provider="azure",
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4"),
            temperature=0.7,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
        print("✓ OpenAI API key loaded")
        llm_client = LLMClient(
            provider="openai",
            model="gpt-4",
            temperature=0.7
        )

    # Initialize strategy generator
    available_indicators = list(INDICATOR_REGISTRY.keys())
    generator = StrategyGenerator(
        llm_client=llm_client,
        available_indicators=available_indicators
    )

    # Generate a single strategy
    print("\nGenerating strategy...")
    try:
        strategy = generator.generate_strategy(
            strategy_type="momentum",
            temperature=0.7
        )

        print(f"\n{'='*80}")
        print(f"Generated Strategy: {strategy['strategy_name']}")
        print(f"{'='*80}")
        print(f"Type: {strategy['strategy_type']}")
        print(f"Hypothesis: {strategy['hypothesis']}")
        print(f"Market Regime: {strategy.get('market_regime', 'any')}")

        print(f"\nIndicators:")
        for ind in strategy['indicators']:
            print(f"  - {ind['name']}: {ind.get('params', {})}")

        print(f"\nEntry Rules:")
        for rule in strategy['entry_rules']:
            print(f"  - {rule}")

        print(f"\nExit Rules:")
        for rule in strategy['exit_rules']:
            print(f"  - {rule}")

        print(f"\nRisk Management:")
        stop_loss = strategy.get('stop_loss', {})
        print(f"  Stop Loss: {stop_loss.get('type', 'none')}" +
              (f" ({stop_loss.get('value')})" if stop_loss.get('value') else ""))

        take_profit = strategy.get('take_profit', {})
        print(f"  Take Profit: {take_profit.get('type', 'none')}" +
              (f" ({take_profit.get('value')})" if take_profit.get('value') else ""))

        print(f"\nPosition Sizing: {strategy.get('position_sizing', 'N/A')}")
        print(f"Expected Holding Period: {strategy.get('expected_holding_period', 'N/A')}")
        print(f"{'='*80}")

        print("\n✅ Success! API is working correctly.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that OPENAI_API_KEY is set: echo $OPENAI_API_KEY")
        print("2. Verify your API key is valid at https://platform.openai.com/api-keys")
        print("3. Ensure you have credits/payment method on your OpenAI account")
        sys.exit(1)


if __name__ == '__main__':
    main()
