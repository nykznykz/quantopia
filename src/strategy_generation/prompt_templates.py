"""Prompt templates for LLM strategy generation."""

from typing import List, Dict, Any
import json


# System prompt for strategy generation
STRATEGY_GENERATION_SYSTEM_PROMPT = """You are an expert quantitative researcher specializing in systematic trading strategies.

Your task is to generate testable, rule-based trading strategies for cryptocurrency markets using only the provided technical indicators.

Requirements:
1. Strategies must be fully rule-based with NO discretionary decisions
2. Entry and exit rules must be clearly defined using mathematical conditions
3. Use ONLY indicators from the provided indicator pool
4. Strategies should be atomic and testable (one core idea per strategy)
5. Include risk management (stop loss, take profit, or time-based exits)
6. Specify the market regime where strategy works best
7. Provide a clear hypothesis for why the strategy should work

Output Format:
Return a JSON object with the following structure:
{
    "strategy_name": "Descriptive strategy name",
    "strategy_type": "mean_reversion" | "trend_following" | "breakout" | "momentum" | "volatility",
    "hypothesis": "Clear explanation of why this strategy should work",
    "market_regime": "trending" | "ranging" | "high_volatility" | "low_volatility" | "any",
    "indicators": [
        {
            "name": "indicator_name",
            "params": {"param1": value1, "param2": value2}
        }
    ],
    "entry_rules": [
        "Clear mathematical condition 1 (e.g., RSI < 30)",
        "AND/OR condition 2",
        ...
    ],
    "exit_rules": [
        "Clear mathematical condition for exit",
        ...
    ],
    "stop_loss": {
        "type": "fixed_percentage" | "atr_based" | "none",
        "value": percentage or multiplier
    },
    "take_profit": {
        "type": "fixed_percentage" | "trailing" | "none",
        "value": percentage or multiplier
    },
    "position_sizing": "fixed_percentage" | "volatility_based",
    "expected_holding_period": "intraday" | "swing" | "position"
}

Be creative but realistic. Focus on strategies with sound logical basis."""


def create_strategy_generation_prompt(
    strategy_type: str = None,
    num_strategies: int = 1,
    indicators_list: List[str] = None,
    avoid_similar_to: List[Dict] = None
) -> str:
    """Create prompt for generating trading strategies.

    Args:
        strategy_type: Desired strategy type (optional)
        num_strategies: Number of strategies to generate
        indicators_list: List of available indicators
        avoid_similar_to: List of existing strategy metadata to avoid duplicates

    Returns:
        Formatted prompt string
    """
    prompt_parts = []

    # Start with task
    if num_strategies == 1:
        prompt_parts.append("Generate ONE novel trading strategy for cryptocurrency markets.")
    else:
        prompt_parts.append(
            f"Generate {num_strategies} DIVERSE trading strategies for cryptocurrency markets. "
            "Each strategy should be significantly different from the others."
        )

    # Add strategy type constraint if specified
    if strategy_type:
        prompt_parts.append(f"\nStrategy Type: Focus on {strategy_type} strategies.")

    # Add indicator pool
    if indicators_list:
        prompt_parts.append(f"\nAvailable Indicators: {', '.join(indicators_list)}")
    else:
        prompt_parts.append(
            "\nAvailable Indicators: EMA, SMA, RSI, MACD, Bollinger Bands, ATR, "
            "Stochastic, ADX, Volume, and other common technical indicators."
        )

    # Add diversity constraint
    if avoid_similar_to:
        prompt_parts.append(
            "\nIMPORTANT: Generate strategies that are DIFFERENT from these existing ones:"
        )
        for i, existing in enumerate(avoid_similar_to[:5], 1):  # Show max 5 examples
            strategy_name = existing.get('strategy_name', f'Strategy {i}')
            strategy_type_existing = existing.get('strategy_type', 'unknown')
            prompt_parts.append(f"  - {strategy_name} ({strategy_type_existing})")

    # Add guidelines
    prompt_parts.append(
        "\nGuidelines:"
        "\n- Each strategy should have a clear, testable hypothesis"
        "\n- Entry and exit rules must be specific and rule-based"
        "\n- Include proper risk management (stop loss/take profit)"
        "\n- Avoid overly complex strategies with too many conditions"
        "\n- Focus on robust ideas that work across different market conditions"
    )

    # Add output format reminder
    if num_strategies == 1:
        prompt_parts.append(
            "\nReturn your strategy as a single JSON object following the specified format."
        )
    else:
        prompt_parts.append(
            f"\nReturn your {num_strategies} strategies as a JSON array of strategy objects, "
            "following the specified format."
        )

    return "\n".join(prompt_parts)


def create_refinement_prompt(
    strategy: Dict[str, Any],
    backtest_results: Dict[str, Any],
    failure_analysis: str
) -> str:
    """Create prompt for refining a failed strategy.

    Args:
        strategy: Original strategy metadata
        backtest_results: Backtest metrics
        failure_analysis: Analysis of why strategy failed

    Returns:
        Refinement prompt
    """
    strategy_json = json.dumps(strategy, indent=2)
    metrics_json = json.dumps(backtest_results.get('metrics', {}), indent=2)

    prompt = f"""The following trading strategy failed backtesting:

Original Strategy:
{strategy_json}

Backtest Results:
{metrics_json}

Failure Analysis:
{failure_analysis}

Your task: Generate an IMPROVED version of this strategy that addresses the identified issues.

Requirements:
1. Keep the core concept but modify parameters or add/remove conditions
2. Address specific failure modes identified in the analysis
3. Maintain the same strategy type unless the failure suggests a different approach
4. Ensure the refined strategy is still rule-based and testable

Return the refined strategy as a JSON object in the same format as the original.
"""

    return prompt


def create_strategy_family_prompt(
    base_strategy: Dict[str, Any],
    num_variations: int = 3
) -> str:
    """Create prompt for generating strategy variations.

    Args:
        base_strategy: Base strategy to create variations of
        num_variations: Number of variations to generate

    Returns:
        Prompt for generating strategy family
    """
    strategy_json = json.dumps(base_strategy, indent=2)

    prompt = f"""You have a successful base trading strategy:

{strategy_json}

Generate {num_variations} VARIATIONS of this strategy that:
1. Keep the core concept but adjust parameters or timeframes
2. Add or remove secondary conditions
3. Use different but related indicators
4. Adjust risk management parameters

Each variation should be distinct enough to potentially perform differently,
but similar enough to be part of the same "strategy family".

Return the variations as a JSON array of strategy objects."""

    return prompt


# Pre-defined strategy type prompts
STRATEGY_TYPE_HINTS = {
    "mean_reversion": """
        Mean reversion strategies assume that prices tend to revert to their average.
        Look for oversold/overbought conditions, price deviations from moving averages,
        RSI extremes, or Bollinger Band touches.
    """,
    "trend_following": """
        Trend following strategies aim to capture sustained directional moves.
        Look for moving average crossovers, ADX strength, price breakouts above resistance,
        or MACD signals.
    """,
    "breakout": """
        Breakout strategies enter when price breaks through key levels.
        Look for volume surges, volatility expansion, Donchian channel breaks,
        or price clearing consolidation ranges.
    """,
    "momentum": """
        Momentum strategies follow strong directional price moves.
        Look for RSI trending above 50, rate of change acceleration,
        or price making higher highs/lower lows.
    """,
    "volatility": """
        Volatility strategies profit from changes in price volatility.
        Look for ATR expansion/contraction, Bollinger Band squeezes,
        or historical volatility regime shifts.
    """
}


def get_indicator_descriptions() -> Dict[str, str]:
    """Get descriptions of available indicators.

    Returns:
        Dict mapping indicator name to description
    """
    return {
        # Trend
        "EMA": "Exponential Moving Average - smoothed price trend",
        "SMA": "Simple Moving Average - basic price average",
        "MACD": "Moving Average Convergence Divergence - trend momentum",
        "ADX": "Average Directional Index - trend strength (0-100)",
        "EMASlope": "EMA rate of change - trend acceleration",

        # Momentum
        "RSI": "Relative Strength Index - overbought/oversold (0-100)",
        "Stochastic": "Stochastic Oscillator - momentum (0-100)",
        "MFI": "Money Flow Index - volume-weighted momentum (0-100)",
        "ROC": "Rate of Change - percentage price change",
        "WilliamsR": "Williams %R - momentum indicator (-100 to 0)",

        # Volatility
        "ATR": "Average True Range - volatility measure",
        "BollingerBands": "Bollinger Bands - volatility channels (upper/middle/lower)",
        "KeltnerChannels": "Keltner Channels - ATR-based channels",
        "HistoricalVolatility": "Annualized historical volatility",
        "DonchianChannels": "Donchian Channels - highest high/lowest low",

        # Regime
        "VolumeZScore": "Volume Z-Score - volume anomalies",
        "HurstExponent": "Hurst Exponent - trend persistence (0-1)",
        "MarketRegime": "Market regime classifier - trend/volatility/volume",
        "TrendStrength": "Trend strength score - linear regression based",
        "VolatilityRegime": "Volatility regime - short/long vol ratio",
    }
