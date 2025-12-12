"""
Strategy Agent: Autonomous strategy researcher exploring the search space.

This agent:
- Queries database to understand what's been tried
- Analyzes performance patterns
- Discovers available ML models
- Decides next strategy using LLM reasoning
- Balances exploration vs exploitation
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.database.manager import StrategyDatabase
from src.ml.model_registry import ModelRegistry
from src.strategy_generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


class StrategyAgent:
    """
    Autonomous strategy researcher that explores the strategy space.

    Uses LLM reasoning to decide what to try next based on:
    - Historical performance patterns
    - Available ML models
    - Underexplored areas
    - Exploration/exploitation tradeoff
    """

    def __init__(
        self,
        database: StrategyDatabase,
        model_registry: ModelRegistry,
        llm_client: Optional[LLMClient] = None,
        exploration_rate: float = 0.3
    ):
        """
        Initialize Strategy Agent.

        Args:
            database: Strategy database for querying history
            model_registry: Model registry for discovering ML models
            llm_client: LLM client for decision making
            exploration_rate: Balance between exploration (0.0) and exploitation (1.0)
        """
        self.database = database
        self.model_registry = model_registry
        self.exploration_rate = exploration_rate

        if llm_client is None:
            self.llm_client = LLMClient(
                provider="openai",
                model="gpt-4",
                temperature=0.7  # Creative but not random
            )
        else:
            self.llm_client = llm_client

        logger.info(f"Initialized StrategyAgent (exploration_rate={exploration_rate})")

    def explore_next_strategy(self) -> Dict[str, Any]:
        """
        Decide what strategy to explore next using LLM reasoning.

        Returns:
            Strategy specification dict with rationale
        """
        logger.info("StrategyAgent: Deciding next strategy...")

        # 1. Gather context from database
        context = self._gather_context()

        # 2. Discover available ML models
        available_models = self._discover_ml_models()

        # 3. Build LLM prompt with context
        prompt = self._build_exploration_prompt(context, available_models)

        # 4. Get LLM decision
        decision = self._get_llm_decision(prompt)

        logger.info(f"Decision: {decision.get('strategy_type')} - {decision.get('rationale')}")

        return decision

    def _gather_context(self) -> Dict[str, Any]:
        """Gather relevant context from database."""
        logger.info("Gathering context from database...")

        # Get statistics
        stats = self.database.get_statistics()

        # Get top performers
        top_overall = self.database.get_top_strategies(metric='sharpe_ratio', limit=5)
        top_pure_tech = self.database.get_top_strategies(
            metric='sharpe_ratio', limit=3, ml_strategy_type='pure_technical'
        )
        top_hybrid = self.database.get_top_strategies(
            metric='sharpe_ratio', limit=3, ml_strategy_type='hybrid_ml'
        )
        top_pure_ml = self.database.get_top_strategies(
            metric='sharpe_ratio', limit=3, ml_strategy_type='pure_ml'
        )

        # Get recent strategies to avoid duplicates
        recent = self.database.get_recent_strategies(days=7, limit=20)

        # Get underexplored areas
        underexplored = self.database.get_underexplored_areas()

        context = {
            'total_strategies': stats.get('total_strategies', 0),
            'by_ml_type': underexplored.get('ml_performance', {}),
            'top_performers': top_overall,
            'top_by_ml_type': {
                'pure_technical': top_pure_tech,
                'hybrid_ml': top_hybrid,
                'pure_ml': top_pure_ml
            },
            'recent_strategies': [s['name'] for s in recent],
            'underexplored': underexplored
        }

        return context

    def _load_market_statistics(self) -> str:
        """Load market statistics for threshold calibration."""
        from pathlib import Path

        stats_path = Path('scratchpad/market_statistics.json')

        if not stats_path.exists():
            logger.warning("Market statistics not found, using defaults")
            return """
RSI: Use realistic thresholds
  - Oversold: < 40-45 (not < 30, too rare)
  - Overbought: > 55-60 (not > 70, too rare)

TrendStrength: Returns FLOAT 0.0-1.0
  - Ranging: < 0.3-0.4
  - Trending: > 0.6-0.7
  - Use numeric comparison (< 0.5), NOT string matching ('ranging')

ROC: Strong momentum
  - Positive: > 1.0-1.5% (not > 2%, too rare)
  - Negative: < -1.0-1.5%

General Rules:
- Use thresholds that occur 20-40% of time for balanced signals
- Avoid extreme thresholds that rarely trigger (< 5% frequency)
"""

        try:
            import json
            with open(stats_path) as f:
                stats = json.load(f)

            # Format for LLM consumption
            output = "RSI:\n"
            if 'RSI' in stats.get('indicators', {}):
                rsi = stats['indicators']['RSI']
                rec = rsi.get('recommendations', {})
                freq = rsi.get('frequency_analysis', {})
                output += f"  - Traditional oversold (< 30): {freq.get('below_30', 5):.1f}% of time (TOO RARE)\n"
                output += f"  - Realistic oversold (< {rec.get('oversold_realistic', 44):.0f}): ~30% of time (BALANCED)\n"
                output += f"  - Traditional overbought (> 70): {freq.get('above_70', 5):.1f}% of time (TOO RARE)\n"
                output += f"  - Realistic overbought (> {rec.get('overbought_realistic', 57):.0f}): ~30% of time (BALANCED)\n"

            output += "\nTrendStrength: Returns FLOAT 0.0-1.0\n"
            if 'TrendStrength' in stats.get('indicators', {}):
                ts = stats['indicators']['TrendStrength']
                rec = ts.get('recommendations', {})
                output += f"  - Ranging threshold: < {rec.get('ranging_threshold', 0.3):.2f}\n"
                output += f"  - Trending threshold: > {rec.get('trending_threshold', 0.7):.2f}\n"
            output += "  - CRITICAL: Use numeric comparison (< 0.5), NOT string matching ('ranging')\n"

            output += "\nROC (Rate of Change):\n"
            if 'ROC' in stats.get('indicators', {}):
                roc = stats['indicators']['ROC']
                rec = roc.get('recommendations', {})
                freq = roc.get('frequency_analysis', {})
                output += f"  - Strong positive: > {rec.get('strong_momentum_positive', 1.2):.2f}%\n"
                output += f"  - Strong negative: < {rec.get('strong_momentum_negative', -1.2):.2f}%\n"
                output += f"  - Note: ROC > 2% occurs only {freq.get('above_2pct', 10):.1f}% of time (too rare)\n"

            if 'BollingerBands' in stats.get('indicators', {}):
                bb = stats['indicators']['BollingerBands']
                rec = bb.get('recommendations', {})
                output += f"\nBollinger Bands:\n"
                output += f"  - Squeeze: width < {rec.get('squeeze_threshold', 1.3):.2f}%\n"
                output += f"  - Expansion: width > {rec.get('expansion_threshold', 3.3):.2f}%\n"

            output += "\nGENERAL RULES:\n"
            output += "  - Use thresholds that occur 20-40% of time for balanced signals\n"
            output += "  - Avoid extreme thresholds that rarely trigger (< 5% frequency)\n"
            output += "  - Make conditions compatible (don't combine low volatility + high momentum)\n"

            return output

        except Exception as e:
            logger.error(f"Error loading market statistics: {e}")
            return "Market statistics unavailable - use conservative thresholds"

    def _discover_ml_models(self) -> List[Dict[str, Any]]:
        """Discover available ML models from registry."""
        logger.info("Discovering available ML models...")

        model_names = self.model_registry.list_all_models()

        available_models = []
        for model_name in model_names:
            latest = self.model_registry.get_latest_version(model_name)
            if latest:
                available_models.append({
                    'model_id': latest['model_id'],
                    'model_name': model_name,
                    'version': latest.get('version', 'unknown'),
                    'model_type': latest.get('model_type'),
                    'target': latest.get('target'),
                    'metrics': latest.get('metrics', {}),
                    'usage_stats': latest.get('usage_stats', {}),
                    'released_at': latest.get('released_at')
                })

        logger.info(f"Found {len(available_models)} available ML models")

        return available_models

    def _build_exploration_prompt(
        self,
        context: Dict[str, Any],
        available_models: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM to decide next strategy."""

        # Format context for LLM
        total = context['total_strategies']
        by_ml = context['by_ml_type']

        # Load market statistics for threshold calibration
        market_stats = self._load_market_statistics()

        prompt = f"""You are an autonomous quant researcher exploring the strategy space.

DATABASE CONTEXT:
Total strategies tried: {total}

Performance by strategy type:
- Pure Technical: {by_ml.get('pure_technical', {}).get('count', 0)} strategies (avg Sharpe: {by_ml.get('pure_technical', {}).get('avg_sharpe', 0):.2f})
- Hybrid ML: {by_ml.get('hybrid_ml', {}).get('count', 0)} strategies (avg Sharpe: {by_ml.get('hybrid_ml', {}).get('avg_sharpe', 0):.2f})
- Pure ML: {by_ml.get('pure_ml', {}).get('count', 0)} strategies (avg Sharpe: {by_ml.get('pure_ml', {}).get('avg_sharpe', 0):.2f})

MARKET STATISTICS (BTC-USD 1h, use these for realistic thresholds):
{market_stats}

TOP 5 PERFORMERS:
"""

        for i, strategy in enumerate(context['top_performers'][:5], 1):
            prompt += f"{i}. {strategy['name']} ({strategy['ml_strategy_type']}) - Sharpe: {strategy['metrics']['sharpe_ratio']:.2f}\n"
            if strategy.get('ml_models_used'):
                prompt += f"   Models: {', '.join(strategy['ml_models_used'])}\n"

        prompt += "\nAVAILABLE ML MODELS:\n"
        if available_models:
            for model in available_models:
                usage = model.get('usage_stats', {})
                prompt += f"- {model['model_id']} ({model['model_type']})\n"
                prompt += f"  Target: {model.get('target', 'unknown')}\n"
                prompt += f"  Test accuracy: {model.get('metrics', {}).get('test_accuracy', 'N/A')}\n"
                prompt += f"  Used in {usage.get('num_strategies_using', 0)} strategies (avg Sharpe: {usage.get('avg_sharpe_of_strategies', 0):.2f})\n"
        else:
            prompt += "(No ML models available yet - suggest pure technical strategies)\n"

        prompt += f"""
UNDEREXPLORED AREAS:
Strategy logic types attempted: {list(context['underexplored']['by_logic_type'].keys())}
Underused indicators: {[ind for ind, count in context['underexplored'].get('underused_indicators', [])]}

RECENT STRATEGIES (avoid similar):
{', '.join(context['recent_strategies'][:10])}

YOUR TASK:
Design a COMPLETE strategy specification. Consider:
1. **Exploit top performers**: Refine successful approaches (especially hybrid ML if they're winning)
2. **Explore new areas**: Try underexplored logic types or indicators
3. **Upgrade models**: Retry past strategies with newer model versions
4. **Balance**: ~70% exploitation, ~30% exploration

CRITICAL REQUIREMENTS:
- Provide EXACT entry/exit logic with specific thresholds (e.g., "RSI(14) < 44", not "RSI oversold")
- Use REALISTIC thresholds from market statistics (e.g., RSI < 44 not < 30)
- Specify AND/OR combination logic explicitly
- Include rationale for each threshold choice
- Make conditions compatible (don't combine low volatility + high momentum)

Output JSON format:
{{
    "strategy_type": "pure_technical|hybrid_ml|pure_ml",
    "strategy_name": "Descriptive name",
    "rationale": "2-3 sentence explanation of why this strategy and expected edge",
    "logic_type": "trend_following|mean_reversion|breakout|momentum|volatility",

    "entry_conditions": {{
        "description": "Human-readable entry logic summary",
        "logic": "Exact boolean expression: (RSI(14) < 44 OR price < BB_lower) AND volume > avg_volume",
        "components": [
            {{
                "indicator": "RSI",
                "period": 14,
                "condition": "< 44",
                "rationale": "Use 44 (30th percentile) not 30 - occurs 30% vs 4.6% of time per market stats"
            }}
        ],
        "combination_logic": "Explain AND/OR choices and why conditions are compatible"
    }},

    "exit_conditions": {{
        "description": "Human-readable exit logic summary",
        "logic": "RSI(14) > 55 OR holding_bars >= 5 OR take_profit_hit OR stop_loss_hit",
        "components": [
            {{
                "indicator": "RSI",
                "period": 14,
                "condition": "> 55",
                "rationale": "Exit when back to neutral zone"
            }}
        ],
        "combination_logic": "Use OR for exits - exit on any condition"
    }},

    "risk_management": {{
        "stop_loss": {{
            "type": "ATR_multiple|percentage|fixed",
            "value": "2.0",
            "description": "2.0 * ATR(14) - dynamic stop based on volatility"
        }},
        "take_profit": {{
            "type": "ATR_multiple|percentage|fixed",
            "value": "1.5",
            "description": "1.5 * ATR(14) - 1:1.5 risk-reward"
        }},
        "position_sizing": "90% of capital"
    }},

    "indicators_required": [
        {{"name": "RSI", "period": 14}},
        {{"name": "ATR", "period": 14}}
    ],

    "ml_requirements": {{  // Only if strategy_type is hybrid_ml or pure_ml
        "model_id": "XGBoost_direction_v3",
        "prediction_role": "entry_signal|exit_signal|position_sizing|both"
    }},

    "exploration_mode": "exploit|explore|upgrade"
}}

Generate the COMPLETE strategy specification JSON now:"""

        return prompt

    def _get_llm_decision(self, prompt: str) -> Dict[str, Any]:
        """Get LLM decision on next strategy."""

        system_prompt = """You are an expert algorithmic trading researcher making strategic decisions about what to explore next.
Provide your decision as valid JSON only, no additional text."""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7  # Balanced creativity
            )

            # Parse JSON response
            # Remove markdown fences if present
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            decision = json.loads(response.strip())

            # Validate required fields for new format
            required = ['strategy_type', 'rationale', 'logic_type', 'entry_conditions', 'exit_conditions']
            for field in required:
                if field not in decision:
                    raise ValueError(f"Missing required field: {field}")

            # Validate entry/exit conditions structure
            if 'logic' not in decision['entry_conditions']:
                raise ValueError("entry_conditions must have 'logic' field with exact boolean expression")
            if 'logic' not in decision['exit_conditions']:
                raise ValueError("exit_conditions must have 'logic' field with exact boolean expression")

            # Extract indicators from components for backward compatibility
            if 'indicators_required' not in decision:
                # Build from entry/exit components
                indicators = set()
                for component in decision.get('entry_conditions', {}).get('components', []):
                    indicators.add(component.get('indicator', ''))
                for component in decision.get('exit_conditions', {}).get('components', []):
                    indicators.add(component.get('indicator', ''))
                decision['indicators_required'] = [{'name': ind} for ind in indicators if ind]

            # Add metadata
            decision['decided_at'] = datetime.now().isoformat()
            decision['agent'] = 'StrategyAgent'

            return decision

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            logger.error(f"Response was: {response if 'response' in locals() else 'no response'}")

            # Fallback to simple exploration
            return self._fallback_decision()

    def _fallback_decision(self) -> Dict[str, Any]:
        """Fallback decision if LLM fails."""
        logger.warning("Using fallback decision strategy")

        # Simple strategy: try pure technical
        return {
            'strategy_type': 'pure_technical',
            'rationale': 'Fallback decision due to LLM error. Trying pure technical strategy.',
            'logic_type': 'trend_following',
            'indicators': ['RSI', 'EMA'],
            'exploration_mode': 'explore',
            'decided_at': datetime.now().isoformat(),
            'agent': 'StrategyAgent (fallback)'
        }
