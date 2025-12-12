"""
Agent Router: Routes strategy requests to appropriate agents.

Routes based on strategy type:
- Pure technical → Code Generator directly
- Hybrid ML / Pure ML → ML Quant Agent → Code Generator
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from src.agents.ml_quant_agent import MLQuantAgent
from src.code_generation.code_generator import CodeGenerator

logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Routes strategy requests through appropriate agent pipeline.

    Decision flow:
    1. Strategy Agent decides what to try
    2. Router checks if ML is needed
    3. If ML: routes through ML Quant Agent
    4. Finally: routes to Code Generator
    """

    def __init__(
        self,
        code_generator: CodeGenerator,
        ml_quant_agent: Optional[MLQuantAgent] = None
    ):
        """
        Initialize Agent Router.

        Args:
            code_generator: Code generation agent
            ml_quant_agent: ML Quant Agent (optional, needed for ML strategies)
        """
        self.code_generator = code_generator
        self.ml_quant_agent = ml_quant_agent

        logger.info("Initialized AgentRouter")

    def route(
        self,
        decision: Dict[str, Any],
        training_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Route strategy decision through appropriate agents.

        Args:
            decision: Strategy decision from Strategy Agent
            training_data: Optional training data for ML models

        Returns:
            Dict with:
                - 'strategy_metadata': Enhanced strategy metadata
                - 'code': Generated Python code
                - 'model_id': ML model ID (if applicable)
        """
        strategy_type = decision.get('strategy_type', 'pure_technical')

        logger.info(f"Routing {strategy_type} strategy: {decision.get('rationale', '')}")

        # Build strategy metadata
        strategy_metadata = self._build_strategy_metadata(decision)

        # Check if ML is required
        if strategy_type in ['hybrid_ml', 'pure_ml']:
            # Route through ML Quant Agent
            if not self.ml_quant_agent:
                raise ValueError("ML Quant Agent required for ML strategies but not provided")

            logger.info("Routing through ML Quant Agent...")

            ml_requirements = decision.get('ml_requirements', {})

            # Get or create model
            model_id = self.ml_quant_agent.get_or_create_model(
                spec=ml_requirements,
                training_data=training_data
            )

            logger.info(f"✓ ML model ready: {model_id}")

            # Add ML info to strategy metadata
            strategy_metadata['ml_strategy_type'] = strategy_type
            strategy_metadata['ml_models_used'] = [model_id]
            strategy_metadata['ml_model_id'] = model_id
            strategy_metadata['ml_prediction_role'] = ml_requirements.get('prediction_role', 'entry_signal')

        else:
            # Pure technical - no ML needed
            strategy_metadata['ml_strategy_type'] = 'pure_technical'

        # Generate code
        logger.info("Routing to Code Generator...")

        code = self.code_generator.generate_strategy_class(strategy_metadata)

        logger.info(f"✓ Code generated: {len(code)} characters")

        return {
            'strategy_metadata': strategy_metadata,
            'code': code,
            'model_id': strategy_metadata.get('ml_model_id'),
            'strategy_type': strategy_type
        }

    def _build_strategy_metadata(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build complete strategy metadata from decision.

        Args:
            decision: Strategy Agent decision

        Returns:
            Strategy metadata dict
        """
        # UPDATED: Pass through complete specification from Strategy Agent
        # NO rule generation - Strategy Agent provides exact logic

        # Extract info from decision
        strategy_type = decision.get('strategy_type', 'pure_technical')
        logic_type = decision.get('logic_type', 'trend_following')
        rationale = decision.get('rationale', '')

        # Get strategy name from decision or generate
        if 'strategy_name' in decision:
            strategy_name = decision['strategy_name']
        else:
            # Fallback for old format
            indicators = decision.get('indicators', [])
            strategy_name = self._generate_strategy_name(logic_type, indicators, strategy_type)

        # Get indicators list (support both old and new format)
        if 'indicators_required' in decision:
            indicators = [ind.get('name', ind) if isinstance(ind, dict) else ind
                         for ind in decision['indicators_required']]
        else:
            indicators = decision.get('indicators', [])

        # Build metadata - pass through Strategy Agent's specifications
        metadata = {
            'strategy_name': strategy_name,
            'strategy_type': logic_type,
            'hypothesis': rationale,

            # NEW: Complete specifications from Strategy Agent
            'entry_conditions': decision.get('entry_conditions', {}),
            'exit_conditions': decision.get('exit_conditions', {}),
            'risk_management': decision.get('risk_management', {
                'stop_loss': {'type': 'percentage', 'value': '2%'},
                'take_profit': {'type': 'percentage', 'value': '5%'},
                'position_sizing': '90% of capital'
            }),

            # Indicator list
            'indicators': [{'name': ind} if isinstance(ind, str) else ind for ind in indicators],

            # Legacy fields for backward compatibility
            'entry_rules': decision.get('entry_conditions', {}).get('components', []),
            'exit_rules': decision.get('exit_conditions', {}).get('components', []),

            # Add decision metadata
            'exploration_mode': decision.get('exploration_mode', 'explore'),
            'decided_at': decision.get('decided_at'),
            'agent': decision.get('agent', 'StrategyAgent')
        }

        logger.info(f"Built metadata for: {strategy_name}")
        if 'entry_conditions' in decision:
            logger.info(f"  Entry logic: {decision['entry_conditions'].get('logic', 'N/A')[:100]}")
            logger.info(f"  Exit logic: {decision['exit_conditions'].get('logic', 'N/A')[:100]}")

        return metadata

    def _generate_strategy_name(
        self,
        logic_type: str,
        indicators: list,
        strategy_type: str
    ) -> str:
        """Generate unique strategy name."""
        import random
        import string

        # Build name from components
        ind_str = '_'.join(indicators[:2])  # First 2 indicators
        ml_suffix = '_ML' if strategy_type != 'pure_technical' else ''

        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

        name = f"{logic_type}_{ind_str}{ml_suffix}_{random_suffix}"

        return name

    # REMOVED: _generate_rules() method
    # Strategy Agent now provides complete entry/exit logic
    # No hardcoded rule generation needed
