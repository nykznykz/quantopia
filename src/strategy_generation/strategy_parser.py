"""Parser for LLM-generated strategy JSON output."""

import json
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StrategyParseError(Exception):
    """Raised when strategy JSON parsing fails."""
    pass


class StrategyParser:
    """Parser and validator for strategy JSON from LLM."""

    REQUIRED_FIELDS = [
        'strategy_name',
        'strategy_type',
        'hypothesis',
        'indicators',
        'entry_rules',
        'exit_rules'
    ]

    VALID_STRATEGY_TYPES = [
        'mean_reversion',
        'trend_following',
        'breakout',
        'momentum',
        'volatility'
    ]

    VALID_MARKET_REGIMES = [
        'trending',
        'ranging',
        'high_volatility',
        'low_volatility',
        'any'
    ]

    def __init__(self, available_indicators: List[str]):
        """Initialize parser.

        Args:
            available_indicators: List of valid indicator names
        """
        self.available_indicators = set(available_indicators)

    def parse(self, json_text: str) -> Dict[str, Any]:
        """Parse and validate strategy JSON.

        Args:
            json_text: JSON string from LLM

        Returns:
            Validated strategy dict

        Raises:
            StrategyParseError: If parsing or validation fails
        """
        # Parse JSON
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise StrategyParseError(f"Invalid JSON: {e}")

        # Handle array of strategies (extract first one)
        if isinstance(data, list):
            if len(data) == 0:
                raise StrategyParseError("Empty strategy array")
            logger.warning("Received array of strategies, using first one")
            data = data[0]

        # Validate required fields
        missing_fields = [
            field for field in self.REQUIRED_FIELDS
            if field not in data
        ]
        if missing_fields:
            raise StrategyParseError(
                f"Missing required fields: {missing_fields}"
            )

        # Validate strategy type
        if data['strategy_type'] not in self.VALID_STRATEGY_TYPES:
            logger.warning(
                f"Unknown strategy type: {data['strategy_type']}. "
                f"Valid types: {self.VALID_STRATEGY_TYPES}"
            )

        # Validate market regime (optional field)
        if 'market_regime' in data:
            if data['market_regime'] not in self.VALID_MARKET_REGIMES:
                logger.warning(
                    f"Unknown market regime: {data['market_regime']}. "
                    f"Valid regimes: {self.VALID_MARKET_REGIMES}"
                )

        # Validate indicators
        invalid_indicators = []
        for indicator in data['indicators']:
            indicator_name = indicator.get('name')
            if indicator_name and indicator_name not in self.available_indicators:
                invalid_indicators.append(indicator_name)

        if invalid_indicators:
            raise StrategyParseError(
                f"Invalid indicators: {invalid_indicators}. "
                f"Available indicators: {sorted(self.available_indicators)}"
            )

        # Validate entry/exit rules
        if not isinstance(data['entry_rules'], list) or len(data['entry_rules']) == 0:
            raise StrategyParseError("entry_rules must be a non-empty list")

        if not isinstance(data['exit_rules'], list) or len(data['exit_rules']) == 0:
            raise StrategyParseError("exit_rules must be a non-empty list")

        # Add defaults for optional fields
        data.setdefault('market_regime', 'any')
        data.setdefault('stop_loss', {'type': 'none'})
        data.setdefault('take_profit', {'type': 'none'})
        data.setdefault('position_sizing', 'fixed_percentage')
        data.setdefault('expected_holding_period', 'swing')

        logger.info(f"Successfully parsed strategy: {data['strategy_name']}")

        return data

    def parse_batch(self, json_text: str) -> List[Dict[str, Any]]:
        """Parse array of strategies.

        Args:
            json_text: JSON string containing array of strategies

        Returns:
            List of validated strategy dicts

        Raises:
            StrategyParseError: If parsing fails
        """
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise StrategyParseError(f"Invalid JSON: {e}")

        # Ensure it's an array
        if not isinstance(data, list):
            # If single strategy, wrap in list
            data = [data]

        # Parse each strategy
        strategies = []
        errors = []

        for i, strategy_data in enumerate(data):
            try:
                # Temporarily convert to JSON string for parse() method
                strategy_json = json.dumps(strategy_data)
                parsed = self.parse(strategy_json)
                strategies.append(parsed)
            except StrategyParseError as e:
                logger.error(f"Error parsing strategy {i+1}: {e}")
                errors.append((i+1, str(e)))

        if len(strategies) == 0:
            raise StrategyParseError(
                f"Failed to parse any strategies. Errors: {errors}"
            )

        if errors:
            logger.warning(
                f"Parsed {len(strategies)}/{len(data)} strategies successfully. "
                f"Errors: {errors}"
            )

        return strategies

    def validate_indicator_params(self, indicator: Dict[str, Any]) -> bool:
        """Validate indicator parameters.

        Args:
            indicator: Indicator dict with name and params

        Returns:
            True if valid, False otherwise
        """
        indicator_name = indicator.get('name')
        params = indicator.get('params', {})

        if not indicator_name:
            logger.error("Indicator missing 'name' field")
            return False

        # Basic validation: params should be a dict
        if not isinstance(params, dict):
            logger.error(f"Indicator {indicator_name}: params must be a dict")
            return False

        # Validate common parameter types
        for param_name, param_value in params.items():
            if not isinstance(param_value, (int, float, str, bool)):
                logger.error(
                    f"Indicator {indicator_name}: "
                    f"param '{param_name}' has invalid type {type(param_value)}"
                )
                return False

        return True

    def to_dict(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Convert strategy to standardized dict format.

        Args:
            strategy: Parsed strategy dict

        Returns:
            Standardized strategy dict
        """
        return {
            'strategy_name': strategy['strategy_name'],
            'strategy_type': strategy['strategy_type'],
            'hypothesis': strategy['hypothesis'],
            'market_regime': strategy.get('market_regime', 'any'),
            'indicators': strategy['indicators'],
            'entry_rules': strategy['entry_rules'],
            'exit_rules': strategy['exit_rules'],
            'stop_loss': strategy.get('stop_loss', {'type': 'none'}),
            'take_profit': strategy.get('take_profit', {'type': 'none'}),
            'position_sizing': strategy.get('position_sizing', 'fixed_percentage'),
            'expected_holding_period': strategy.get('expected_holding_period', 'swing')
        }
