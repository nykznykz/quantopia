"""Indicator parameter specifications for code generation.

This module provides structured specifications for all available indicators,
including their parameter signatures, return types, and usage examples.
These specs are used by the code generator to ensure correct parameter names
and types when generating strategy code.
"""

import inspect
from typing import Dict, Any, List
from . import INDICATOR_REGISTRY


def _extract_parameter_spec(indicator_class) -> Dict[str, Any]:
    """Extract parameter specifications from an indicator class.

    Args:
        indicator_class: Indicator class to inspect

    Returns:
        Dict with parameter specifications
    """
    # Get __init__ signature
    sig = inspect.signature(indicator_class.__init__)

    parameters = {}
    for param_name, param in sig.parameters.items():
        # Skip 'self'
        if param_name == 'self':
            continue

        # Extract parameter info
        param_info = {}

        # Get type annotation if available
        if param.annotation != inspect.Parameter.empty:
            param_info['type'] = param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)
        else:
            param_info['type'] = 'Any'

        # Get default value if available
        if param.default != inspect.Parameter.empty:
            param_info['default'] = param.default

        parameters[param_name] = param_info

    return parameters


def _get_return_type_info(indicator_name: str) -> Dict[str, Any]:
    """Get return type information for an indicator.

    Args:
        indicator_name: Name of the indicator

    Returns:
        Dict with return type info
    """
    # Multi-output indicators (return DataFrame)
    multi_output_indicators = {
        'MACD': ['macd', 'signal', 'histogram'],
        'Stochastic': ['k', 'd'],
        'BollingerBands': ['middle', 'upper', 'lower', 'bandwidth'],
        'KeltnerChannels': ['middle', 'upper', 'lower'],
        'ADX': ['adx', 'plus_di', 'minus_di'],
        'DonchianChannels': ['upper', 'middle', 'lower'],
        'MarketRegime': ['trend_strength', 'volatility_regime', 'volume_regime'],
    }

    if indicator_name in multi_output_indicators:
        return {
            'returns': 'DataFrame',
            'columns': multi_output_indicators[indicator_name]
        }
    else:
        return {'returns': 'Series'}


def _generate_usage_example(indicator_name: str, parameters: Dict[str, Any]) -> str:
    """Generate usage example code for an indicator.

    Args:
        indicator_name: Name of the indicator
        parameters: Parameter specifications

    Returns:
        Example code string
    """
    # Build parameter string with defaults
    param_strs = []
    for param_name, param_info in parameters.items():
        if 'default' in param_info:
            default = param_info['default']
            if isinstance(default, str):
                default = f"'{default}'"
            elif isinstance(default, tuple):
                default = str(default)
            param_strs.append(f"{param_name}={default}")

    param_str = ', '.join(param_strs) if param_strs else ''
    return f"self.{indicator_name.lower()} = INDICATOR_REGISTRY['{indicator_name}']({param_str})"


# Generate complete indicator specifications
INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {}

for indicator_name, indicator_class in INDICATOR_REGISTRY.items():
    # Create temporary instance to get category
    try:
        temp_instance = indicator_class()
        category = temp_instance.category
    except:
        category = 'unknown'

    # Extract parameter specifications
    parameters = _extract_parameter_spec(indicator_class)

    # Get return type info
    return_info = _get_return_type_info(indicator_name)

    # Generate usage example
    usage_example = _generate_usage_example(indicator_name, parameters)

    # Build complete spec
    INDICATOR_SPECS[indicator_name] = {
        'class_name': indicator_name,
        'category': category,
        'parameters': parameters,
        **return_info,
        'usage_example': usage_example
    }


# Formatted specification string for inclusion in prompts
INDICATOR_SPECS_STRING = """
## Available Indicators and Their Parameters

CRITICAL: Always use the EXACT parameter names shown below. Never infer or guess parameter names.

"""

# Group by category
categories = {}
for name, spec in INDICATOR_SPECS.items():
    category = spec['category']
    if category not in categories:
        categories[category] = []
    categories[category].append((name, spec))

# Format each category
for category in ['trend', 'momentum', 'volatility', 'regime']:
    if category not in categories:
        continue

    INDICATOR_SPECS_STRING += f"### {category.upper()} Indicators\n\n"

    for indicator_name, spec in sorted(categories[category]):
        INDICATOR_SPECS_STRING += f"**{indicator_name}**\n"

        # Parameters
        if spec['parameters']:
            INDICATOR_SPECS_STRING += "- Parameters:\n"
            for param_name, param_info in spec['parameters'].items():
                param_type = param_info.get('type', 'Any')
                default = param_info.get('default', 'required')
                if default != 'required' and isinstance(default, str):
                    default = f"'{default}'"
                INDICATOR_SPECS_STRING += f"  - `{param_name}`: {param_type} (default: {default})\n"
        else:
            INDICATOR_SPECS_STRING += "- Parameters: None\n"

        # Return type
        if spec['returns'] == 'DataFrame':
            columns = ', '.join(spec['columns'])
            INDICATOR_SPECS_STRING += f"- Returns: DataFrame with columns: {columns}\n"
        else:
            INDICATOR_SPECS_STRING += "- Returns: Series\n"

        # Usage example
        INDICATOR_SPECS_STRING += f"- Example: `{spec['usage_example']}`\n\n"


def get_indicator_spec(indicator_name: str) -> Dict[str, Any]:
    """Get specification for a specific indicator.

    Args:
        indicator_name: Name of the indicator (e.g., 'RSI', 'MACD')

    Returns:
        Indicator specification dict

    Raises:
        KeyError: If indicator not found
    """
    if indicator_name not in INDICATOR_SPECS:
        raise KeyError(
            f"Unknown indicator: {indicator_name}. "
            f"Available indicators: {list(INDICATOR_SPECS.keys())}"
        )
    return INDICATOR_SPECS[indicator_name]


def get_parameter_names(indicator_name: str) -> List[str]:
    """Get list of parameter names for an indicator.

    Args:
        indicator_name: Name of the indicator

    Returns:
        List of parameter names
    """
    spec = get_indicator_spec(indicator_name)
    return list(spec['parameters'].keys())


def validate_parameters(indicator_name: str, params: Dict[str, Any]) -> bool:
    """Validate that parameters match the indicator specification.

    Args:
        indicator_name: Name of the indicator
        params: Parameters to validate

    Returns:
        True if valid

    Raises:
        ValueError: If parameters are invalid
    """
    spec = get_indicator_spec(indicator_name)
    valid_params = set(spec['parameters'].keys())
    provided_params = set(params.keys())

    # Check for unknown parameters
    unknown = provided_params - valid_params
    if unknown:
        raise ValueError(
            f"Unknown parameters for {indicator_name}: {unknown}. "
            f"Valid parameters: {valid_params}"
        )

    return True


__all__ = [
    'INDICATOR_SPECS',
    'INDICATOR_SPECS_STRING',
    'get_indicator_spec',
    'get_parameter_names',
    'validate_parameters',
]
