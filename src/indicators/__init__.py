"""Indicator pool for OmniAlpha strategy generation."""

from .base import BaseIndicator

# Trend indicators
from .trend import EMA, SMA, MACD, ADX, EMASlope

# Momentum indicators
from .momentum import RSI, Stochastic, MFI, ROC, WilliamsR

# Volatility indicators
from .volatility import (
    ATR, BollingerBands, KeltnerChannels,
    HistoricalVolatility, DonchianChannels
)

# Regime indicators
from .regime import (
    VolumeZScore, HurstExponent, MarketRegime,
    TrendStrength, VolatilityRegime
)


# Indicator registry for strategy generation
INDICATOR_REGISTRY = {
    # Trend indicators
    'EMA': EMA,
    'SMA': SMA,
    'MACD': MACD,
    'ADX': ADX,
    'EMASlope': EMASlope,

    # Momentum indicators
    'RSI': RSI,
    'Stochastic': Stochastic,
    'MFI': MFI,
    'ROC': ROC,
    'WilliamsR': WilliamsR,

    # Volatility indicators
    'ATR': ATR,
    'BollingerBands': BollingerBands,
    'KeltnerChannels': KeltnerChannels,
    'HistoricalVolatility': HistoricalVolatility,
    'DonchianChannels': DonchianChannels,

    # Regime indicators
    'VolumeZScore': VolumeZScore,
    'HurstExponent': HurstExponent,
    'MarketRegime': MarketRegime,
    'TrendStrength': TrendStrength,
    'VolatilityRegime': VolatilityRegime,
}


def get_indicator_by_name(name: str, **params):
    """Get indicator instance by name.

    Args:
        name: Indicator name (e.g., 'RSI', 'EMA')
        **params: Indicator parameters

    Returns:
        Indicator instance

    Raises:
        ValueError: If indicator name not found
    """
    if name not in INDICATOR_REGISTRY:
        raise ValueError(
            f"Unknown indicator: {name}. "
            f"Available indicators: {list(INDICATOR_REGISTRY.keys())}"
        )

    indicator_class = INDICATOR_REGISTRY[name]
    return indicator_class(**params)


def get_indicators_by_category(category: str):
    """Get all indicators in a category.

    Args:
        category: Category name ('trend', 'momentum', 'volatility', 'regime')

    Returns:
        Dict of indicator name -> class for the category
    """
    valid_categories = ['trend', 'momentum', 'volatility', 'regime']
    if category not in valid_categories:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Valid categories: {valid_categories}"
        )

    return {
        name: cls for name, cls in INDICATOR_REGISTRY.items()
        if cls().__class__.__base__ != object and
        hasattr(cls, '__init__') and
        (lambda c: c().category == category if hasattr(c(), 'category') else False)(cls)
    }


def list_indicators():
    """List all available indicators.

    Returns:
        Dict mapping category -> list of indicator names
    """
    indicators_by_category = {
        'trend': [],
        'momentum': [],
        'volatility': [],
        'regime': []
    }

    for name, cls in INDICATOR_REGISTRY.items():
        # Create temporary instance to get category
        try:
            instance = cls()
            category = instance.category
            if category in indicators_by_category:
                indicators_by_category[category].append(name)
        except:
            pass

    return indicators_by_category


__all__ = [
    # Base
    'BaseIndicator',

    # Trend
    'EMA', 'SMA', 'MACD', 'ADX', 'EMASlope',

    # Momentum
    'RSI', 'Stochastic', 'MFI', 'ROC', 'WilliamsR',

    # Volatility
    'ATR', 'BollingerBands', 'KeltnerChannels',
    'HistoricalVolatility', 'DonchianChannels',

    # Regime
    'VolumeZScore', 'HurstExponent', 'MarketRegime',
    'TrendStrength', 'VolatilityRegime',

    # Utilities
    'INDICATOR_REGISTRY',
    'get_indicator_by_name',
    'get_indicators_by_category',
    'list_indicators',
]
