"""Base indicator interface for OmniAlpha."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class BaseIndicator(ABC):
    """Abstract base class for all technical indicators."""

    def __init__(self, name: str, category: str, params: Optional[Dict[str, Any]] = None):
        """Initialize indicator.

        Args:
            name: Indicator name (e.g., 'EMA', 'RSI')
            category: Category ('trend', 'momentum', 'volatility', 'regime')
            params: Indicator parameters (e.g., {'period': 14})
        """
        self.name = name
        self.category = category
        self.params = params or {}

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate indicator values.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume

        Returns:
            Series with indicator values
        """
        pass

    def validate_data(self, data: pd.DataFrame):
        """Validate input data format.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If data format is invalid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in data.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Data must have: {required_columns}"
            )

        if len(data) == 0:
            raise ValueError("Data is empty")

    def __repr__(self) -> str:
        """String representation of indicator."""
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})" if params_str else self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert indicator to dictionary representation.

        Returns:
            Dict with indicator metadata
        """
        return {
            'name': self.name,
            'category': self.category,
            'params': self.params
        }
