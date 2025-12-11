"""Base classes for feature engineering."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Feature categories."""
    PRICE = "price"
    TECHNICAL = "technical"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TEMPORAL = "temporal"
    CROSS_SECTIONAL = "cross_sectional"


class BaseFeature(ABC):
    """Base class for feature extractors."""

    def __init__(self, name: str, category: FeatureCategory):
        """
        Initialize feature extractor.

        Args:
            name: Feature name
            category: Feature category
        """
        self.name = name
        self.category = category

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features from data.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with computed features
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data format.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If data format is invalid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Required: {required_columns}"
            )

        if len(data) < 2:
            raise ValueError("Data must have at least 2 rows")

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names this extractor produces.

        Returns:
            List of feature names
        """
        return [self.name]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', category={self.category.value})"


class FeatureSet:
    """Container for multiple related features."""

    def __init__(self, name: str, features: List[BaseFeature]):
        """
        Initialize feature set.

        Args:
            name: Feature set name
            features: List of feature extractors
        """
        self.name = name
        self.features = features

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features in the set.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with all computed features
        """
        logger.info(f"Computing feature set: {self.name} ({len(self.features)} features)")

        result_df = pd.DataFrame(index=data.index)

        for feature in self.features:
            try:
                feature_data = feature.compute(data)
                result_df = pd.concat([result_df, feature_data], axis=1)
                logger.debug(f"  ✓ Computed {feature.name}")
            except Exception as e:
                logger.error(f"  ✗ Failed to compute {feature.name}: {e}")
                raise

        logger.info(f"Feature set {self.name} computed: {len(result_df.columns)} features")
        return result_df

    def get_feature_names(self) -> List[str]:
        """Get all feature names in this set."""
        names = []
        for feature in self.features:
            names.extend(feature.get_feature_names())
        return names

    def __len__(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return f"FeatureSet(name='{self.name}', features={len(self.features)})"
