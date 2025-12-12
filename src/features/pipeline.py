"""Feature generation pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import joblib
from pathlib import Path

from .base import BaseFeature, FeatureSet
from .technical import TechnicalFeatures

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Pipeline for feature generation and preprocessing."""

    def __init__(
        self,
        technical_configs: Optional[Dict] = None,
        normalize: bool = True,
        fill_method: str = 'forward'
    ):
        """
        Initialize feature pipeline.

        Args:
            technical_configs: Configuration for technical indicators
            normalize: Whether to normalize features
            fill_method: Method for filling missing values ('forward', 'drop', 'zero')
        """
        self.technical_features = TechnicalFeatures(technical_configs)
        self.normalize = normalize
        self.fill_method = fill_method

        # Fitted parameters (for transform)
        self._feature_means = None
        self._feature_stds = None
        self._feature_names = None
        self._is_fitted = False

    def fit_transform(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit pipeline on data and transform it.

        Args:
            data: OHLCV DataFrame
            target: Optional target variable (will be aligned with features)

        Returns:
            Tuple of (features_df, aligned_target)
        """
        logger.info("Fitting and transforming feature pipeline")

        # Generate features
        features = self.technical_features.compute_all(data)

        # Handle missing values
        features = self._handle_missing_values(features)

        # Fit normalization parameters
        if self.normalize:
            self._feature_means = features.mean()
            self._feature_stds = features.std().replace(0, 1)  # Avoid division by zero
            features = (features - self._feature_means) / self._feature_stds
            logger.info("Normalized features using fitted mean/std")

        self._feature_names = features.columns.tolist()
        self._is_fitted = True

        logger.info(f"Pipeline fitted with {len(self._feature_names)} features")

        # Align target if provided
        aligned_target = None
        if target is not None:
            aligned_target = target.loc[features.index]
            logger.info(f"Aligned target: {len(aligned_target)} samples")

        return features, aligned_target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.

        Args:
            data: OHLCV DataFrame

        Returns:
            Transformed features

        Raises:
            RuntimeError: If pipeline not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform. Use fit_transform first.")

        logger.info("Transforming data with fitted pipeline")

        # Generate features
        features = self.technical_features.compute_all(data)

        # Handle missing values
        features = self._handle_missing_values(features)

        # Apply fitted normalization
        if self.normalize:
            features = (features - self._feature_means) / self._feature_stds

        # Ensure same features as training
        missing_features = set(self._feature_names) - set(features.columns)
        if missing_features:
            logger.warning(f"Missing features in transform: {missing_features}")
            for feature in missing_features:
                features[feature] = 0

        # Reorder to match training
        features = features[self._feature_names]

        logger.info(f"Transformed {len(features)} samples with {len(features.columns)} features")

        return features

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to fill_method."""
        if self.fill_method == 'forward':
            features = features.fillna(method='ffill')
            features = features.fillna(0)  # Fill any remaining NaNs with 0
        elif self.fill_method == 'drop':
            features = features.dropna()
        elif self.fill_method == 'zero':
            features = features.fillna(0)
        else:
            raise ValueError(f"Unknown fill_method: {self.fill_method}")

        return features

    def save(self, path: str) -> None:
        """
        Save fitted pipeline to disk.

        Args:
            path: Path to save pipeline
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline")

        pipeline_state = {
            'feature_means': self._feature_means,
            'feature_stds': self._feature_stds,
            'feature_names': self._feature_names,
            'normalize': self.normalize,
            'fill_method': self.fill_method,
            'technical_configs': self.technical_features.indicator_configs,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_state, path)
        logger.info(f"Saved pipeline to {path}")

    @classmethod
    def load(cls, path: str) -> 'FeaturePipeline':
        """
        Load fitted pipeline from disk.

        Args:
            path: Path to load pipeline from

        Returns:
            Loaded pipeline
        """
        pipeline_state = joblib.load(path)

        pipeline = cls(
            technical_configs=pipeline_state['technical_configs'],
            normalize=pipeline_state['normalize'],
            fill_method=pipeline_state['fill_method']
        )

        pipeline._feature_means = pipeline_state['feature_means']
        pipeline._feature_stds = pipeline_state['feature_stds']
        pipeline._feature_names = pipeline_state['feature_names']
        pipeline._is_fitted = True

        logger.info(f"Loaded pipeline from {path}")
        return pipeline

    def get_feature_info(self) -> Dict:
        """
        Get information about features.

        Returns:
            Dict with feature information
        """
        if not self._is_fitted:
            return {'fitted': False}

        return {
            'fitted': True,
            'num_features': len(self._feature_names),
            'feature_names': self._feature_names,
            'normalized': self.normalize,
        }


def create_target_variable(
    data: pd.DataFrame,
    target_type: str = 'return',
    horizon: int = 1,
    threshold: float = 0.0
) -> pd.Series:
    """
    Create target variable for ML models.

    Args:
        data: OHLCV DataFrame
        target_type: Type of target ('return', 'direction', 'binary')
        horizon: Prediction horizon in periods
        threshold: Threshold for binary classification (default: 0)

    Returns:
        Target variable series
    """
    if target_type == 'return':
        # Predict future return
        target = data['close'].pct_change(horizon).shift(-horizon)

    elif target_type == 'direction':
        # Predict direction: -1 (down), 0 (neutral), 1 (up)
        future_return = data['close'].pct_change(horizon).shift(-horizon)
        target = pd.Series(0, index=data.index)
        target[future_return > threshold] = 1
        target[future_return < -threshold] = -1

    elif target_type == 'binary':
        # Binary: 0 (down), 1 (up)
        future_return = data['close'].pct_change(horizon).shift(-horizon)
        target = (future_return > threshold).astype(int)

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    logger.info(
        f"Created {target_type} target with horizon={horizon}, "
        f"valid samples: {target.notna().sum()}/{len(target)}"
    )

    return target
