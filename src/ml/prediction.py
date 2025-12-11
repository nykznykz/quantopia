"""Model prediction and inference."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from .base_model import BaseModel
from .sklearn_models import XGBoostModel, RandomForestModel, LightGBMModel
from .model_registry import ModelRegistry
from src.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handle model predictions and inference."""

    def __init__(
        self,
        model: Optional[BaseModel] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        registry: Optional[ModelRegistry] = None
    ):
        """
        Initialize model predictor.

        Args:
            model: Loaded ML model (optional)
            feature_pipeline: Feature pipeline (optional)
            registry: Model registry (optional)
        """
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.registry = registry or ModelRegistry()

    @classmethod
    def from_registry(
        cls,
        model_id: str,
        registry: Optional[ModelRegistry] = None
    ) -> 'ModelPredictor':
        """
        Load model and pipeline from registry.

        Args:
            model_id: Model ID from registry
            registry: Model registry

        Returns:
            ModelPredictor instance
        """
        registry = registry or ModelRegistry()
        model_entry = registry.get_model(model_id)

        if model_entry is None:
            raise ValueError(f"Model {model_id} not found in registry")

        # Load model
        model_path = model_entry['model_path']
        model_name = model_entry['model_name']

        # Determine model class
        if 'XGBoost' in model_name:
            model = XGBoostModel()
        elif 'RandomForest' in model_name:
            model = RandomForestModel()
        elif 'LightGBM' in model_name:
            model = LightGBMModel()
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        model.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Load feature pipeline if available
        feature_pipeline = None
        pipeline_path = model_entry.get('feature_pipeline_path')
        if pipeline_path and Path(pipeline_path).exists():
            feature_pipeline = FeaturePipeline.load(pipeline_path)
            logger.info(f"Loaded feature pipeline from {pipeline_path}")

        return cls(model, feature_pipeline, registry)

    @classmethod
    def from_files(
        cls,
        model_path: str,
        pipeline_path: Optional[str] = None,
        model_type: str = 'xgboost'
    ) -> 'ModelPredictor':
        """
        Load model and pipeline from files.

        Args:
            model_path: Path to model file
            pipeline_path: Path to pipeline file (optional)
            model_type: Type of model ('xgboost', 'randomforest', 'lightgbm')

        Returns:
            ModelPredictor instance
        """
        # Load model
        if model_type.lower() == 'xgboost':
            model = XGBoostModel()
        elif model_type.lower() == 'randomforest':
            model = RandomForestModel()
        elif model_type.lower() == 'lightgbm':
            model = LightGBMModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.load(model_path)

        # Load pipeline
        feature_pipeline = None
        if pipeline_path:
            feature_pipeline = FeaturePipeline.load(pipeline_path)

        return cls(model, feature_pipeline)

    def predict(
        self,
        data: pd.DataFrame,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            data: OHLCV DataFrame or pre-computed features
            return_proba: Whether to return probabilities (classification only)

        Returns:
            Predictions array
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        # Generate features if pipeline available and data looks like OHLCV
        if self.feature_pipeline is not None and 'close' in data.columns:
            logger.info("Generating features from OHLCV data")
            features = self.feature_pipeline.transform(data)
        else:
            features = data

        # Make predictions
        if return_proba and hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(features)
        else:
            predictions = self.model.predict(features)

        logger.info(f"Generated {len(predictions)} predictions")
        return predictions

    def predict_with_confidence(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions with confidence scores.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with predictions and confidence scores
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        # Generate features
        if self.feature_pipeline is not None and 'close' in data.columns:
            features = self.feature_pipeline.transform(data)
        else:
            features = data

        # Get predictions
        predictions = self.model.predict(features)

        result = pd.DataFrame(index=features.index)
        result['prediction'] = predictions

        # Add confidence scores if classification
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
            result['confidence'] = probabilities.max(axis=1)

            # Add probabilities for each class
            for i in range(probabilities.shape[1]):
                result[f'prob_class_{i}'] = probabilities[:, i]

        return result

    def get_signals(
        self,
        data: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.Series:
        """
        Generate trading signals from predictions.

        Args:
            data: OHLCV DataFrame
            threshold: Confidence threshold for signals

        Returns:
            Series with signals (1: buy, -1: sell, 0: hold)
        """
        predictions_df = self.predict_with_confidence(data)

        signals = pd.Series(0, index=predictions_df.index)

        if 'confidence' in predictions_df.columns:
            # For classification: use confidence threshold
            high_confidence = predictions_df['confidence'] >= threshold

            signals[high_confidence & (predictions_df['prediction'] == 1)] = 1
            signals[high_confidence & (predictions_df['prediction'] == 0)] = -1
        else:
            # For regression: use prediction value directly
            signals[predictions_df['prediction'] > threshold] = 1
            signals[predictions_df['prediction'] < -threshold] = -1

        logger.info(
            f"Generated signals - Buy: {(signals == 1).sum()}, "
            f"Sell: {(signals == -1).sum()}, Hold: {(signals == 0).sum()}"
        )

        return signals

    def backtest_predictions(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        actual: pd.Series
    ) -> Dict[str, Any]:
        """
        Backtest prediction quality.

        Args:
            data: OHLCV DataFrame
            predictions: Model predictions
            actual: Actual outcomes

        Returns:
            Backtest metrics
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        # Align predictions with actuals
        predictions_series = pd.Series(predictions, index=data.index[:len(predictions)])
        aligned_actual = actual.loc[predictions_series.index]

        # Calculate metrics
        metrics = self.model.evaluate(
            pd.DataFrame(predictions_series),
            aligned_actual
        )

        # Add prediction statistics
        metrics['prediction_mean'] = predictions_series.mean()
        metrics['prediction_std'] = predictions_series.std()

        if hasattr(self.model, 'predict_proba'):
            # Classification-specific metrics
            correct_predictions = (predictions_series == aligned_actual).sum()
            metrics['num_correct'] = int(correct_predictions)
            metrics['num_total'] = len(predictions_series)

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """
        Get feature importance from model.

        Args:
            top_n: Number of top features to return

        Returns:
            Series with feature importance scores
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        importance = self.model.get_feature_importance()

        if importance is None:
            logger.warning("Model does not support feature importance")
            return pd.Series()

        return importance.head(top_n)
