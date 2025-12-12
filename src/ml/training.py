"""Model training pipeline."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from .base_model import BaseModel, ModelType
from .model_registry import ModelRegistry
from src.features.pipeline import FeaturePipeline, create_target_variable

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Pipeline for training ML models."""

    def __init__(
        self,
        model: BaseModel,
        feature_pipeline: FeaturePipeline,
        registry: Optional[ModelRegistry] = None,
        models_dir: str = "models"
    ):
        """
        Initialize model trainer.

        Args:
            model: ML model to train
            feature_pipeline: Feature generation pipeline
            registry: Model registry (optional)
            models_dir: Directory to save models
        """
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.registry = registry or ModelRegistry()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_type: str = 'binary',
        horizon: int = 1,
        threshold: float = 0.0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for training.

        Args:
            data: OHLCV DataFrame
            target_type: Type of target variable
            horizon: Prediction horizon
            threshold: Classification threshold
            train_ratio: Training set ratio
            val_ratio: Validation set ratio (test = 1 - train - val)

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Preparing data for training")

        # Create target variable
        target = create_target_variable(data, target_type, horizon, threshold)

        # Generate features
        features, aligned_target = self.feature_pipeline.fit_transform(data, target)

        # Remove samples with NaN target
        valid_idx = aligned_target.notna()
        features = features[valid_idx]
        aligned_target = aligned_target[valid_idx]

        logger.info(f"Valid samples after alignment: {len(features)}")

        # Split data (chronological split for time series)
        n = len(features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train = features.iloc[:train_end]
        y_train = aligned_target.iloc[:train_end]

        X_val = features.iloc[train_end:val_end]
        y_val = aligned_target.iloc[train_end:val_end]

        X_test = features.iloc[val_end:]
        y_test = aligned_target.iloc[val_end:]

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Check class distribution for classification
        if self.model.model_type == ModelType.CLASSIFICATION:
            train_dist = y_train.value_counts(normalize=True)
            logger.info(f"Training set class distribution:\n{train_dist}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model.name} model")

        metrics = self.model.fit(X_train, y_train, X_val, y_val)

        logger.info("Training completed")
        return metrics

    def train_and_evaluate(
        self,
        data: pd.DataFrame,
        target_type: str = 'binary',
        horizon: int = 1,
        threshold: float = 0.0,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        save_model: bool = True,
        register_model: bool = True
    ) -> Dict[str, Any]:
        """
        Complete training pipeline: prepare data, train, evaluate, save.

        Args:
            data: OHLCV DataFrame
            target_type: Type of target variable
            horizon: Prediction horizon
            threshold: Classification threshold
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            save_model: Whether to save the model
            register_model: Whether to register in registry

        Returns:
            Dict with metrics and model info
        """
        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(
            data, target_type, horizon, threshold, train_ratio, val_ratio
        )

        # Train model
        metrics = self.train(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        test_metrics = self.model.evaluate(X_test, y_test)
        metrics['test'] = test_metrics

        logger.info(f"Test set metrics: {test_metrics}")

        # Get feature importance
        feature_importance = self.model.get_feature_importance()
        if feature_importance is not None:
            logger.info(f"Top 10 features:\n{feature_importance.head(10)}")

        # Save model and pipeline
        model_path = None
        pipeline_path = None
        model_id = None

        if save_model:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            model_path = self.models_dir / f"{self.model.name}_{timestamp}.joblib"
            pipeline_path = self.models_dir / f"{self.model.name}_{timestamp}_pipeline.joblib"

            self.model.save(str(model_path))
            self.feature_pipeline.save(str(pipeline_path))

            logger.info(f"Saved model to {model_path}")
            logger.info(f"Saved pipeline to {pipeline_path}")

        # Register model
        if register_model and model_path:
            model_id = self.registry.register_model(
                model_name=self.model.name,
                model_path=str(model_path),
                model_type=self.model.model_type.value,
                metrics=metrics,
                feature_pipeline_path=str(pipeline_path),
                metadata={
                    'target_type': target_type,
                    'horizon': horizon,
                    'num_features': len(X_train.columns),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                }
            )

            logger.info(f"Registered model with ID: {model_id}")

        return {
            'metrics': metrics,
            'model_id': model_id,
            'model_path': str(model_path) if model_path else None,
            'pipeline_path': str(pipeline_path) if pipeline_path else None,
            'feature_importance': feature_importance,
        }

    def cross_validate(
        self,
        data: pd.DataFrame,
        target_type: str = 'binary',
        horizon: int = 1,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.

        Args:
            data: OHLCV DataFrame
            target_type: Type of target variable
            horizon: Prediction horizon
            n_splits: Number of CV splits

        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {n_splits}-fold time series cross-validation")

        # Create target
        target = create_target_variable(data, target_type, horizon)

        # Generate features
        features, aligned_target = self.feature_pipeline.fit_transform(data, target)

        # Remove NaN
        valid_idx = aligned_target.notna()
        features = features[valid_idx]
        aligned_target = aligned_target[valid_idx]

        # Time series split
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(features), 1):
            logger.info(f"Training fold {fold}/{n_splits}")

            X_train = features.iloc[train_idx]
            y_train = aligned_target.iloc[train_idx]
            X_val = features.iloc[val_idx]
            y_val = aligned_target.iloc[val_idx]

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate
            val_metrics = self.model.evaluate(X_val, y_val)
            cv_results.append(val_metrics)

            logger.info(f"Fold {fold} validation metrics: {val_metrics}")

        # Aggregate results
        aggregated_metrics = {}
        for metric_name in cv_results[0].keys():
            values = [result[metric_name] for result in cv_results]
            aggregated_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

        logger.info("Cross-validation completed")
        return {
            'fold_results': cv_results,
            'aggregated_metrics': aggregated_metrics,
        }
