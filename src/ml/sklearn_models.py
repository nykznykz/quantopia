"""Scikit-learn based models (XGBoost, RandomForest, LightGBM)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import joblib
from pathlib import Path

from .base_model import BaseModel, ModelType

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost model for classification or regression."""

    def __init__(
        self,
        model_type: ModelType = ModelType.CLASSIFICATION,
        name: str = "XGBoost",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize XGBoost model.

        Args:
            model_type: Model type (classification/regression)
            name: Model name
            params: XGBoost hyperparameters
        """
        super().__init__(model_type, name, params)

        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        default_params.update(self.params)
        self.params = default_params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Train XGBoost model."""
        import xgboost as xgb

        logger.info(f"Training XGBoost {self.model_type.value} model")
        logger.info(f"Training samples: {len(X_train)}, Features: {len(X_train.columns)}")

        if self.model_type == ModelType.CLASSIFICATION:
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)

        # Prepare evaluation set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        # Train model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False
        )

        self._is_fitted = True

        # Evaluate
        train_metrics = self.evaluate(X_train, y_train)
        metrics = {'train': train_metrics}

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics['validation'] = val_metrics
            logger.info(f"Validation metrics: {val_metrics}")

        logger.info(f"✓ XGBoost model trained successfully")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if self.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba only available for classification models")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'name': self.name,
            'params': self.params,
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved XGBoost model to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.name = model_data['name']
        self.params = model_data['params']
        self._is_fitted = True

        logger.info(f"Loaded XGBoost model from {path}")

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        return pd.Series(importance, index=feature_names).sort_values(ascending=False)


class RandomForestModel(BaseModel):
    """Random Forest model for classification or regression."""

    def __init__(
        self,
        model_type: ModelType = ModelType.CLASSIFICATION,
        name: str = "RandomForest",
        params: Optional[Dict[str, Any]] = None
    ):
        """Initialize Random Forest model."""
        super().__init__(model_type, name, params)

        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
        }
        default_params.update(self.params)
        self.params = default_params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Train Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        logger.info(f"Training RandomForest {self.model_type.value} model")

        if self.model_type == ModelType.CLASSIFICATION:
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)

        self.model.fit(X_train, y_train)
        self._is_fitted = True

        # Evaluate
        train_metrics = self.evaluate(X_train, y_train)
        metrics = {'train': train_metrics}

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics['validation'] = val_metrics
            logger.info(f"Validation metrics: {val_metrics}")

        logger.info(f"✓ RandomForest model trained successfully")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if self.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba only available for classification models")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'name': self.name,
            'params': self.params,
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved RandomForest model to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.name = model_data['name']
        self.params = model_data['params']
        self._is_fitted = True

        logger.info(f"Loaded RandomForest model from {path}")

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        return pd.Series(importance, index=feature_names).sort_values(ascending=False)


class LightGBMModel(BaseModel):
    """LightGBM model for classification or regression."""

    def __init__(
        self,
        model_type: ModelType = ModelType.CLASSIFICATION,
        name: str = "LightGBM",
        params: Optional[Dict[str, Any]] = None
    ):
        """Initialize LightGBM model."""
        super().__init__(model_type, name, params)

        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
        }
        default_params.update(self.params)
        self.params = default_params

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Train LightGBM model."""
        import lightgbm as lgb

        logger.info(f"Training LightGBM {self.model_type.value} model")

        if self.model_type == ModelType.CLASSIFICATION:
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)

        # Prepare evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set
        )

        self._is_fitted = True

        # Evaluate
        train_metrics = self.evaluate(X_train, y_train)
        metrics = {'train': train_metrics}

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics['validation'] = val_metrics
            logger.info(f"Validation metrics: {val_metrics}")

        logger.info(f"✓ LightGBM model trained successfully")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if self.model_type != ModelType.CLASSIFICATION:
            raise ValueError("predict_proba only available for classification models")
        return self.model.predict_proba(X)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'name': self.name,
            'params': self.params,
        }

        joblib.dump(model_data, path)
        logger.info(f"Saved LightGBM model to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.name = model_data['name']
        self.params = model_data['params']
        self._is_fitted = True

        logger.info(f"Loaded LightGBM model from {path}")

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        importance = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
