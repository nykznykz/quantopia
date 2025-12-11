"""Base class for ML models."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTICLASS = "multiclass"


class BaseModel(ABC):
    """Base class for ML models."""

    def __init__(
        self,
        model_type: ModelType,
        name: str,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model.

        Args:
            model_type: Type of model (regression/classification)
            name: Model name
            params: Model hyperparameters
        """
        self.model_type = model_type
        self.name = name
        self.params = params or {}
        self.model = None
        self._is_fitted = False

    @abstractmethod
    def fit(
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
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dict with training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (for classification models).

        Args:
            X: Features

        Returns:
            Class probabilities

        Raises:
            NotImplementedError: If model doesn't support probabilities
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support probability predictions"
        )

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        pass

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores.

        Returns:
            Series with feature importance (if supported)
        """
        return None

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True targets

        Returns:
            Dict with evaluation metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")

        predictions = self.predict(X)

        if self.model_type == ModelType.REGRESSION:
            return self._evaluate_regression(y, predictions)
        else:
            return self._evaluate_classification(y, predictions)

    def _evaluate_regression(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate regression model."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)

        # Direction accuracy (for returns)
        direction_accuracy = np.mean(
            (y_true > 0).values == (y_pred > 0)
        )

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }

    def _evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate classification model."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score
        )

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
        }

        # For binary classification
        if len(np.unique(y_true)) == 2:
            metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)

            # ROC-AUC if probabilities available
            try:
                y_proba = self.predict_proba(pd.DataFrame(y_true.index))[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                pass

        # For multiclass
        else:
            metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return metrics

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', type={self.model_type.value}, {fitted_str})"
