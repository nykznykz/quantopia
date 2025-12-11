"""Model registry for tracking trained models."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for tracking trained models and their metadata."""

    def __init__(self, registry_dir: str = "models/registry"):
        """
        Initialize model registry.

        Args:
            registry_dir: Directory to store registry data
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"

        self._registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': []}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self._registry, f, indent=2, default=str)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        metrics: Dict[str, Any],
        feature_pipeline_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a trained model.

        Args:
            model_name: Name of the model
            model_path: Path to saved model
            model_type: Type of model (classification/regression)
            metrics: Training and validation metrics
            feature_pipeline_path: Path to feature pipeline (optional)
            metadata: Additional metadata

        Returns:
            Model ID
        """
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_entry = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'model_path': str(model_path),
            'feature_pipeline_path': str(feature_pipeline_path) if feature_pipeline_path else None,
            'metrics': metrics,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
        }

        self._registry['models'].append(model_entry)
        self._save_registry()

        logger.info(f"Registered model: {model_id}")
        return model_id

    def get_model(self, model_id: str) -> Optional[Dict]:
        """
        Get model entry by ID.

        Args:
            model_id: Model ID

        Returns:
            Model entry dict or None
        """
        for model in self._registry['models']:
            if model['model_id'] == model_id:
                return model
        return None

    def get_latest_model(self, model_name: Optional[str] = None) -> Optional[Dict]:
        """
        Get the most recently registered model.

        Args:
            model_name: Filter by model name (optional)

        Returns:
            Latest model entry or None
        """
        models = self._registry['models']

        if model_name:
            models = [m for m in models if m['model_name'] == model_name]

        if not models:
            return None

        # Sort by created_at and return latest
        return sorted(models, key=lambda x: x['created_at'], reverse=True)[0]

    def list_models(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[Dict]:
        """
        List registered models.

        Args:
            model_name: Filter by model name
            model_type: Filter by model type

        Returns:
            List of model entries
        """
        models = self._registry['models']

        if model_name:
            models = [m for m in models if m['model_name'] == model_name]

        if model_type:
            models = [m for m in models if m['model_type'] == model_type]

        return sorted(models, key=lambda x: x['created_at'], reverse=True)

    def get_best_model(
        self,
        model_name: Optional[str] = None,
        metric: str = 'accuracy'
    ) -> Optional[Dict]:
        """
        Get best model by metric.

        Args:
            model_name: Filter by model name
            metric: Metric to use for ranking

        Returns:
            Best model entry or None
        """
        models = self.list_models(model_name=model_name)

        if not models:
            return None

        # Find model with best validation metric
        best_model = None
        best_score = -float('inf')

        for model in models:
            val_metrics = model['metrics'].get('validation', {})
            if metric in val_metrics:
                score = val_metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model = model

        return best_model

    def delete_model(self, model_id: str) -> bool:
        """
        Delete model from registry.

        Args:
            model_id: Model ID

        Returns:
            True if deleted, False if not found
        """
        initial_count = len(self._registry['models'])
        self._registry['models'] = [
            m for m in self._registry['models']
            if m['model_id'] != model_id
        ]

        if len(self._registry['models']) < initial_count:
            self._save_registry()
            logger.info(f"Deleted model: {model_id}")
            return True

        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary statistics."""
        models = self._registry['models']

        model_names = set(m['model_name'] for m in models)
        model_types = {}

        for model in models:
            model_type = model['model_type']
            model_types[model_type] = model_types.get(model_type, 0) + 1

        return {
            'total_models': len(models),
            'unique_model_names': len(model_names),
            'model_types': model_types,
            'model_names': list(model_names),
        }
