"""Model registry for tracking trained models with versioning and lineage."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for tracking trained models and their metadata.

    Supports versioning (v1, v2, v3...), lineage tracking, and usage statistics.
    Critical for agent-first architecture where models evolve autonomously.
    """

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
        return {'models': [], 'model_names': {}}

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
        Register a trained model (legacy method - use register_version for versioned models).

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

    def register_version(
        self,
        model_name: str,
        version: str,
        model_path: str,
        model_type: str,
        target: str,
        features_used: List[str],
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        feature_pipeline_path: Optional[str] = None,
        supersedes: Optional[str] = None,
        improvement: Optional[str] = None,
        released_by: str = 'manual',
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a versioned model (e.g., XGBoost_direction_v3).

        Args:
            model_name: Base model name (e.g., 'XGBoost_direction')
            version: Version string (e.g., 'v3')
            model_path: Path to saved model
            model_type: Type of model (classification/regression)
            target: Target variable (e.g., 'next_1h_direction')
            features_used: List of feature categories
            hyperparameters: Model hyperparameters
            metrics: Performance metrics
            feature_pipeline_path: Path to feature pipeline
            supersedes: Previous version this supersedes (e.g., 'XGBoost_direction_v2')
            improvement: Description of improvement from previous version
            released_by: Who/what released this (manual/ml_background_worker)
            metadata: Additional metadata

        Returns:
            Full model ID (e.g., 'XGBoost_direction_v3')
        """
        model_id = f"{model_name}_{version}"

        # Check if this version already exists
        existing = self.get_model_by_id(model_id)
        if existing:
            logger.warning(f"Model {model_id} already exists, updating...")
            self.delete_model(model_id)

        model_entry = {
            'model_id': model_id,
            'model_name': model_name,
            'version': version,
            'model_type': model_type,
            'target': target,
            'features_used': features_used,
            'feature_count': len(features_used),
            'hyperparameters': hyperparameters,
            'model_path': str(model_path),
            'feature_pipeline_path': str(feature_pipeline_path) if feature_pipeline_path else None,
            'metrics': metrics,
            'supersedes': supersedes,
            'improvement': improvement,
            'released_by': released_by,
            'released_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            # Usage tracking
            'usage_stats': {
                'num_strategies_using': 0,
                'avg_sharpe_of_strategies': 0.0,
                'best_strategy': None,
                'total_uses': 0
            }
        }

        self._registry['models'].append(model_entry)

        # Update model_names index for fast lookup
        if model_name not in self._registry['model_names']:
            self._registry['model_names'][model_name] = []
        self._registry['model_names'][model_name].append(version)

        self._save_registry()

        logger.info(f"Registered versioned model: {model_id}")
        if supersedes:
            logger.info(f"  Supersedes: {supersedes}")
            logger.info(f"  Improvement: {improvement}")

        return model_id

    def get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """
        Get model entry by ID.

        Args:
            model_id: Full model ID (e.g., 'XGBoost_direction_v3')

        Returns:
            Model entry dict or None
        """
        for model in self._registry['models']:
            if model['model_id'] == model_id:
                return model
        return None

    def get_version(self, model_name: str, version: str) -> Optional[Dict]:
        """
        Get specific version of a model.

        Args:
            model_name: Base model name
            version: Version string (e.g., 'v3')

        Returns:
            Model entry or None
        """
        model_id = f"{model_name}_{version}"
        return self.get_model_by_id(model_id)

    def get_latest_version(self, model_name: str) -> Optional[Dict]:
        """
        Get latest version of a model.

        Args:
            model_name: Base model name

        Returns:
            Latest model version or None
        """
        models = self.list_versions(model_name)
        if not models:
            return None

        # Sort by version number (extract vN and sort numerically)
        def version_key(m):
            version = m.get('version', 'v0')
            match = re.search(r'v(\d+)', version)
            return int(match.group(1)) if match else 0

        return sorted(models, key=version_key, reverse=True)[0]

    def list_versions(self, model_name: str) -> List[Dict]:
        """
        List all versions of a model.

        Args:
            model_name: Base model name

        Returns:
            List of model entries, sorted by version
        """
        models = [m for m in self._registry['models'] if m.get('model_name') == model_name]

        # Sort by version
        def version_key(m):
            version = m.get('version', 'v0')
            match = re.search(r'v(\d+)', version)
            return int(match.group(1)) if match else 0

        return sorted(models, key=version_key)

    def list_all_models(self) -> List[str]:
        """
        List all model names (base names, not including versions).

        Returns:
            List of unique model names
        """
        model_names = set()
        for model in self._registry['models']:
            model_name = model.get('model_name')
            if model_name:
                model_names.add(model_name)
        return sorted(list(model_names))

    def update_model_stats(self, model_id: str, strategy_results: Dict[str, Any]) -> None:
        """
        Update usage statistics for a model based on strategy performance.

        Args:
            model_id: Model ID
            strategy_results: Strategy backtest results
        """
        model = self.get_model_by_id(model_id)
        if not model:
            logger.warning(f"Model {model_id} not found for stats update")
            return

        # Update usage count
        model['usage_stats']['num_strategies_using'] += 1
        model['usage_stats']['total_uses'] += 1

        # Update average Sharpe
        sharpe = strategy_results.get('metrics', {}).get('sharpe_ratio')
        if sharpe is not None:
            current_avg = model['usage_stats']['avg_sharpe_of_strategies']
            n = model['usage_stats']['num_strategies_using']
            new_avg = ((current_avg * (n - 1)) + sharpe) / n
            model['usage_stats']['avg_sharpe_of_strategies'] = new_avg

            # Update best strategy
            best_sharpe = model['usage_stats'].get('best_strategy_sharpe', -999)
            if sharpe > best_sharpe:
                model['usage_stats']['best_strategy'] = strategy_results.get('strategy_name')
                model['usage_stats']['best_strategy_sharpe'] = sharpe

        self._save_registry()
        logger.info(f"Updated stats for {model_id}: {model['usage_stats']['num_strategies_using']} strategies using it")

    def get_usage_stats(self, model_id: str) -> Optional[Dict]:
        """
        Get usage statistics for a model.

        Args:
            model_id: Model ID

        Returns:
            Usage stats dict or None
        """
        model = self.get_model_by_id(model_id)
        if model:
            return model.get('usage_stats', {})
        return None

    def get_models_by_usage(self, min_uses: int = 1) -> List[Dict]:
        """
        Get models sorted by usage count.

        Args:
            min_uses: Minimum number of uses

        Returns:
            List of models sorted by usage (descending)
        """
        models = [
            m for m in self._registry['models']
            if m.get('usage_stats', {}).get('num_strategies_using', 0) >= min_uses
        ]

        return sorted(
            models,
            key=lambda m: m.get('usage_stats', {}).get('num_strategies_using', 0),
            reverse=True
        )

    def get_model(self, model_id: str) -> Optional[Dict]:
        """Alias for get_model_by_id (for backward compatibility)."""
        return self.get_model_by_id(model_id)

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
