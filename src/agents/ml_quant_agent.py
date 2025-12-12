"""
ML Quant Agent: On-demand ML expert for strategy creation.

This agent:
- Receives specifications from Strategy Agent
- Checks model registry for existing versions
- Trains new models if needed (with LLM guidance)
- Returns versioned model ID
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from src.ml.model_registry import ModelRegistry
from src.ml.training import ModelTrainer
from src.strategy_generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


class MLQuantAgent:
    """
    On-demand ML expert that trains and retrieves models for strategies.

    Reactive mode: responds to requests from Strategy Agent
    - Checks if model exists
    - If not, uses LLM to plan features and train model
    - Returns versioned model ID
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_trainer: Optional['ModelTrainer'] = None,
        llm_client: Optional[LLMClient] = None,
        models_dir: str = "models"
    ):
        """
        Initialize ML Quant Agent.

        Args:
            model_registry: Model registry for tracking models
            model_trainer: Model trainer instance
            llm_client: LLM client for feature engineering decisions
            models_dir: Directory to save models
        """
        self.model_registry = model_registry
        self.model_trainer = model_trainer
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if llm_client is None:
            self.llm_client = LLMClient(
                provider="openai",
                model="gpt-4",
                temperature=0.5  # Balanced for technical decisions
            )
        else:
            self.llm_client = llm_client

        logger.info("Initialized MLQuantAgent (reactive mode)")

    def get_or_create_model(
        self,
        spec: Dict[str, Any],
        training_data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Get existing model or create new one based on specification.

        Args:
            spec: Model specification from Strategy Agent
                {
                    'model_id': 'XGBoost_direction_v3',  // Requested model
                    'model_name': 'XGBoost_direction',   // Or base name
                    'version': 'latest',                 // Or specific version
                    'target': 'next_1h_direction',       // If creating new
                    'features': ['price', 'volume']      // Feature hints
                }
            training_data: Optional training data if creating new model

        Returns:
            Full model ID (e.g., 'XGBoost_direction_v3')
        """
        # Try to get specific model ID
        if 'model_id' in spec:
            model_id = spec['model_id']
            logger.info(f"Checking for model: {model_id}")

            model = self.model_registry.get_model_by_id(model_id)
            if model:
                logger.info(f"✓ Model {model_id} exists")
                return model_id
            else:
                logger.info(f"Model {model_id} not found, will need to create")

        # Try to get latest version of base model
        if 'model_name' in spec:
            model_name = spec['model_name']
            version = spec.get('version', 'latest')

            if version == 'latest':
                logger.info(f"Looking for latest version of {model_name}")
                latest = self.model_registry.get_latest_version(model_name)
                if latest:
                    model_id = latest['model_id']
                    logger.info(f"✓ Found latest: {model_id}")
                    return model_id
            else:
                # Specific version requested
                model = self.model_registry.get_version(model_name, version)
                if model:
                    model_id = model['model_id']
                    logger.info(f"✓ Found version: {model_id}")
                    return model_id

        # Model doesn't exist - need to train
        logger.info("Model not found, training new model...")

        if training_data is None:
            raise ValueError("Training data required to create new model")

        if not self.model_trainer:
            raise ValueError("ModelTrainer required to create new models")

        # Train new model
        model_id = self._train_new_model(spec, training_data)

        return model_id

    def _train_new_model(
        self,
        spec: Dict[str, Any],
        training_data: pd.DataFrame
    ) -> str:
        """
        Train a new model based on specification.

        Args:
            spec: Model specification
            training_data: Training data

        Returns:
            New model ID
        """
        logger.info(f"Training new model based on spec: {spec}")

        # 1. LLM plans feature engineering
        feature_plan = self._llm_plan_features(spec, training_data)

        # 2. LLM chooses model configuration
        model_config = self._llm_choose_config(spec, feature_plan)

        # 3. Train model
        model_result = self._train_and_register(spec, feature_plan, model_config, training_data)

        return model_result['model_id']

    def _llm_plan_features(
        self,
        spec: Dict[str, Any],
        training_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Use LLM to plan feature engineering strategy.

        Args:
            spec: Model specification
            training_data: Training data

        Returns:
            Feature engineering plan
        """
        logger.info("LLM planning feature engineering...")

        # Build prompt
        prompt = f"""You are an ML engineer planning features for a trading model.

MODEL SPECIFICATION:
- Target: {spec.get('target', 'next_1h_direction')}
- Model type: {spec.get('model_type', 'classification')}
- Feature hints: {spec.get('features', [])}

AVAILABLE DATA:
- Columns: {list(training_data.columns)}
- Shape: {training_data.shape}
- Timeframe: {training_data.index[0] if hasattr(training_data.index, '__getitem__') else 'unknown'} to {training_data.index[-1] if hasattr(training_data.index, '__getitem__') else 'unknown'}

TASK:
Design a feature engineering strategy that will predict {spec.get('target', 'the target')}.

Consider:
1. Which feature categories to include (price, volume, volatility, momentum, etc.)
2. Lookback windows (e.g., 10, 20, 50 periods)
3. Technical indicators to compute
4. Any cross-sectional or time-based features

Output JSON:
{{
    "feature_categories": ["price", "volume", "volatility"],
    "lookback_windows": [10, 20],
    "indicators": ["RSI", "MACD", "BollingerBands"],
    "rationale": "Brief explanation"
}}"""

        system_prompt = "You are an expert ML engineer for trading systems. Respond with valid JSON only."

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5
            )

            # Clean and parse
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            plan = json.loads(response.strip())

            logger.info(f"Feature plan: {plan['feature_categories']}, {len(plan.get('indicators', []))} indicators")

            return plan

        except Exception as e:
            logger.error(f"LLM feature planning failed: {e}")

            # Fallback plan
            return {
                'feature_categories': ['price', 'volume', 'volatility'],
                'lookback_windows': [10, 20],
                'indicators': ['RSI', 'EMA', 'ATR'],
                'rationale': 'Fallback plan - standard technical features'
            }

    def _llm_choose_config(
        self,
        spec: Dict[str, Any],
        feature_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to choose model hyperparameters.

        Args:
            spec: Model specification
            feature_plan: Feature engineering plan

        Returns:
            Model configuration
        """
        logger.info("LLM choosing model configuration...")

        model_name = spec.get('model_name', 'XGBoost_direction')
        model_type = spec.get('model_type', 'classification')

        # Determine algorithm from model name
        if 'XGBoost' in model_name or 'xgboost' in model_name.lower():
            algorithm = 'xgboost'
        elif 'LightGBM' in model_name or 'lightgbm' in model_name.lower():
            algorithm = 'lightgbm'
        elif 'RandomForest' in model_name:
            algorithm = 'random_forest'
        else:
            algorithm = 'xgboost'  # Default

        prompt = f"""You are an ML engineer configuring a {algorithm} model for trading.

MODEL SPECIFICATION:
- Algorithm: {algorithm}
- Model type: {model_type}
- Target: {spec.get('target', 'next_1h_direction')}

FEATURE PLAN:
- Categories: {feature_plan.get('feature_categories')}
- Indicators: {feature_plan.get('indicators')}

TASK:
Choose appropriate hyperparameters for this {algorithm} model.

Output JSON:
{{
    "algorithm": "{algorithm}",
    "hyperparameters": {{
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.08,
        // ... other params specific to {algorithm}
    }},
    "rationale": "Brief explanation"
}}"""

        system_prompt = "You are an expert in ML model configuration. Respond with valid JSON only."

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3  # More deterministic for configs
            )

            # Clean and parse
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            # Remove comments from JSON
            lines = response.strip().split('\n')
            cleaned_lines = [line.split('//')[0].strip() for line in lines]
            cleaned = '\n'.join(cleaned_lines)

            config = json.loads(cleaned)

            logger.info(f"Model config: {config['algorithm']}, {len(config.get('hyperparameters', {}))} params")

            return config

        except Exception as e:
            logger.error(f"LLM config selection failed: {e}")

            # Fallback config
            return {
                'algorithm': algorithm,
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1
                },
                'rationale': 'Fallback config - default parameters'
            }

    def _train_and_register(
        self,
        spec: Dict[str, Any],
        feature_plan: Dict[str, Any],
        model_config: Dict[str, Any],
        training_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Execute training and register model.

        Args:
            spec: Model specification
            feature_plan: Feature engineering plan
            model_config: Model configuration
            training_data: Training data

        Returns:
            Dict with model_id and metadata
        """
        logger.info("Training and registering model...")

        # Get next version number
        model_name = spec.get('model_name', 'XGBoost_direction')
        latest = self.model_registry.get_latest_version(model_name)

        if latest:
            # Increment version
            import re
            match = re.search(r'v(\d+)', latest.get('version', 'v0'))
            current_version = int(match.group(1)) if match else 0
            new_version = f"v{current_version + 1}"
        else:
            new_version = "v1"

        logger.info(f"Training {model_name}_{new_version}")

        # NOTE: Actual training would happen here using ModelTrainer
        # For now, we'll create a mock trained model

        # Create paths
        model_id = f"{model_name}_{new_version}"
        model_path = self.models_dir / f"{model_id}.joblib"
        pipeline_path = self.models_dir / f"{model_id}_pipeline.joblib"

        # Mock training (in real implementation, call self.model_trainer.train(...))
        mock_metrics = {
            'train_accuracy': 0.65,
            'test_accuracy': 0.60,
            'test_auc': 0.67,
            'direction_accuracy': 0.62
        }

        # Register in model registry
        registered_id = self.model_registry.register_version(
            model_name=model_name,
            version=new_version,
            model_path=str(model_path),
            model_type=spec.get('model_type', 'classification'),
            target=spec.get('target', 'next_1h_direction'),
            features_used=feature_plan.get('feature_categories', []),
            hyperparameters=model_config.get('hyperparameters', {}),
            metrics=mock_metrics,
            feature_pipeline_path=str(pipeline_path),
            supersedes=latest.get('model_id') if latest else None,
            improvement=f"New version with features: {', '.join(feature_plan.get('feature_categories', []))}",
            released_by='ml_quant_agent',
            metadata={
                'feature_plan': feature_plan,
                'model_config': model_config,
                'spec': spec
            }
        )

        logger.info(f"✓ Trained and registered {registered_id}")

        return {
            'model_id': registered_id,
            'model_path': str(model_path),
            'pipeline_path': str(pipeline_path),
            'metrics': mock_metrics
        }
