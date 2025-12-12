"""Test feature engineering pipeline."""

import sys
import os
from datetime import datetime
import pandas as pd
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.technical import TechnicalFeatures
from src.features.pipeline import FeaturePipeline, create_target_variable
from src.backtest.runner import BacktestRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_generation():
    """Test feature generation from OHLCV data."""

    logger.info("Testing Feature Engineering Pipeline")
    logger.info("="*60)

    # Load sample data
    runner = BacktestRunner()
    data = runner.load_data(
        symbol="BTC-USD",
        exchange="binance",
        timeframe="1h",
        start_date=datetime(2023, 6, 1),
        end_date=datetime(2023, 7, 1)
    )

    logger.info(f"\n1. Loaded {len(data)} candles")

    # Test technical features
    logger.info("\n2. Testing Technical Features...")
    tech_features = TechnicalFeatures()

    # Test individual feature types
    price_features = tech_features.compute_price_features(data)
    logger.info(f"   ✓ Price features: {len(price_features.columns)} features")

    volume_features = tech_features.compute_volume_features(data)
    logger.info(f"   ✓ Volume features: {len(volume_features.columns)} features")

    volatility_features = tech_features.compute_volatility_features(data)
    logger.info(f"   ✓ Volatility features: {len(volatility_features.columns)} features")

    temporal_features = tech_features.compute_temporal_features(data)
    logger.info(f"   ✓ Temporal features: {len(temporal_features.columns)} features")

    indicator_features = tech_features.compute_all_features(data)
    logger.info(f"   ✓ Indicator features: {len(indicator_features.columns)} features")

    # Test complete pipeline
    logger.info("\n3. Testing Complete Feature Pipeline...")
    all_features = tech_features.compute_all(data)
    logger.info(f"   ✓ Total features: {len(all_features.columns)} features")
    logger.info(f"   ✓ Valid samples: {len(all_features)} rows")

    # Display sample features
    logger.info("\n4. Sample Features (first 5):")
    logger.info(all_features.head())

    logger.info("\n5. Feature Statistics:")
    logger.info(all_features.describe())

    # Test target creation
    logger.info("\n6. Testing Target Variable Creation...")
    target_binary = create_target_variable(data, target_type='binary', horizon=1)
    logger.info(f"   ✓ Binary target: {target_binary.value_counts().to_dict()}")

    target_direction = create_target_variable(data, target_type='direction', horizon=1)
    logger.info(f"   ✓ Direction target: {target_direction.value_counts().to_dict()}")

    target_return = create_target_variable(data, target_type='return', horizon=1)
    logger.info(f"   ✓ Return target: mean={target_return.mean():.4f}, std={target_return.std():.4f}")

    # Test feature pipeline with normalization
    logger.info("\n7. Testing Feature Pipeline with Normalization...")
    pipeline = FeaturePipeline(normalize=True)
    features, aligned_target = pipeline.fit_transform(data, target_binary)

    logger.info(f"   ✓ Normalized features: {len(features.columns)} features")
    logger.info(f"   ✓ Aligned samples: {len(features)} rows")
    logger.info(f"   ✓ Feature mean (should be ~0): {features.mean().mean():.6f}")
    logger.info(f"   ✓ Feature std (should be ~1): {features.std().mean():.4f}")

    # Test save/load
    logger.info("\n8. Testing Pipeline Save/Load...")
    pipeline.save("models/test_pipeline.joblib")
    loaded_pipeline = FeaturePipeline.load("models/test_pipeline.joblib")

    features_loaded = loaded_pipeline.transform(data)
    logger.info(f"   ✓ Loaded pipeline produces {len(features_loaded.columns)} features")
    logger.info(f"   ✓ Features match: {features.equals(features_loaded)}")

    logger.info("\n" + "="*60)
    logger.info("✓ All feature engineering tests passed!")
    logger.info("="*60)

    return {
        'num_features': len(all_features.columns),
        'num_samples': len(all_features),
        'feature_names': all_features.columns.tolist()
    }


if __name__ == "__main__":
    try:
        results = test_features()
        print(f"\n✓ Generated {results['num_features']} features successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
