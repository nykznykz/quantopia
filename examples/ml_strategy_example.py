"""Example: Train ML model and create ML-augmented trading strategy.

This example demonstrates Phase 1 capabilities:
1. Feature engineering from technical indicators
2. Training XGBoost model for direction prediction
3. Using ML predictions in trading strategy
"""

import sys
import os
from datetime import datetime
import pandas as pd
import logging

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.pipeline import FeaturePipeline, create_target_variable
from src.ml.sklearn_models import XGBoostModel
from src.ml.training import ModelTrainer
from src.ml.model_registry import ModelRegistry
from src.ml.base_model import ModelType
from src.backtest.runner import BacktestRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main example workflow."""

    logger.info("="*80)
    logger.info("Phase 1 ML Example: Train Model and Backtest ML Strategy")
    logger.info("="*80)

    # Configuration
    symbol = "BTC-USD"
    exchange = "binance"
    timeframe = "1h"

    # Step 1: Load historical data
    logger.info("\n[Step 1] Loading historical data...")
    backtest_runner = BacktestRunner(initial_capital=10000.0)

    data = backtest_runner.load_data(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1)
    )

    logger.info(f"Loaded {len(data)} candles")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Step 2: Setup feature pipeline
    logger.info("\n[Step 2] Setting up feature engineering pipeline...")
    feature_pipeline = FeaturePipeline(
        technical_configs=None,  # Use defaults
        normalize=True,
        fill_method='forward'
    )

    # Step 3: Setup ML model
    logger.info("\n[Step 3] Setting up XGBoost model...")
    model = XGBoostModel(
        model_type=ModelType.CLASSIFICATION,
        name="XGBoost_Direction",
        params={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        }
    )

    # Step 4: Train model
    logger.info("\n[Step 4] Training model...")
    registry = ModelRegistry(registry_dir="models/registry")

    trainer = ModelTrainer(
        model=model,
        feature_pipeline=feature_pipeline,
        registry=registry,
        models_dir="models"
    )

    # Train with train/val/test split
    results = trainer.train_and_evaluate(
        data=data,
        target_type='binary',  # Predict up (1) or down (0)
        horizon=1,  # Next period
        threshold=0.0,
        train_ratio=0.6,
        val_ratio=0.2,
        save_model=True,
        register_model=True
    )

    logger.info("\n[Step 5] Training Results:")
    logger.info(f"Model ID: {results['model_id']}")
    logger.info(f"Model path: {results['model_path']}")

    logger.info("\nTrain Metrics:")
    for metric, value in results['metrics']['train'].items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\nValidation Metrics:")
    for metric, value in results['metrics']['validation'].items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\nTest Metrics:")
    for metric, value in results['metrics']['test'].items():
        logger.info(f"  {metric}: {value:.4f}")

    if results['feature_importance'] is not None:
        logger.info("\nTop 10 Most Important Features:")
        for feature, importance in results['feature_importance'].head(10).items():
            logger.info(f"  {feature}: {importance:.4f}")

    # Step 6: Create ML-augmented strategy
    logger.info("\n[Step 6] Creating ML-augmented strategy...")

    strategy_code = f'''
"""ML-augmented trading strategy."""
import pandas as pd
from src.code_generation.strategy_base import BaseStrategy

class MLAugmentedStrategy(BaseStrategy):
    """Strategy that combines technical indicators with ML predictions."""

    def __init__(self, exchange, price_feed, symbol, initial_capital=10000.0):
        super().__init__(exchange, price_feed, symbol, initial_capital)

        self.strategy_name = "MLAugmentedStrategy"
        self.strategy_type = "ml_hybrid"
        self.description = "Combines RSI with XGBoost predictions"

        # Load ML model
        self.load_ml_model(
            model_path="{results['model_path']}",
            pipeline_path="{results['pipeline_path']}"
        )

    def calculate_indicators(self, data):
        """Calculate technical indicators."""
        # Simple RSI calculation
        if len(data) < 14:
            return {{'rsi': 50, 'ml_signal': 0}}

        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Get ML signal
        ml_signal = self.get_ml_signal(data, threshold=0.6)

        return {{
            'rsi': rsi.iloc[-1],
            'ml_signal': ml_signal
        }}

    def should_enter(self):
        """Entry logic: RSI oversold AND ML predicts up."""
        rsi = self.indicators.get('rsi', 50)
        ml_signal = self.indicators.get('ml_signal', 0)

        # Both conditions must be true
        return rsi < 30 and ml_signal == 1

    def should_exit(self):
        """Exit logic: RSI overbought OR ML predicts down."""
        rsi = self.indicators.get('rsi', 50)
        ml_signal = self.indicators.get('ml_signal', 0)

        # Either condition triggers exit
        return rsi > 70 or ml_signal == -1
'''

    # Step 7: Backtest ML strategy
    logger.info("\n[Step 7] Backtesting ML-augmented strategy...")

    # Reload data for backtest
    backtest_data = backtest_runner.load_data(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1)
    )

    backtest_results = backtest_runner.run_from_code(
        strategy_code=strategy_code,
        data=backtest_data,
        symbol=symbol,
        strategy_name="MLAugmentedStrategy"
    )

    # Display backtest results
    logger.info("\n[Step 8] Backtest Results:")
    metrics = backtest_results['metrics']

    logger.info(f"Initial Capital: ${backtest_results['initial_capital']:.2f}")
    logger.info(f"Final Capital: ${backtest_results['final_capital']:.2f}")
    logger.info(f"Total Return: {metrics.get('total_return', 0)*100:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
    logger.info(f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
    logger.info(f"Total Trades: {metrics.get('num_trades', 0)}")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

    # Step 9: Compare with baseline
    logger.info("\n[Step 9] Comparison:")
    logger.info("ML Model Performance:")
    logger.info(f"  Test Accuracy: {results['metrics']['test']['accuracy']:.2f}")
    logger.info(f"  Test Direction Accuracy: {results['metrics']['test'].get('direction_accuracy', 0):.2f}")

    logger.info("\nTrading Strategy Performance:")
    logger.info(f"  Return: {metrics.get('total_return', 0)*100:.2f}%")
    logger.info(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")

    # Step 10: Model registry info
    logger.info("\n[Step 10] Model Registry:")
    summary = registry.get_summary()
    logger.info(f"Total models registered: {summary['total_models']}")
    logger.info(f"Model types: {summary['model_types']}")

    best_model = registry.get_best_model(metric='accuracy')
    if best_model:
        logger.info(f"\nBest model by accuracy:")
        logger.info(f"  ID: {best_model['model_id']}")
        logger.info(f"  Accuracy: {best_model['metrics']['validation']['accuracy']:.4f}")

    logger.info("\n" + "="*80)
    logger.info("Phase 1 ML Example Complete!")
    logger.info("="*80)

    return {
        'model_results': results,
        'backtest_results': backtest_results,
        'registry_summary': summary
    }


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        sys.exit(1)
