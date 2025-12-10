"""Fee models for order execution."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging

from .models import OrderType

logger = logging.getLogger(__name__)


class FeeModel(ABC):
    """Abstract base class for fee models."""

    @abstractmethod
    def calculate_fee(
        self,
        trade_value: float,
        order_type: OrderType
    ) -> float:
        """Calculate fee for a trade.

        Args:
            trade_value: Total value of trade (price * size)
            order_type: Type of order (market or limit)

        Returns:
            Fee amount in USD
        """
        pass


class FlatFeeModel(FeeModel):
    """Flat fee model - same fee for all orders."""

    def __init__(self, fee_rate: float = 0.001):
        """Initialize flat fee model.

        Args:
            fee_rate: Fee rate as decimal (default 0.001 = 0.1%)
        """
        self.fee_rate = fee_rate

    def calculate_fee(
        self,
        trade_value: float,
        order_type: OrderType
    ) -> float:
        """Calculate flat fee."""
        fee = trade_value * self.fee_rate
        logger.debug(f"Flat fee: ${fee:.4f} ({self.fee_rate:.4%} of ${trade_value:.2f})")
        return fee


class TieredFeeModel(FeeModel):
    """Tiered fee model - different fees for makers vs takers."""

    def __init__(
        self,
        maker_fee: float = 0.0000,  # Hyperliquid default
        taker_fee: float = 0.00025   # Hyperliquid default
    ):
        """Initialize tiered fee model.

        Args:
            maker_fee: Fee rate for limit orders (makers)
            taker_fee: Fee rate for market orders (takers)
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def calculate_fee(
        self,
        trade_value: float,
        order_type: OrderType
    ) -> float:
        """Calculate tiered fee based on order type.

        Market orders pay taker fee, limit orders pay maker fee.
        """
        if order_type == OrderType.MARKET:
            fee = trade_value * self.taker_fee
            logger.debug(
                f"Taker fee: ${fee:.4f} ({self.taker_fee:.4%} of ${trade_value:.2f})"
            )
        else:  # LIMIT
            fee = trade_value * self.maker_fee
            logger.debug(
                f"Maker fee: ${fee:.4f} ({self.maker_fee:.4%} of ${trade_value:.2f})"
            )
        return fee


class NoFeeModel(FeeModel):
    """No fee model for testing."""

    def calculate_fee(
        self,
        trade_value: float,
        order_type: OrderType
    ) -> float:
        """Return zero fee."""
        return 0.0


def create_fee_model(config: Optional[Dict] = None) -> FeeModel:
    """Factory function to create fee model from config.

    Args:
        config: Fee configuration dict

    Returns:
        FeeModel instance
    """
    if config is None:
        config = {}

    model_type = config.get('model', 'tiered')

    if model_type == 'flat':
        return FlatFeeModel(
            fee_rate=config.get('fee_rate', 0.001)
        )
    elif model_type == 'tiered':
        return TieredFeeModel(
            maker_fee=config.get('maker_fee', 0.0000),
            taker_fee=config.get('taker_fee', 0.00025)
        )
    elif model_type == 'none':
        return NoFeeModel()
    else:
        raise ValueError(f"Unknown fee model: {model_type}")
