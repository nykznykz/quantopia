"""Slippage models for order execution."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging

from .models import OrderSide

logger = logging.getLogger(__name__)


class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def calculate_slippage(
        self,
        price: float,
        size: float,
        side: OrderSide,
        volume: float = None
    ) -> float:
        """Calculate slippage for an order.

        Args:
            price: Expected execution price
            size: Order size
            side: Order side (buy/sell)
            volume: Bar volume (optional, for volume-based models)

        Returns:
            Slipped price (actual execution price)
        """
        pass


class FixedSlippageModel(SlippageModel):
    """Fixed percentage slippage model."""

    def __init__(self, fixed_bps: float = 5):
        """Initialize fixed slippage model.

        Args:
            fixed_bps: Fixed slippage in basis points (default 5 = 0.05%)
        """
        self.fixed_bps = fixed_bps
        self.slippage_pct = fixed_bps / 10000.0  # Convert bps to percentage

    def calculate_slippage(
        self,
        price: float,
        size: float,
        side: OrderSide,
        volume: float = None
    ) -> float:
        """Calculate fixed slippage.

        Buys slip upward (pay more), sells slip downward (receive less).
        """
        if side == OrderSide.BUY:
            slipped_price = price * (1 + self.slippage_pct)
        else:  # SELL
            slipped_price = price * (1 - self.slippage_pct)

        logger.debug(
            f"Fixed slippage: {price:.2f} -> {slipped_price:.2f} "
            f"({self.fixed_bps} bps, {side.value})"
        )
        return slipped_price


class VolumeBasedSlippageModel(SlippageModel):
    """Volume-based slippage model."""

    def __init__(
        self,
        volume_limit: float = 0.10,
        price_impact: float = 0.1
    ):
        """Initialize volume-based slippage model.

        Args:
            volume_limit: Maximum order size as fraction of bar volume (default 0.10 = 10%)
            price_impact: Price impact factor (0.1 = 10 bps per 1% of volume)
        """
        self.volume_limit = volume_limit
        self.price_impact = price_impact

    def calculate_slippage(
        self,
        price: float,
        size: float,
        side: OrderSide,
        volume: float = None
    ) -> float:
        """Calculate volume-based slippage.

        Larger orders relative to volume have more market impact.
        """
        if volume is None or volume <= 0:
            logger.warning("No volume data, using minimal slippage")
            volume = size * 100  # Assume order is 1% of volume

        # Calculate volume share
        volume_share = size / volume if volume > 0 else 0

        # Calculate market impact
        impact = volume_share * self.price_impact / 100.0  # Convert to percentage

        # Apply slippage direction based on side
        if side == OrderSide.BUY:
            slipped_price = price * (1 + impact)
        else:  # SELL
            slipped_price = price * (1 - impact)

        logger.debug(
            f"Volume slippage: {price:.2f} -> {slipped_price:.2f} "
            f"(vol_share={volume_share:.2%}, impact={impact:.4%}, {side.value})"
        )
        return slipped_price


class HybridSlippageModel(SlippageModel):
    """Hybrid slippage model combining base + volume + volatility."""

    def __init__(
        self,
        base_bps: float = 3,
        volume_limit: float = 0.10,
        price_impact: float = 0.1,
        use_volatility: bool = True,
        volatility_lookback: int = 20
    ):
        """Initialize hybrid slippage model.

        Args:
            base_bps: Base slippage in basis points
            volume_limit: Maximum order size as fraction of volume
            price_impact: Price impact factor
            use_volatility: Whether to adjust for volatility
            volatility_lookback: Number of bars for volatility calculation
        """
        self.base_bps = base_bps
        self.base_pct = base_bps / 10000.0
        self.volume_limit = volume_limit
        self.price_impact = price_impact
        self.use_volatility = use_volatility
        self.volatility_lookback = volatility_lookback
        self._price_history = []

    def calculate_slippage(
        self,
        price: float,
        size: float,
        side: OrderSide,
        volume: float = None
    ) -> float:
        """Calculate hybrid slippage.

        Combines base slippage, market impact, and volatility adjustment.
        """
        # Base slippage
        base_slip = self.base_pct

        # Market impact
        if volume is not None and volume > 0:
            volume_share = size / volume
            market_impact = (volume_share * self.price_impact) / 100.0
        else:
            market_impact = 0

        # Volatility multiplier
        volatility_mult = 1.0
        if self.use_volatility and len(self._price_history) > 1:
            volatility_mult = self._calculate_volatility_multiplier()

        # Combine components
        total_slippage = (base_slip + market_impact) * volatility_mult

        # Apply direction
        if side == OrderSide.BUY:
            slipped_price = price * (1 + total_slippage)
        else:  # SELL
            slipped_price = price * (1 - total_slippage)

        # Update price history
        self._price_history.append(price)
        if len(self._price_history) > self.volatility_lookback:
            self._price_history.pop(0)

        logger.debug(
            f"Hybrid slippage: {price:.2f} -> {slipped_price:.2f} "
            f"(base={base_slip:.4%}, impact={market_impact:.4%}, "
            f"vol_mult={volatility_mult:.2f}, {side.value})"
        )
        return slipped_price

    def _calculate_volatility_multiplier(self) -> float:
        """Calculate volatility multiplier based on recent price history."""
        if len(self._price_history) < 2:
            return 1.0

        import numpy as np
        prices = np.array(self._price_history)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)

        # Scale volatility (0-10% vol -> 1.0-2.0 multiplier)
        multiplier = 1.0 + (volatility * 10)
        return min(max(multiplier, 1.0), 3.0)  # Cap between 1.0 and 3.0


class NoSlippageModel(SlippageModel):
    """No slippage model for testing."""

    def calculate_slippage(
        self,
        price: float,
        size: float,
        side: OrderSide,
        volume: float = None
    ) -> float:
        """Return price unchanged."""
        return price


def create_slippage_model(config: Optional[Dict] = None) -> SlippageModel:
    """Factory function to create slippage model from config.

    Args:
        config: Slippage configuration dict

    Returns:
        SlippageModel instance
    """
    if config is None:
        config = {}

    model_type = config.get('model', 'fixed')

    if model_type == 'fixed':
        return FixedSlippageModel(
            fixed_bps=config.get('fixed_bps', 5)
        )
    elif model_type == 'volume_based':
        return VolumeBasedSlippageModel(
            volume_limit=config.get('volume_limit', 0.10),
            price_impact=config.get('price_impact', 0.1)
        )
    elif model_type == 'hybrid':
        return HybridSlippageModel(
            base_bps=config.get('base_bps', 3),
            volume_limit=config.get('volume_limit', 0.10),
            price_impact=config.get('price_impact', 0.1),
            use_volatility=config.get('use_volatility', True),
            volatility_lookback=config.get('volatility_lookback', 20)
        )
    elif model_type == 'none':
        return NoSlippageModel()
    else:
        raise ValueError(f"Unknown slippage model: {model_type}")
