"""Custom exceptions for SimulatedExchange."""


class SimulatedExchangeError(Exception):
    """Base exception for all simulated exchange errors."""
    pass


class InsufficientFundsError(SimulatedExchangeError):
    """Raised when there are insufficient funds for an order."""
    pass


class InvalidOrderError(SimulatedExchangeError):
    """Raised when order parameters are invalid."""
    pass


class OrderNotFoundError(SimulatedExchangeError):
    """Raised when an order cannot be found."""
    pass


class PositionNotFoundError(SimulatedExchangeError):
    """Raised when a position cannot be found."""
    pass


class DataFeedError(SimulatedExchangeError):
    """Raised when there are issues with price feed data."""
    pass


class ConnectionError(SimulatedExchangeError):
    """Raised when connection to live feed fails."""
    pass
