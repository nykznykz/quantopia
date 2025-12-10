"""Live price feed implementations for real-time trading."""

import json
import threading
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import websocket

from .price_feed import PriceFeed
from .models import OHLCV
from .exceptions import DataFeedError, ConnectionError as ConnError

logger = logging.getLogger(__name__)


class LivePriceFeed(PriceFeed):
    """Base class for live price feeds."""

    def __init__(
        self,
        exchange: str,
        symbols: List[str],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True
    ):
        """Initialize live price feed.

        Args:
            exchange: Exchange name ('binance', 'hyperliquid')
            symbols: List of trading symbols
            api_key: API key (optional for public data)
            api_secret: API secret (optional for public data)
            testnet: Use testnet/paper trading (default True)
        """
        self.exchange = exchange.lower()
        self.symbols = symbols
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Price data storage
        self._prices: Dict[str, Dict] = {}
        self._connected = False
        self._ws = None
        self._ws_thread = None
        self._running = False

        # Initialize exchange-specific feed
        if self.exchange == 'binance':
            self._impl = BinanceLiveFeed(symbols, testnet)
        elif self.exchange == 'hyperliquid':
            self._impl = HyperliquidLiveFeed(symbols, testnet)
        else:
            raise DataFeedError(f"Unsupported exchange: {exchange}")

        logger.info(f"Initialized {exchange} live feed for {symbols}")

    def connect(self):
        """Establish WebSocket connection."""
        self._impl.connect()
        self._connected = True
        logger.info(f"Connected to {self.exchange}")

    def disconnect(self):
        """Close WebSocket connection."""
        self._impl.disconnect()
        self._connected = False
        logger.info(f"Disconnected from {self.exchange}")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected and self._impl.is_connected()

    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        return self._impl.get_current_price(symbol)

    def get_current_volume(self, symbol: str) -> float:
        """Get current volume for symbol."""
        return self._impl.get_current_volume(symbol)

    def get_current_timestamp(self) -> datetime:
        """Get current timestamp."""
        return datetime.now()

    def get_ohlcv(self, symbol: str) -> OHLCV:
        """Get current OHLCV bar."""
        return self._impl.get_ohlcv(symbol)


class BinanceLiveFeed:
    """Binance WebSocket price feed."""

    def __init__(self, symbols: List[str], testnet: bool = True):
        """Initialize Binance feed."""
        self.symbols = symbols
        self.testnet = testnet
        self._prices: Dict[str, Dict] = {s: {} for s in symbols}
        self._ws = None
        self._ws_thread = None
        self._running = False

        # Convert symbols to Binance format (BTC-USD -> btcusdt)
        self._binance_symbols = [self._to_binance_symbol(s) for s in symbols]

    def connect(self):
        """Connect to Binance WebSocket."""
        # Note: For getting public price data, we use production WebSocket
        # It's read-only and doesn't require authentication
        # Testnet WebSocket has limited availability for spot trading
        base_url = "wss://stream.binance.com:9443/ws"

        # Create stream URL for combined streams
        if len(self._binance_symbols) == 1:
            # Single stream
            stream = f"{self._binance_symbols[0].lower()}@ticker"
            url = f"{base_url}/{stream}"
        else:
            # Combined streams
            streams = [f"{s.lower()}@ticker" for s in self._binance_symbols]
            url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

        logger.info(f"Connecting to Binance WebSocket: {url}")

        self._running = True
        self._ws_thread = threading.Thread(target=self._run_websocket, args=(url,))
        self._ws_thread.daemon = True
        self._ws_thread.start()

        # Wait for initial connection
        time.sleep(2)

    def disconnect(self):
        """Disconnect from Binance WebSocket."""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._ws_thread:
            self._ws_thread.join(timeout=5)

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._running and self._ws_thread is not None and self._ws_thread.is_alive()

    def get_current_price(self, symbol: str) -> float:
        """Get current price."""
        binance_symbol = self._to_binance_symbol(symbol)
        if binance_symbol not in self._prices or not self._prices[binance_symbol]:
            raise DataFeedError(f"No price data for {symbol}")
        return float(self._prices[binance_symbol].get('c', 0))

    def get_current_volume(self, symbol: str) -> float:
        """Get current volume."""
        binance_symbol = self._to_binance_symbol(symbol)
        if binance_symbol not in self._prices or not self._prices[binance_symbol]:
            raise DataFeedError(f"No volume data for {symbol}")
        return float(self._prices[binance_symbol].get('v', 0))

    def get_ohlcv(self, symbol: str) -> OHLCV:
        """Get current OHLCV."""
        binance_symbol = self._to_binance_symbol(symbol)
        if binance_symbol not in self._prices or not self._prices[binance_symbol]:
            raise DataFeedError(f"No data for {symbol}")

        data = self._prices[binance_symbol]
        return OHLCV(
            timestamp=datetime.now(),
            open=float(data.get('o', 0)),
            high=float(data.get('h', 0)),
            low=float(data.get('l', 0)),
            close=float(data.get('c', 0)),
            volume=float(data.get('v', 0))
        )

    def _run_websocket(self, url: str):
        """Run WebSocket connection."""
        def on_message(ws, message):
            try:
                data = json.loads(message)

                # Handle combined stream format (has 'stream' and 'data' keys)
                if 'stream' in data and 'data' in data:
                    data = data['data']

                # Check if this is ticker data
                if 's' in data:  # Symbol present
                    symbol = data['s'].lower()
                    self._prices[symbol] = data
                    logger.debug(f"Updated price for {symbol}: {data.get('c')}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")

        def on_open(ws):
            logger.info("Binance WebSocket connected")

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                self._ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    time.sleep(5)
                else:
                    break

    def _to_binance_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Binance format.

        Args:
            symbol: Symbol in format 'BTC-USD'

        Returns:
            Binance symbol format 'btcusdt'
        """
        if '-' in symbol:
            base, quote = symbol.split('-')
            if quote == 'USD':
                quote = 'USDT'
            return f"{base.lower()}{quote.lower()}"
        return symbol.lower()


class HyperliquidLiveFeed:
    """Hyperliquid WebSocket price feed."""

    def __init__(self, symbols: List[str], testnet: bool = True):
        """Initialize Hyperliquid feed."""
        self.symbols = symbols
        self.testnet = testnet
        self._prices: Dict[str, Dict] = {s: {} for s in symbols}
        self._ws = None
        self._ws_thread = None
        self._running = False

    def connect(self):
        """Connect to Hyperliquid WebSocket."""
        if self.testnet:
            url = "wss://api.hyperliquid-testnet.xyz/ws"
        else:
            url = "wss://api.hyperliquid.xyz/ws"

        logger.info(f"Connecting to Hyperliquid WebSocket: {url}")

        self._running = True
        self._ws_thread = threading.Thread(target=self._run_websocket, args=(url,))
        self._ws_thread.daemon = True
        self._ws_thread.start()

        time.sleep(2)

    def disconnect(self):
        """Disconnect from Hyperliquid WebSocket."""
        self._running = False
        if self._ws:
            self._ws.close()
        if self._ws_thread:
            self._ws_thread.join(timeout=5)

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._running and self._ws_thread is not None and self._ws_thread.is_alive()

    def get_current_price(self, symbol: str) -> float:
        """Get current price."""
        hl_symbol = self._to_hyperliquid_symbol(symbol)
        if hl_symbol not in self._prices or not self._prices[hl_symbol]:
            raise DataFeedError(f"No price data for {symbol}")
        return float(self._prices[hl_symbol].get('price', 0))

    def get_current_volume(self, symbol: str) -> float:
        """Get current volume."""
        hl_symbol = self._to_hyperliquid_symbol(symbol)
        if hl_symbol not in self._prices or not self._prices[hl_symbol]:
            raise DataFeedError(f"No volume data for {symbol}")
        return float(self._prices[hl_symbol].get('volume', 0))

    def get_ohlcv(self, symbol: str) -> OHLCV:
        """Get current OHLCV."""
        hl_symbol = self._to_hyperliquid_symbol(symbol)
        if hl_symbol not in self._prices or not self._prices[hl_symbol]:
            raise DataFeedError(f"No data for {symbol}")

        data = self._prices[hl_symbol]
        price = float(data.get('price', 0))

        return OHLCV(
            timestamp=datetime.now(),
            open=price,  # For real-time, we approximate OHLC with current price
            high=price,
            low=price,
            close=price,
            volume=float(data.get('volume', 0))
        )

    def _run_websocket(self, url: str):
        """Run WebSocket connection."""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                # Hyperliquid-specific message parsing
                # Note: This is a simplified implementation
                # Actual Hyperliquid API may have different format
                if 'data' in data and 'symbol' in data['data']:
                    symbol = data['data']['symbol']
                    self._prices[symbol] = data['data']
                    logger.debug(f"Updated price for {symbol}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")

        def on_open(ws):
            logger.info("Hyperliquid WebSocket connected")
            # Subscribe to symbols
            subscribe_msg = {
                "method": "subscribe",
                "subscription": {
                    "type": "allMids"
                }
            }
            ws.send(json.dumps(subscribe_msg))

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                self._ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    time.sleep(5)
                else:
                    break

    def _to_hyperliquid_symbol(self, symbol: str) -> str:
        """Convert standard symbol to Hyperliquid format.

        Args:
            symbol: Symbol in format 'BTC-USD'

        Returns:
            Hyperliquid symbol format
        """
        if '-' in symbol:
            base, quote = symbol.split('-')
            return base  # Hyperliquid typically uses just the base (e.g., 'BTC')
        return symbol
