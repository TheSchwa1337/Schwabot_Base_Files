from enum import Enum

"""API enumeration types."""

"""API System Enums ===================="







Contains all enumerations for the Schwabot live API integration system.



"""

# =====================================================================


#  API Enumerations


# =====================================================================


class ExchangeType(str, Enum):
    """Supported exchanges."""

    BINANCE = "binance"

    COINBASE = "coinbase"

    KRAKEN = "kraken"

    CUSTOM = "custom"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"

    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"

    LIMIT = "limit"


class DataType(str, Enum):
    """Data types for API payloads."""

    TRADE = "trade"

    ORDER_BOOK = "order_book"

    NEWS = "news"


class ConnectionStatus(Enum):
    """Connection status."""

    DISCONNECTED = "disconnected"

    CONNECTING = "connecting"

    CONNECTED = "connected"

    ERROR = "error"

    RECONNECTING = "reconnecting"
