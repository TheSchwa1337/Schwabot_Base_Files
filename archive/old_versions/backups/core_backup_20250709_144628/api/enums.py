"""Module for Schwabot trading system."""

from enum import Enum

"""API enumeration types."""

"""API System Enums ===================="







Contains all enumerations for the Schwabot live API integration system.



"""

# =====================================================================


#  API Enumerations


# =====================================================================


    class ExchangeType(str, Enum):
    """Class for Schwabot trading functionality."""
    """Class for Schwabot trading functionality."""
    """Supported exchanges."""

    BINANCE = "binance"

    COINBASE = "coinbase"

    KRAKEN = "kraken"

    CUSTOM = "custom"


        class OrderSide(str, Enum):
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """Order side."""

        BUY = "buy"

        SELL = "sell"


            class OrderType(str, Enum):
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            """Order type."""

            MARKET = "market"

            LIMIT = "limit"


                class DataType(str, Enum):
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                """Data types for API payloads."""

                TRADE = "trade"

                ORDER_BOOK = "order_book"

                NEWS = "news"


                    class ConnectionStatus(Enum):
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """Connection status."""

                    DISCONNECTED = "disconnected"

                    CONNECTING = "connecting"

                    CONNECTED = "connected"

                    ERROR = "error"

                    RECONNECTING = "reconnecting"
