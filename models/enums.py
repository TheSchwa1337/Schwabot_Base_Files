from enum import Enum

class Side(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"

class FillType(str, Enum):
    BUY_FILL  = "BUY_FILL"
    SELL_FILL = "SELL_FILL"

class OrderState(str, Enum):
    OPEN    = "open"
    PARTIAL = "partial"
    FILLED  = "filled"
    CANCELED= "canceled" 