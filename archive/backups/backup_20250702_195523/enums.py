""API enumeration types.from enum import Enum


# !/usr/bin/env python3
# -*- coding: utf-8 -*-

API System Enums ================

Contains all enumerations for the Schwabot live API integration system.


# =====================================================================
#  API Enumerations
# =====================================================================


class ExchangeType(str, Enum):Supported exchanges.BINANCE =  binanceCOINBASE =  coinbaseKRAKEN =  krakenCUSTOM =  customclass OrderSide(str, Enum):Order side.BUY = buySELL =  sellclass OrderType(str, Enum):Order type.MARKET = marketLIMIT =  limitclass DataType(str, Enum):Data types for API payloads.TRADE = tradeORDER_BOOK =  order_bookNEWS =  newsclass ConnectionStatus(Enum):Connection status.DISCONNECTED =  disconnectedCONNECTING =  connectingCONNECTED =  connectedERROR =  errorRECONNECTING =  reconnecting
