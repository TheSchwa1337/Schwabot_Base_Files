from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, List
from models.enums import FillType, Side

class OrderBookUpdate(BaseModel):
    ts:      datetime
    price:   float
    volume:  float
    side:    Side

class Fill(BaseModel):
    ts:       datetime
    ftype:    FillType
    price:    float
    quantity: float
    meta:     Dict = Field(default_factory=dict)

class TickPacket(BaseModel):
    tick_id:    int
    mid_price:  float
    wall_snap:  Dict
    tension:    float
    zeta:       bool
    fills:      List[Fill] = Field(default_factory=list) 