"""Broker adapters + order/fill persistence.

Public surface::

    from trading.execution import Broker, BrokerError, Simulator, IbkrBroker
    from trading.execution import OrderStore, new_client_order_id
"""

from __future__ import annotations

from trading.execution.base import Broker, BrokerError, NotConnectedError
from trading.execution.ibkr import IbkrBroker, new_client_order_id
from trading.execution.simulator import Simulator
from trading.execution.store import OrderStore

__all__ = [
    "Broker",
    "BrokerError",
    "IbkrBroker",
    "NotConnectedError",
    "OrderStore",
    "Simulator",
    "new_client_order_id",
]
