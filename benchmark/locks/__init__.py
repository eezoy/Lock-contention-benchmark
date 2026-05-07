"""
Locks sub-package.

Exports all lock implementations plus the abstract base class.
"""

from benchmark.locks.base import BaseLock
from benchmark.locks.spinlock import Spinlock
from benchmark.locks.ticket_lock import TicketLock
from benchmark.locks.mcs_lock import MCSLock
from benchmark.locks.rw_lock import RWLock

__all__ = [
    "BaseLock",
    "Spinlock",
    "TicketLock",
    "MCSLock",
    "RWLock",
]
