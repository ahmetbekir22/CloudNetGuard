"""
CloudNetGuard — SDN aksiyon tanımları.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    BLOCK    = "BLOCK"
    REDIRECT = "REDIRECT"
    MIRROR   = "MIRROR"
    ALLOW    = "ALLOW"


@dataclass
class ActionResult:
    action: Action
    reason: str
    target: str = ""        # REDIRECT için hedef IP/honeypot adresi

    def describe(self) -> str:
        descriptions = {
            Action.BLOCK:    f"Kaynak IP engellendi. Sebep: {self.reason}",
            Action.REDIRECT: f"Trafik honeypot'a yönlendirildi ({self.target}). Sebep: {self.reason}",
            Action.MIRROR:   f"Trafik izleme için kopyalandı. Sebep: {self.reason}",
            Action.ALLOW:    f"Trafik geçişine izin verildi. Sebep: {self.reason}",
        }
        return descriptions[self.action]


# Sabit honeypot adresi (simüle)
HONEYPOT_IP = "10.0.99.1"
