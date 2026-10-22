"""Broker interface. The order manager only talks to `Broker`.

  ShadowBroker  logs target portfolios, sends nothing (shadow mode, 4-8 weeks)
  PaperBroker   simulated fills with the SAME cost model as the backtest, state on disk
  AlpacaBroker  Alpaca paper/live REST API (US equities), keys from env vars
  KiteBroker    Zerodha Kite Connect (India) - check SEBI's current retail algo rules first

Orders are computed as share deltas from target weights, rounded to whole shares,
and reconciled against reported fills.
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from rltrade.envs.costs import CostModel

log = logging.getLogger(__name__)


class Broker(ABC):
    @abstractmethod
    def positions(self) -> dict[str, float]:
        """Shares held per symbol."""

    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def submit(self, orders: dict[str, int], prices: dict[str, float]) -> list[dict]:
        """Submit share orders (+buy/-sell). Returns fills."""

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash() + sum(q * prices[s] for s, q in self.positions().items() if s in prices)

    def orders_for(self, target_w: dict[str, float], prices: dict[str, float], min_notional: float = 100.0) -> dict[str, int]:
        eq = self.equity(prices)
        pos = self.positions()
        orders = {}
        for s, w in target_w.items():
            want = int(np.floor(w * eq / prices[s]))
            d = want - int(pos.get(s, 0))
            if abs(d) * prices[s] >= min_notional:
                orders[s] = d
        for s, q in pos.items():  # close anything not in the target
            if s not in target_w and q:
                orders[s] = -int(q)
        return orders


class ShadowBroker(Broker):
    def __init__(self, capital: float = 1e6):
        self._cash = capital

    def positions(self):
        return {}

    def cash(self):
        return self._cash

    def submit(self, orders, prices):
        log.info("[shadow] would submit %d orders (nothing sent)", len(orders))
        return []


class PaperBroker(Broker):
    """Local simulated account; fills at the given price plus modelled costs."""

    def __init__(self, state_path: Path, capital: float = 1e6, cost: CostModel | None = None):
        self.path = Path(state_path)
        self.cost = cost or CostModel()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"cash": capital, "positions": {}, "fills": []}
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=1))

    def positions(self):
        return dict(self.state["positions"])

    def cash(self):
        return float(self.state["cash"])

    def submit(self, orders, prices, adv: dict[str, float] | None = None, sigma: dict[str, float] | None = None):
        fills = []
        eq = self.equity(prices)
        for s, q in orders.items():
            px = prices[s]
            dw = np.array([q * px / eq])
            rate = self.cost.rates(dw, eq, np.array([px]), np.array([(adv or {}).get(s, 1e7)]),
                                   np.array([(sigma or {}).get(s, 0.02)]))[0]
            fill_px = px * (1 + rate * np.sign(q))
            self.state["cash"] -= q * fill_px
            self.state["positions"][s] = self.state["positions"].get(s, 0) + q
            fills.append({"symbol": s, "qty": q, "price": fill_px, "ref_price": px, "ts": time.time()})
        self.state["positions"] = {s: q for s, q in self.state["positions"].items() if q}
        self.state["fills"].extend(fills)
        self._save()
        return fills


class AlpacaBroker(Broker):
    """Alpaca REST v2. Set ALPACA_KEY_ID / ALPACA_SECRET_KEY; paper endpoint by default."""

    def __init__(self, base_url: str = "https://paper-api.alpaca.markets"):
        import requests

        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"],
                               "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]})
        self.base = base_url.rstrip("/")

    def _get(self, path):
        r = self.s.get(self.base + path, timeout=10)
        r.raise_for_status()
        return r.json()

    def positions(self):
        return {p["symbol"]: float(p["qty"]) for p in self._get("/v2/positions")}

    def cash(self):
        return float(self._get("/v2/account")["cash"])

    def submit(self, orders, prices, retries: int = 3):
        out = []
        for s, q in orders.items():
            body = {"symbol": s, "qty": abs(int(q)), "side": "buy" if q > 0 else "sell", "type": "market",
                    "time_in_force": "opg"}  # market-on-open: matches the backtest's open fill
            for attempt in range(retries):
                try:
                    r = self.s.post(self.base + "/v2/orders", json=body, timeout=10)
                    r.raise_for_status()
                    out.append(r.json())
                    break
                except Exception as e:  # noqa: BLE001 - retry any transport error
                    log.warning("order %s attempt %d failed: %s", s, attempt + 1, e)
                    time.sleep(2**attempt)
        return out


class KiteBroker(Broker):
    """Zerodha Kite Connect skeleton. Fill in with kiteconnect.KiteConnect once API access
    is approved, and confirm the current SEBI framework for retail algo orders
    (registered algo IDs / broker-side controls) before sending anything live."""

    def __init__(self, *_, **__):
        raise NotImplementedError("Kite integration requires API keys and SEBI-compliant setup; see docstring.")

    def positions(self): ...

    def cash(self): ...

    def submit(self, orders, prices): ...
