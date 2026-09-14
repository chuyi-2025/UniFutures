"""Execution: T+2 (Moses MWU) and fixed hold-period."""

from __future__ import annotations

import numpy as np


class T2Executor:
    """T+2: signal at T, return from T+1; min hold before flat/flip."""

    def __init__(self, flat_threshold: float = 0.05, min_hold_days: int = 2) -> None:
        self.flat_threshold = flat_threshold
        self.min_hold_days = min_hold_days
        self.position = 0.0
        self.hold_days = 0

    def reset(self) -> None:
        self.position = 0.0
        self.hold_days = 0

    def _target(self, signal: float) -> float:
        if abs(signal) < self.flat_threshold:
            return 0.0
        return float(signal)

    def step(self, signal: float, log_ret: float) -> tuple[float, float]:
        strategy_ret = self.position * log_ret
        target = self._target(signal)

        if self.position == 0.0:
            if target != 0.0:
                self.position = target
                self.hold_days = 1
        else:
            self.hold_days += 1
            if self.hold_days >= self.min_hold_days:
                if target == 0.0:
                    self.position = 0.0
                    self.hold_days = 0
                elif np.sign(target) != np.sign(self.position):
                    self.position = target
                    self.hold_days = 1
                else:
                    self.position = target

        return strategy_ret, self.position


class HoldExecutor:
    """Rebalance at day open when hold expires; flat if signal=0."""

    def __init__(self, hold_days: int) -> None:
        self.hold_days = hold_days
        self.position = 0
        self.remaining = 0

    def step(self, signal: int, log_ret: float) -> tuple[float, int]:
        if self.remaining == 0:
            if signal == 0:
                self.position = 0
            else:
                self.position = 1 if signal > 0 else -1
                self.remaining = self.hold_days - 1
        else:
            self.remaining -= 1

        return self.position * log_ret, self.position
