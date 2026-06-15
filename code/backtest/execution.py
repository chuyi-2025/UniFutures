"""Position execution with minimum hold period."""

from __future__ import annotations


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
