import time

class Timer:
    """Records named time intervals for profiling pipeline stages."""

    def __init__(self):
        self._last = time.time()
        self._records = {}

    def record(self, name: str):
        """Record elapsed time since the last record (or creation) under *name*."""
        now = time.time()
        self._records[name] = now - self._last
        self._last = now

    def summary(self) -> dict:
        """Return all recorded intervals in minutes plus the total (rounded to 2 decimal places)."""
        total = sum(self._records.values())
        result = {k: round(v / 60, 2) for k, v in self._records.items()}
        result["total"] = round(total / 60, 2)
        return result
