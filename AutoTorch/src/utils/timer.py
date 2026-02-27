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
        """Return all recorded intervals plus the total."""
        total = sum(self._records.values())
        return {**self._records, "total": total}

    def print_formatted_summary(self) -> str:
        """Return a human-readable string of all recorded intervals."""
        lines = ["=" * 40]
        for name, secs in self.summary().items():
            mins, s = divmod(int(secs), 60)
            hrs, mins = divmod(mins, 60)
            lines.append(f"  {name}: {hrs:02d}:{mins:02d}:{s:02d}")
        lines.append("=" * 40)
        print("\n".join(lines))
