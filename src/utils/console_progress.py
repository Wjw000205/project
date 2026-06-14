import os
import shutil
import sys
import time
from typing import Optional


class PurpleProgressBar:
    """Small single-line progress bar for interactive terminals."""

    PURPLE = "\033[95m"
    RESET = "\033[0m"
    CLEAR_LINE = "\033[K"

    def __init__(
        self,
        total: int,
        label: str,
        unit: str = "step",
        width: int = 28,
        min_interval: float = 0.2,
        leave: Optional[bool] = None,
        stream=None,
    ) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.unit = unit
        self.width = max(int(width), 8)
        self.min_interval = max(float(min_interval), 0.0)
        self.stream = stream if stream is not None else sys.stdout
        self.current = 0
        self.started_at = time.perf_counter()
        self._last_render_at = 0.0
        self._last_len = 0
        self.enabled = self._is_interactive()
        if leave is None:
            leave = os.environ.get("MOELOSS_PROGRESS_LEAVE", "1") != "0"
        self.leave = bool(leave)

    def _is_interactive(self) -> bool:
        if os.environ.get("MOELOSS_PROGRESS_FORCE") == "1":
            return True
        if os.environ.get("NO_COLOR"):
            return False
        isatty = getattr(self.stream, "isatty", None)
        return bool(isatty and isatty())

    def _format_time(self, seconds: Optional[float]) -> str:
        if seconds is None or seconds < 0:
            return "--:--"
        seconds = int(seconds)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _line(self, current: int, suffix: str) -> str:
        current = max(0, min(int(current), self.total))
        frac = current / float(self.total)
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.started_at
        rate = current / elapsed if elapsed > 0 and current > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0 and current < self.total else None
        fixed_visible = (
            f"{self.label} [{bar}] "
            f"{current}/{self.total} {frac * 100:5.1f}% "
            f"elapsed={self._format_time(elapsed)} eta={self._format_time(eta)}"
        )
        suffix_text = ""
        if suffix:
            columns = max(shutil.get_terminal_size((120, 20)).columns, 40)
            allowance = columns - len(fixed_visible) - 4
            if allowance > 3:
                suffix_fit = suffix
                if len(suffix_fit) > allowance:
                    suffix_fit = suffix_fit[: max(0, allowance - 3)] + "..."
                suffix_text = f" | {suffix_fit}"
        return (
            f"{self.label} {self.PURPLE}[{bar}]{self.RESET} "
            f"{current}/{self.total} {frac * 100:5.1f}% "
            f"elapsed={self._format_time(elapsed)} eta={self._format_time(eta)}"
            f"{suffix_text}"
        )

    def update(self, current: int, suffix: str = "", force: bool = False) -> None:
        self.current = max(0, min(int(current), self.total))
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and (now - self._last_render_at) < self.min_interval:
            return
        line = self._line(self.current, suffix)
        pad = " " * max(0, self._last_len - len(line))
        self.stream.write("\r" + line + self.CLEAR_LINE + pad)
        self.stream.flush()
        self._last_render_at = now
        self._last_len = len(line)

    def finish(self, current: Optional[int] = None, suffix: str = "") -> None:
        if current is not None:
            self.current = max(0, min(int(current), self.total))
        if not self.enabled:
            return
        if not self.leave:
            self.stream.write("\r" + self.CLEAR_LINE)
            self.stream.flush()
            self._last_len = 0
            return
        line = self._line(self.current, suffix)
        pad = " " * max(0, self._last_len - len(line))
        self.stream.write("\r" + line + self.CLEAR_LINE + pad + "\n")
        self.stream.flush()
        self._last_len = 0
