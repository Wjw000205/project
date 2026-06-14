from __future__ import annotations

import io

from src.utils.console_progress import PurpleProgressBar


class NonInteractiveStream(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_progress_bar_can_be_forced_for_web_log_capture(monkeypatch) -> None:
    monkeypatch.setenv("MOELOSS_PROGRESS_FORCE", "1")
    stream = NonInteractiveStream()
    progress = PurpleProgressBar(total=10, label="Train ETTm1 H=96", stream=stream)

    progress.update(4, suffix="epoch=2/5 batch=2/5 loss=0.123", force=True)

    text = stream.getvalue()
    assert "Train ETTm1 H=96" in text
    assert "4/10" in text
    assert "epoch=2/5 batch=2/5" in text
