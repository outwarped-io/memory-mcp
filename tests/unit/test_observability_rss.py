"""Process RSS helper coverage."""

from __future__ import annotations

import io
import builtins

from memory_mcp import stats


def test_read_process_rss_linux(monkeypatch) -> None:
    monkeypatch.setattr(stats.os.path, "exists", lambda path: path == "/proc/self/statm")
    monkeypatch.setattr(stats.os, "sysconf", lambda name: 4096)

    def fake_open(path, encoding=None):
        assert path == "/proc/self/statm"
        assert encoding == "ascii"
        return io.StringIO("10 3 0 0 0 0 0\n")

    monkeypatch.setattr(builtins, "open", fake_open)

    out = stats.read_process_rss()

    assert out.rss_bytes == 3 * 4096
    assert out.rss_reason is None
    assert out.uptime_seconds is not None


def test_read_process_rss_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(stats.os.path, "exists", lambda _path: False)

    out = stats.read_process_rss()

    assert out.rss_bytes is None
    assert out.rss_reason == "unsupported_os"
