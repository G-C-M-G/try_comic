"""Run progress kernel for the one-click pipeline.

The worker writes a small status file (``run.json``, atomic tmp+rename,
throttled) that any consumer (terminal, GUI poller) can read to render
"which stage, which item, done/total, elapsed, ETA".  A rolling log
(``run.log``) keeps the last lines for review.

Stages follow the pipeline: capture -> dataset -> ocr -> translate ->
typeset -> pdf.  Each stage carries its own done/total counters (page or
crop level).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

STAGES = ("capture", "dataset", "ocr", "translate", "typeset", "pdf")


@dataclass
class RunStatus:
    run_id: str = ""
    stage: str = "idle"
    stage_index: int = 0
    total_stages: int = len(STAGES)
    current_item: str = ""
    done: int = 0
    total: int = 0
    sub_done: int = 0
    sub_total: int = 0
    started_at: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0
    logs: list[str] = field(default_factory=list)
    error: str = ""
    pid: int = 0
    finished: bool = False
    ok: bool = True

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "current_item": self.current_item,
            "done": self.done,
            "total": self.total,
            "sub_done": self.sub_done,
            "sub_total": self.sub_total,
            "started_at": self.started_at,
            "elapsed": round(self.elapsed, 1),
            "eta": round(self.eta, 1),
            "logs": self.logs[-8:],
            "error": self.error,
            "pid": self.pid,
            "finished": self.finished,
            "ok": self.ok,
        }


class ProgressFile:
    """Atomic, throttled run.json + run.log writer (thread-safe)."""

    def __init__(self, out_root: Path, throttle: float = 0.5) -> None:
        self.path = Path(out_root) / "run.json"
        self.log_path = Path(out_root) / "run.log"
        self.throttle = throttle
        self._lock = threading.Lock()
        self.status = RunStatus(
            run_id=time.strftime("%Y%m%d-%H%M%S"),
            started_at=time.time(),
            pid=os.getpid(),
        )
        self._last_write = 0.0
        self._write(force=True)

    # ---- stage control ----
    def set_stage(self, stage: str, total: int = 0) -> None:
        with self._lock:
            if stage not in STAGES:
                raise ValueError(f"unknown stage: {stage!r}")
            self.status.stage = stage
            self.status.stage_index = STAGES.index(stage)
            self.status.done = 0
            self.status.total = total
            self.status.current_item = ""
            self.status.sub_done = 0
            self.status.sub_total = 0
        self._maybe_write()

    def progress(self, done: int, item: str = "", *, total: int | None = None) -> None:
        with self._lock:
            self.status.done = done
            if total is not None:
                self.status.total = total
            if item:
                self.status.current_item = item
        self._maybe_write()

    def sub_progress(self, done: int, total: int, item: str = "") -> None:
        with self._lock:
            self.status.sub_done = done
            self.status.sub_total = total
            if item:
                self.status.current_item = item
        self._maybe_write()

    def log(self, text: str) -> None:
        with self._lock:
            line = f"[{time.strftime('%H:%M:%S')}] {text}"
            self.status.logs.append(line)
            try:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                pass
        self._maybe_write()

    def fail(self, message: str) -> None:
        with self._lock:
            self.status.error = message
            self.status.finished = True
            self.status.ok = False
        self.log(f"ERROR: {message}")
        self._write(force=True)

    def done(self, ok: bool = True) -> None:
        with self._lock:
            self.status.finished = True
            self.status.ok = ok
            self.status.elapsed = time.time() - self.status.started_at
        self._write(force=True)

    # ---- internals ----
    def _maybe_write(self) -> None:
        now = time.time()
        if now - self._last_write >= self.throttle:
            self._write()

    def _write(self, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            self.status.elapsed = now - self.status.started_at
            if (
                self.status.total > 0
                and self.status.done > 0
                and self.status.elapsed > 2
            ):
                rate = self.status.done / self.status.elapsed
                remaining = max(0, self.status.total - self.status.done)
                self.status.eta = remaining / rate if rate > 0 else 0.0
            payload = self.status.to_dict()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
            self._last_write = now
        except OSError:
            pass
