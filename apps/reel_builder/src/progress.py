"""
progress.py — console ETA + elapsed-time reporting for the pipeline.

The pipeline already logs a "━━━ Step N: … ━━━" header at each stage. This
module learns how long each stage typically takes from past run logs, then
prints a "time remaining" estimate to the *console only* (never to the log
file) as each step starts. It also exposes a duration formatter for the final
"completed in …" line.

Estimates use the median of past runs, which is robust to the cancelled /
stalled runs and to the big variance in Step 1 (skipped when a normalised file
already exists). It's a rough guide, not a promise.
"""

from __future__ import annotations

import glob
import logging
import re
import statistics
import sys
import time
from pathlib import Path

# Canonical step sequence, in execution order: (header substring, short label).
# The substrings must be distinctive enough not to cross-match (e.g. "Step 4d:"
# vs "Step 4d.2:").
STEP_SEQUENCE: list[tuple[str, str]] = [
    ("Step 1: Ingesting clips",        "ingest"),
    ("Step 2: Splitting scenes",       "scenes"),
    ("Step 3: Audio analysis",         "audio"),
    ("Step 4: Classifying segments",   "classify+scan"),
    ("Step 4b: Banner detection",      "banner"),
    ("Step 4c: Banner-gating",         "gate"),
    ("Step 4d: Game clock detection",  "clock"),
    ("Step 4d.2: Faceoff-pattern",     "faceoff"),
    ("Step 4d.5: VS screen",           "vs"),
    ("Step 4d.9: Pause-menu",          "pause"),
    ("Step 4f: Chaining",              "chain"),
    ("Step 4f.5: Rescuing",            "rescue"),
    ("Step 5: Scoring",                "score"),
    # Reel build + stats/star scan run concurrently → one combined wall-clock
    # stage. Matched on "Step 6: Building reel", which is present in every
    # historical log (old sequential layout and new concurrent header alike),
    # so it has data immediately rather than only from runs after this change.
    ("Step 6: Building reel",          "reel+scan"),
]
_FINAL_LABEL = STEP_SEQUENCE[-1][1]
# Steps whose duration scales with the number of segments rather than being a
# fixed cost. For these we learn seconds-per-segment and multiply by the actual
# segment count of the current run (known after scene-splitting). "classify+scan"
# is by far the largest stage and is dominated by per-segment ML inference.
_PER_SEGMENT_STEPS = {"classify+scan"}
# A run is considered finished if it logged either of these (old / new wording).
_DONE_MARKERS = ("Pipeline completed in", "Done!")
_MAX_LOGS = 40           # only learn from the most recent N runs
_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\b")
_CLASSIFY_RE = re.compile(r"Classifying:.*scene-")


def format_duration(seconds: float) -> str:
    """'1h 23m' / '5m 12s' / '47s'."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _parse_log(path: str) -> tuple[list[tuple[str, float]], int, float | None, bool]:
    """Return (events, n_segments, last_ts, completed) for one run.

    events are step headers in order; n_segments is the number of classified
    segments; last_ts is the final timestamp seen (used to size the last stage,
    which has no following header); completed indicates the run reached the end.
    Handles HH:MM:SS-only timestamps rolling over midnight.
    """
    events: list[tuple[str, float]] = []
    n_segments = 0
    last_ts: float | None = None
    completed = False
    prev: float | None = None
    offset = 0
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return events, 0, None, False
    for line in lines:
        if _CLASSIFY_RE.search(line):
            n_segments += 1
        if any(mk in line for mk in _DONE_MARKERS):
            completed = True
        m = _TS_RE.match(line)
        if not m:
            continue
        h, mi, s = (int(x) for x in m.groups())
        t = h * 3600 + mi * 60 + s
        if prev is not None and t < prev - 3600:   # clock went backwards → next day
            offset += 86400
        prev = t
        t += offset
        last_ts = t
        for sub, label in STEP_SEQUENCE:
            if sub in line:
                events.append((label, t))
                break
    return events, n_segments, last_ts, completed


def _resolve_logs_dir(logs_dir: Path | None) -> tuple[Path, Path | None]:
    """Find the log directory and the current run's log file (to exclude it)."""
    exclude: Path | None = None
    if logs_dir is None:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.FileHandler):
                exclude = Path(h.baseFilename).resolve()
                logs_dir = exclude.parent
                break
    return (Path(logs_dir) if logs_dir else Path("logs")), exclude


def estimate_step_durations(
    logs_dir: Path | None = None,
) -> tuple[dict[str, float], dict[str, float], int]:
    """Learn per-step timings from recent runs.

    Returns (durations, per_segment, n_runs):
      - durations:   median seconds per step (flat cost)
      - per_segment: for _PER_SEGMENT_STEPS, median seconds *per segment*
      - n_runs:      number of runs that contributed
    """
    base, exclude = _resolve_logs_dir(logs_dir)
    files = sorted(glob.glob(str(base / "pipeline_*_highlights.log")))
    if exclude is not None:
        files = [f for f in files if Path(f).resolve() != exclude]
    files = files[-_MAX_LOGS:]

    samples: dict[str, list[float]] = {label: [] for _, label in STEP_SEQUENCE}
    per_seg_samples: dict[str, list[float]] = {label: [] for label in _PER_SEGMENT_STEPS}
    n_runs = 0
    for fp in files:
        events, n_segments, last_ts, completed = _parse_log(fp)
        if len(events) >= 2:
            n_runs += 1
        for i in range(len(events) - 1):
            label, t0 = events[i]
            _, t1 = events[i + 1]
            d = t1 - t0
            if label not in samples or not (0 <= d < 7200):  # ignore stalls
                continue
            samples[label].append(d)
            if label in per_seg_samples and n_segments > 10:
                per_seg_samples[label].append(d / n_segments)
        # The final stage has no following header — size it from its start to
        # the end of the run, but only for runs that actually finished (so a
        # run cancelled mid-stage doesn't contribute a truncated duration).
        if completed and events and last_ts is not None and events[-1][0] == _FINAL_LABEL:
            d = last_ts - events[-1][1]
            if 0 <= d < 7200:
                samples[_FINAL_LABEL].append(d)

    durations = {label: statistics.median(v) for label, v in samples.items() if v}
    per_segment = {label: statistics.median(v) for label, v in per_seg_samples.items() if v}
    return durations, per_segment, n_runs


class ETAReporter(logging.Handler):
    """Logging handler that prints a console-only ETA on each step header.

    Attached to the root logger, it inspects every record but acts only on the
    step headers, writing the estimate to stdout via print (so it never lands
    in the log file, which is owned by a separate FileHandler).
    """

    def __init__(self, durations: dict[str, float], per_segment: dict[str, float] | None = None):
        super().__init__()
        self._durations = durations
        self._per_segment = per_segment or {}
        self._labels = [label for _, label in STEP_SEQUENCE]
        self.segment_count: int | None = None
        self.t0 = time.time()
        self._color = sys.stdout.isatty()

    def _step_duration(self, label: str) -> float:
        """Estimated seconds for one step — scaled by segment count where known."""
        if label in self._per_segment and self.segment_count:
            return self._per_segment[label] * self.segment_count
        return self._durations.get(label, 0.0)

    @property
    def total(self) -> float:
        return sum(self._step_duration(l) for l in self._labels)

    def _remaining_from(self, label: str) -> float:
        i = self._labels.index(label)
        return sum(self._step_duration(l) for l in self._labels[i:])

    def set_segment_count(self, n: int) -> None:
        """Refine estimates once the run's segment count is known (after Step 2)."""
        self.segment_count = n

    def _write(self, text: str) -> None:
        if self._color:
            text = f"\033[2m{text}\033[0m"
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            for sub, label in STEP_SEQUENCE:
                if sub in msg:
                    if self.total > 0:
                        elapsed = time.time() - self.t0
                        self._write(
                            f"   ⏳ ~{format_duration(self._remaining_from(label))} remaining"
                            f"   (elapsed {format_duration(elapsed)})"
                        )
                    return
        except Exception:        # a reporting glitch must never break the run
            pass

    @classmethod
    def attach(cls, logs_dir: Path | None = None) -> "ETAReporter":
        """Compute estimates from past logs, attach to root logger, announce total."""
        durations, per_segment, n_runs = estimate_step_durations(logs_dir)
        reporter = cls(durations, per_segment)
        logging.getLogger().addHandler(reporter)
        if reporter.total > 0:
            reporter._write(
                f"   ⏳ Estimated total run time ~{format_duration(reporter.total)} "
                f"(from {n_runs} past run{'s' if n_runs != 1 else ''})"
            )
        return reporter
