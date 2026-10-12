"""End-to-end tests for apps/processor/process_local.py.

The real pipeline (hours of GPU/CPU) and the real YouTube uploader are replaced by tiny
stand-in scripts placed under a throwaway APP_DIR, so these tests exercise the
processor's own logic: state tracking, retry, archive/cleanup, locking.
"""
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

PROCESSOR = Path(__file__).resolve().parents[1] / "apps" / "processor" / "process_local.py"

STUB_PIPELINE = textwrap.dedent('''
    import argparse, os, sys, time
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--file"); p.add_argument("--output"); p.add_argument("--local", action="store_true")
    a = p.parse_args()
    calls = Path(os.environ["APP_DIR"]) / "stub_calls"
    calls.mkdir(exist_ok=True)
    with open(calls / "pipeline_calls.txt", "a") as f:
        f.write(a.file + "|" + ("local" if a.local else "") + "\\n")
    (calls / "pipeline_started").write_text("1")
    time.sleep(float(os.environ.get("STUB_PIPELINE_SLEEP", "0")))
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"reel")
    out.with_suffix(".txt").write_text("__TITLE__\\n\\nstats go here")
''')

STUB_UPLOAD = textwrap.dedent('''
    import argparse, json, os, sys
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--file"); p.add_argument("--title"); p.add_argument("--description")
    p.add_argument("--privacy")
    a = p.parse_args()
    calls = Path(os.environ["APP_DIR"]) / "stub_calls"
    calls.mkdir(exist_ok=True)
    with open(calls / "upload_calls.jsonl", "a") as f:
        f.write(json.dumps(vars(a)) + "\\n")
    if (calls / "upload_fail").exists():
        print("quota exceeded", file=sys.stderr)
        sys.exit(1)
    print("Upload complete! https://youtube.com/watch?v=abc123")
''')


@pytest.fixture
def app(tmp_path):
    root = tmp_path / "app"
    (root / "apps" / "reel_builder").mkdir(parents=True)
    (root / "apps" / "uploader").mkdir(parents=True)
    (root / "apps" / "reel_builder" / "pipeline.py").write_text(STUB_PIPELINE)
    (root / "apps" / "uploader" / "upload_youtube.py").write_text(STUB_UPLOAD)
    raw = root / "apps" / "shared" / "data" / "raw"
    raw.mkdir(parents=True)
    return root


def run(app, *args, env_extra=None, timeout=60):
    import os
    env = {**os.environ, "APP_DIR": str(app), "LOCAL_SETTLE_SECS": "0", **(env_extra or {})}
    return subprocess.run([sys.executable, str(PROCESSOR), *args], env=env,
                          capture_output=True, text=True, timeout=timeout)


def state(app):
    return json.loads((app / "apps" / "shared" / "data" / "processed_videos.json").read_text())


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def calls(app, name):
    return lines(app / "stub_calls" / name)


def make_recording(app, name="NHL 25_20261011210148.webm"):
    f = app / "apps" / "shared" / "data" / "raw" / name
    f.write_bytes(b"video")
    return f


def test_happy_path_uploads_records_and_cleans_up(app):
    src = make_recording(app)
    r = run(app)
    assert r.returncode == 0, r.stdout + r.stderr

    entry = state(app)["processed"]["local-NHL_25_20261011210148"]
    assert entry["status"] == "done"
    assert entry["title"] == "NHL 25_20261011210148"
    assert state(app)["uploaded_reels"] == ["abc123"]

    # upload got the title substituted into the description template, default privacy
    up = json.loads(calls(app, "upload_calls.jsonl")[0])
    assert up["title"] == "NHL 25_20261011210148"
    assert up["description"].startswith("NHL 25_20261011210148\n")
    assert "__TITLE__" not in up["description"]
    assert up["privacy"] == "unlisted"

    # source archived, exported reel deleted
    assert not src.exists()
    assert (src.parent / "processed_local" / src.name).exists()
    assert not (app / "apps" / "shared" / "data" / "exports" / "local-NHL_25_20261011210148_highlights.mp4").exists()


def test_privacy_comes_from_env(app):
    make_recording(app)
    assert run(app, env_extra={"UPLOAD_PRIVACY": "private"}).returncode == 0
    assert json.loads(calls(app, "upload_calls.jsonl")[0])["privacy"] == "private"


def test_failed_upload_is_retried_without_rerunning_the_pipeline(app):
    src = make_recording(app)
    (app / "stub_calls").mkdir()
    (app / "stub_calls" / "upload_fail").write_text("x")

    r1 = run(app)
    assert r1.returncode == 1
    entry = state(app)["processed"]["local-NHL_25_20261011210148"]
    assert entry["status"] == "upload_failed"
    assert src.exists(), "source must stay put until the upload succeeds"
    assert Path(entry["reel_path"]).exists(), "reel must be kept for the retry"
    assert len(calls(app, "pipeline_calls.txt")) == 1

    (app / "stub_calls" / "upload_fail").unlink()
    r2 = run(app)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert state(app)["processed"]["local-NHL_25_20261011210148"]["status"] == "done"
    assert len(calls(app, "pipeline_calls.txt")) == 1, "retry must be upload-only"
    assert len(calls(app, "upload_calls.jsonl")) == 2
    assert not src.exists()


def test_done_files_are_skipped_unless_forced(app):
    make_recording(app)
    assert run(app).returncode == 0
    make_recording(app)  # same recording shows up again
    assert run(app).returncode == 0
    assert len(calls(app, "pipeline_calls.txt")) == 1, "done file must not be reprocessed"
    assert run(app, "--force").returncode == 0
    assert len(calls(app, "pipeline_calls.txt")) == 2


def test_stale_processing_entry_from_a_crashed_run_is_redone(app):
    make_recording(app)
    data = app / "apps" / "shared" / "data"
    (data / "processed_videos.json").write_text(json.dumps(
        {"processed": {"local-NHL_25_20261011210148": {"status": "processing", "title": "x"}}}))
    assert run(app).returncode == 0
    assert state(app)["processed"]["local-NHL_25_20261011210148"]["status"] == "done"
    assert len(calls(app, "pipeline_calls.txt")) == 1


def test_overlapping_runs_do_not_double_process(app):
    make_recording(app)
    import os
    env = dict(os.environ, APP_DIR=str(app), LOCAL_SETTLE_SECS="0", STUB_PIPELINE_SLEEP="4")
    first = subprocess.Popen([sys.executable, str(PROCESSOR)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = app / "stub_calls" / "pipeline_started"
    deadline = time.time() + 30
    while not started.exists() and time.time() < deadline:
        time.sleep(0.2)
    assert started.exists(), "first run never reached the pipeline"

    second = run(app)  # while the first is mid-pipeline
    assert second.returncode == 0
    assert "already in progress" in second.stdout

    out, err = first.communicate(timeout=60)
    assert first.returncode == 0, out + err
    assert len(calls(app, "pipeline_calls.txt")) == 1


def test_file_still_copying_is_left_alone(app):
    make_recording(app)  # mtime = now
    r = run(app, env_extra={"LOCAL_SETTLE_SECS": "3600"})
    assert r.returncode == 0
    assert calls(app, "pipeline_calls.txt") == []
    assert "settling" in r.stdout


def test_specific_file_by_name_and_local_flag(app):
    make_recording(app, "a.webm")
    make_recording(app, "b.webm")
    assert run(app, "--file", "b.webm", "--local").returncode == 0
    pc = calls(app, "pipeline_calls.txt")
    assert len(pc) == 1 and pc[0].endswith("|local") and "b.webm" in pc[0]
    assert (app / "apps" / "shared" / "data" / "raw" / "a.webm").exists()


def test_pipeline_failure_marks_error_and_next_run_retries(app):
    make_recording(app)
    (app / "apps" / "reel_builder" / "pipeline.py").write_text("import sys; sys.exit(3)")
    assert run(app).returncode == 1
    e = state(app)["processed"]["local-NHL_25_20261011210148"]
    assert e["status"] == "error" and "code 3" in e["error"]
    (app / "apps" / "reel_builder" / "pipeline.py").write_text(STUB_PIPELINE)
    assert run(app).returncode == 0
    assert state(app)["processed"]["local-NHL_25_20261011210148"]["status"] == "done"


def test_normalised_copy_is_used_to_skip_the_reencode(app):
    make_recording(app)
    processed = app / "apps" / "shared" / "data" / "processed"
    processed.mkdir(parents=True)
    norm = processed / "NHL 25_20261011210148_norm.mp4"
    norm.write_bytes(b"norm")
    assert run(app).returncode == 0
    assert Path(calls(app, "pipeline_calls.txt")[0].split("|")[0]) == norm
