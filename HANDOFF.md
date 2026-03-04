# NHL Highlighter — Handoff Notes (4 March 2026)

## Goal
Auto-generate NHL 25 highlight reels from raw gameplay recordings. The pipeline
segments video, classifies each segment, and assembles a reel. A separate data-
collection script auto-extracts training clips for shots-on-goal.

The immediate next task is **retrain the VideoMAE classifier from scratch** on
freshly gathered data, then run the full pipeline end-to-end.

---

## Repo & environment

- **Repo:** `/Users/tsantamaa/home-projects/nhl-highlighter`
- **Python:** 3.12, venv at `.venv`
- **Key deps:** `torch`, `transformers` (VideoMAE), `opencv-python`, `pytesseract`,
  `yt-dlp`, `ffmpeg`, `scenedetect`
- **Activate:** `source .venv/bin/activate`

---

## Folder structure

```
data/
  raw/              ← raw .mp4 gameplay recordings (input to pipeline)
  processed/
    segments/       ← scene-split, normalised 1920×1080 clips (one subdir per recording)
    audio/          ← extracted audio .wav files
  labeled/
    goal/
    celebration/
    replay/
    scoring_chance/ ← auto-populated by collect_scoring_chances.py
    other/
  exports/          ← finished reel output

models/
  checkpoints/
    videomae_nhl/   ← fine-tuned VideoMAE (retrain here)

configs/
  config.yaml
  goal_banner_template*.png   ← template(s) for GOAL! HUD banner matching

scripts/
  collect_scoring_chances.py  ← auto-collects shot-on-goal training clips
  label_segments.py

src/
  detection/
    classifier.py         ← VideoMAE inference
    banner_detector.py    ← template match for GOAL! HUD overlay
    score_detector.py     ← OCR-based HUD reader (score + SOG)
  ingestion/ingest.py
  preprocessing/
    scene_splitter.py
    audio_analyzer.py
  assembly/reel_builder.py
  training/
    dataset.py
    trainer.py

pipeline.py         ← end-to-end runner
```

---

## Pipeline (`pipeline.py`)

```bash
python pipeline.py --folder data/raw --output data/exports/reel.mp4
# or a single file:
python pipeline.py --file data/raw/game.mp4 --output data/exports/reel.mp4
```

**Stages:**

| Step | Module | Notes |
|------|--------|-------|
| 1 | `ingest.py` | Normalises to 1920×1080, consistent fps |
| 2 | `scene_splitter.py` | PySceneDetect → segments per clip |
| 3 | `audio_analyzer.py` | Detects crowd/commentary energy spikes |
| 4 | `classifier.py` | VideoMAE per-segment label + confidence |
| 4b | `banner_detector.py` | Template-match GOAL! banner → hard `goal` label |
| 4c | — | Demotes `goal` labels without banner confirmation if conf < 0.92 |
| 4e | `score_detector.py` | OCR HUD; score change boosts confidence to 0.85 (never overrides label) |
| 4f | — | Chains goal → celebration → replay sequences |
| 5 | `audio_analyzer.py` | Audio spike scoring boost |
| 6 | `reel_builder.py` | FFmpeg concat + optional background music |

**Key design decisions:**
- **Banner detector is ground truth.** A pixel-level GOAL! HUD match always
  wins over classifier output. Score-change OCR is a secondary aid signal only
  (confidence boost, no label override).
- **Demote threshold:** `goal` labels without banner + conf < 0.92 are demoted
  to `other`.
- **Goal chaining:** segments following a banner-confirmed goal are searched for
  `celebration` and `replay` labels (up to 10 segments, stops after 10 s of
  non-highlight content) and included in order.

---

## Label taxonomy

| Label | Description |
|-------|-------------|
| `goal` | GOAL! banner confirmed (hard set by banner_detector) |
| `celebration` | Player/bench celebration after goal |
| `replay` | Slow-motion replay |
| `scoring_chance` | Shot on net, close chance — auto-collected by collect script |
| `other` | Normal gameplay, faceoffs, zone play, etc. |

---

## ScoreDetector (`src/detection/score_detector.py`)

Reads the NHL 25 HUD scoreboard via OCR (pytesseract + OpenCV).

**ROI constants** (fraction of 1920×1080 frame):
```python
ROI_X = 0.02, ROI_Y = 0.035, ROI_W = 0.2176, ROI_H = 0.077
```

**3 sub-crop layout** (% of ROI width):
- `15–65%` — team abbrev + score, top=away / bottom=home (PSM 7)
- `65–78%` — SOG digits, top 40%=away / bottom 60%=home (PSM 10)
- `80–100%` — game clock (PSM 7)

**Threshold:** Otsu with a floor of 90 — handles both the normal opaque HUD and
the semi-transparent version (background brightens to ~103 from ~60–80).

**`detect_in_segment(path, fast=False)` returns:**
```python
{
  "home":             int | None,   # parsed home score
  "away":             int | None,   # parsed away score
  "sog_home":         int | None,   # parsed home SOG count
  "sog_away":         int | None,   # parsed away SOG count
  "score_changed":    bool,         # incremented vs previous segment
  "sog_changed":      bool,         # any shot detected this segment
  "sog_change_count": int,          # total shots counted within segment
  "shot_timestamps":  list[float],  # seconds-into-segment for each shot
  "hud_visible":      bool,
  "raw_text":         str,          # raw OCR string for debugging
}
```

**Important behaviours:**
- `_last_hud_visible` gate — changes only fire when the HUD was visible in
  *both* the current and previous segment. Prevents false positives when the HUD
  reappears after replays/cutscenes.
- Intra-segment shot tracking — compares consecutive sampled-frame reads;
  accepts SOG increments of 1–2, rejects decrements and large jumps (OCR noise).
- `best_sog_home/away` is updated **independently of score parse** — the away
  score string is noisy and often fails, which previously blocked SOG tracking.
- `fast=True` uses 6 frames; `fast=False` uses `sample_frames` (default 12).
  `score_segments()` automatically uses fast mode for `other`/unlabelled
  segments and full mode for highlight candidates.
- `OPENCV_FFMPEG_CAPTURE_OPTIONS=hwaccel;none` is set at module import level to
  prevent a deadlock when OpenCV is called after PyTorch/MPS has claimed the GPU
  on macOS.

**Tune the ROI:** `ScoreDetector().debug_frame("path/to.mp4")` saves a debug
crop to `/tmp/score_roi_debug.png` and prints the OCR result.

**Known issue:** A value like `15` can occasionally be misread as `19` by
tesseract in the SOG sub-crop. The `<= 2` increment cap catches most of this,
but noisy sessions may still show spurious jumps. Cap is in
`detect_in_segment`, intra-segment tracking block.

---

## collect_scoring_chances.py (`scripts/collect_scoring_chances.py`)

Scans a processed-segments folder in chronological order, detects SOG
increments, and outputs windowed clips as `scoring_chance` training data.

```bash
python scripts/collect_scoring_chances.py \
  --segments_dir "data/processed/segments/NHL 25_20260225202819_norm" \
  --output_dir data/labeled/scoring_chance \
  [--dry_run] [--include_goals] [--pre_shot 5.0] [--post_shot 2.0]
```

**Clip window:** `[shot_time − pre_shot_s .. shot_time + post_shot_s]`.
`shot_time` is seconds into the segment where the SOG counter incremented.
If the pre-window extends before the segment start, the deficit is drawn from
the tail of the previous segment via ffmpeg concat. Falls back to
`shutil.copy2` if ffmpeg fails.

**Per-shot naming:**
- One shot in segment → `scene-042_scoring_chance.mp4`
- Multiple shots → `scene-006_shot01.mp4`, `scene-006_shot02.mp4`, …

Detector state is reset automatically between match subfolders.

---

## Training

Training code lives in `src/training/`. The base model is
`MCG-NJU/videomae-base` fine-tuned on the NHL label taxonomy above.

```bash
# Example — check src/training/trainer.py for full args
python -m src.training.trainer \
  --data_dir data/labeled \
  --output_dir models/checkpoints/videomae_nhl \
  --epochs 10
```

The pipeline's `--checkpoint` arg defaults to `models/checkpoints/videomae_nhl`.
After retraining, point it at the new checkpoint dir (or the best checkpoint
subdir, e.g. `checkpoint-370`).

---

## Retraining checklist (current next task)

1. Collect/organise labeled clips into `data/labeled/<label>/`
2. Verify class balance — `scoring_chance` and `other` tend to be
   underrepresented; run `collect_scoring_chances.py` over any processed
   segments folders to top up `scoring_chance/`
3. Run trainer
4. Evaluate on held-out clips; check confusion between `scoring_chance` / `other`
5. Update `--checkpoint` in pipeline invocations

---

## Useful one-liners

```bash
# Debug OCR ROI on a specific segment
python -c "
from src.detection.score_detector import ScoreDetector
ScoreDetector().debug_frame('data/processed/segments/MATCH_norm/scene-006.mp4')
"

# Dry-run shot collection (no files written)
python scripts/collect_scoring_chances.py \
  --segments_dir "data/processed/segments/NHL 25_20260225202819_norm" \
  --dry_run

# Single-file pipeline run
python pipeline.py --file data/raw/game.mp4 --output data/exports/reel.mp4
```
