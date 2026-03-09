# NHL 25 Highlight Reel Generator

Automatically generate broadcast-style highlight reels from your NHL 25 PS5 clips using a fine-tuned VideoMAE model, template-based detectors, audio analysis, and FFmpeg assembly.

The end goal is to have a pipeline that automatically detects uploaded Youtube clips from PS5 and then converts them to a highlight reel which is subsequently uploaded back.

## Notes
This project is a quick 100% vibe coded slop for personal use. I didn't even write this disclaimer myself. Please don't let it be a reason for not headhunting, or let it be, but don't let the quality of programming or lack of it be it.

---

## Project Structure

```
nhl-highlighter/
├── configs/
│   ├── config.yaml                  # Tunable parameters
│   ├── goal_banner_template*.png    # GOAL! HUD banner templates (5 variants)
│   ├── game_end_template.png        # Clock region template for game-end detection
│   ├── vs_screen_template.png       # VS matchup screen template (intro detection)
│   └── pause_menu_template.png      # ESC/pause menu template (menu frame removal)
├── data/
│   ├── raw/                 # Clips copied from PS5 (USB / PS App)
│   ├── processed/           # Normalised clips + scene segments
│   ├── labeled/             # Training data (subfolders = class names)
│   │   ├── goal/            # Scene contains the GOAL! HUD banner
│   │   ├── celebration/     # Player celebration sequences
│   │   ├── replay/          # Slow-motion replays following a goal
│   │   ├── scoring_chance/  # Shots on goal that didn't score
│   │   ├── other/           # Faceoffs, stoppages, normal gameplay
│   │   └── transition/      # Short black-frame scene cuts
│   └── exports/             # Final highlight reels
├── models/
│   ├── checkpoints/         # Fine-tuned VideoMAE checkpoint
│   └── exported/            # ONNX / quantised exports
├── notebooks/               # Exploration & labelling helpers
├── scripts/
│   └── label_segments.py    # Interactive labelling tool for scene segments
├── src/
│   ├── ingestion/
│   │   └── ingest.py           # PS5 clip import & FFmpeg normalisation
│   ├── preprocessing/
│   │   ├── scene_splitter.py   # PySceneDetect scene cutting
│   │   └── audio_analyzer.py   # librosa energy spike detection
│   ├── detection/
│   │   ├── classifier.py          # VideoMAE inference wrapper
│   │   ├── banner_detector.py     # OpenCV template match for GOAL! banner
│   │   ├── game_clock_detector.py # OCR-based clock reader (game/period end)
│   │   ├── score_detector.py      # OCR shots-on-goal tracking
│   │   ├── vs_screen_detector.py  # VS matchup screen detection (intro)
│   │   └── pause_menu_detector.py # ESC/pause menu detection
│   ├── training/
│   │   ├── dataset.py       # PyTorch Dataset for labeled segments
│   │   └── trainer.py       # HuggingFace Trainer fine-tuning script
│   ├── assembly/
│   │   └── reel_builder.py  # FFmpeg concat + overlays + music
│   └── utils/
├── tests/
├── pipeline.py              # End-to-end CLI entry point
└── requirements.txt
```

---

## Quick Start

### 1. Set Up Environment

```bash
cd nhl-highlighter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify FFmpeg and Tesseract are installed:
```bash
ffmpeg -version
tesseract --version
```
If missing: `brew install ffmpeg tesseract`

### 2. Get Clips Off PS5

- Insert `.mp4` files from the game into `data/raw/`

### 3. Run the Full Pipeline

```bash
python pipeline.py --input data/raw
```

The output reel is saved to `data/exports/` automatically, named after the input clip.

Optional flags:
```bash
python pipeline.py \
    --input data/raw \
    --checkpoint models/checkpoints/videomae_nhl \
    --music path/to/music.mp3 \
    --max_clips 10
```

---

## Pipeline Overview

```
PS5 clips (.mp4)
    ↓ Step 1   — FFmpeg normalise (1080p30, libx264)
    ↓ Step 2   — PySceneDetect → scene segments
    ↓ Step 3   — librosa audio spike detection
    ↓ Step 4   — VideoMAE classifier → goal / celebration / replay / scoring_chance / other
    ↓ Step 4b  — GOAL! banner detection (OpenCV template match, raw video scan)
    ↓ Step 4c  — Banner gating: demote any "goal" without confirmed banner
    ↓ Step 4d  — Game clock detection (OCR): tag final 8s + 20s post-buzzer as game_end
    ↓ Step 4d.5— VS screen detection: tag pre-game intro segment
    ↓ Step 4d.9— Pause menu detection: demote ESC/menu segments
    ↓ Step 4e  — Score/SOG change detection (OCR)
    ↓ Step 4f  — Chain goal → celebration → replay sequences
    ↓ Step 5   — Segment scoring (confidence + audio boost)
    ↓ Step 6   — FFmpeg concat: intro → goal chains → game_end outro
    → data/exports/<game>_reel.mp4
```

### What each detection step does

| Step | Method | Purpose |
|---|---|---|
| VideoMAE classifier | ML model | Broad scene categorisation |
| Banner detector | Template match | Confirms real goals (prevents false positives) |
| Game clock detector | OCR (Tesseract) | Finds exact final buzzer timestamp for outro clip |
| VS screen detector | Template match | Detects pre-game matchup screen for intro clip |
| Pause menu detector | Template match | Removes ESC menu frames from the reel |
| Score detector | OCR (Tesseract) | Tracks shots on goal for context |

---

## Labelling Training Data

The classifier is trained on six classes. Sort short scene clips (2–30s) into the matching folder under `data/labeled/`:

```
data/labeled/
    goal/            → scene contains the GOAL! HUD banner
    celebration/     → players celebrating after a goal
    replay/          → slow-motion replay following a goal
    scoring_chance/  → shot on goal that didn't score (puck visible, no banner)
    other/           → faceoffs, stoppages, normal gameplay
    transition/      → very short black-frame cuts between scenes
```

**Tips:**
- `goal` and `celebration`/`replay` frequently appear back-to-back — the pipeline chains them automatically, so label each scene by what it primarily shows
- `scoring_chance` is the hardest class — include close shots that look like goals but have no banner
- `transition` clips are usually under 1 second and are skipped during reel assembly
- Aim for **50+ clips per class** before training; balance matters more than raw count

Use the interactive labelling script to quickly sort processed scene segments:

```bash
python scripts/label_segments.py \
    --segments data/processed/segments/<game_folder> \
    --output data/labeled
```

---

## Training

Training requires a GPU. Use [Runpod](https://runpod.io) or [Lambda Labs](https://lambdalabs.com) (RTX 4090 ~$0.50/hr) for cloud training.

```bash
python -m src.training.trainer \
    --data_dir data/labeled \
    --output_dir models/checkpoints/videomae_nhl \
    --epochs 10 \
    --batch_size 4
```

Track experiments at [wandb.ai](https://wandb.ai) — set `WANDB_API_KEY` in your environment first.

The trainer saves checkpoints to `models/checkpoints/videomae_nhl/`. Use the final checkpoint (or the best validation checkpoint) when running the pipeline.

---

## Template Images

The detectors that use OpenCV template matching require cropped reference images in `configs/`. If you're setting up fresh or the templates don't match your capture setup, replace them:

| File | What to crop | Used by |
|---|---|---|
| `goal_banner_template*.png` | The "GOAL!" text banner from the HUD | Banner detector |
| `game_end_template.png` | The clock region (top-left HUD area) | Game clock detector |
| `vs_screen_template.png` | Any distinctive part of the VS matchup screen | VS screen detector |
| `pause_menu_template.png` | Top-left ~250×80px of the pause/ESC menu screen | Pause menu detector |

Multiple `goal_banner_template` files are supported (numbered 1–N) to handle different team banner colours.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Video processing | FFmpeg, OpenCV, PySceneDetect |
| Audio analysis | librosa |
| OCR | Tesseract (via pytesseract) |
| Model | VideoMAE (HuggingFace Transformers) |
| Training | PyTorch, HuggingFace Trainer |
| Experiment tracking | Weights & Biases |

---

## Development

```bash
# Run tests
pytest tests/

# Format
black src/ && ruff check src/
```
