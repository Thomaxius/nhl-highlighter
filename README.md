# NHL 25 Highlight Reel Generator

Automatically generate broadcast-style highlight reels from your NHL 25 PS5 clips using a fine-tuned VideoMAE model, template-based detectors, audio analysis, and FFmpeg assembly.

The end goal is to have a pipeline that automatically detects uploaded YouTube clips from PS5, converts them to a highlight reel, and uploads the result back to YouTube.

## Notes
This project is a quick 100% vibe coded slop for personal use. I didn't even write this disclaimer myself. Please don't let it be a reason for not headhunting, or let it be, but don't let the quality of programming or lack of it be it.

- [NHL 25 Highlight Reel Generator](#nhl-25-highlight-reel-generator)
  - [Notes](#notes)
  - [Architecture](#architecture)
  - [Pipeline Overview](#pipeline-overview)
    - [What each detection step does](#what-each-detection-step-does)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
    - [OAuth Setup (required for poller + uploader)](#oauth-setup-required-for-poller--uploader)
  - [App 1 — Poller (`apps/poller`)](#app-1--poller-appspoller)
  - [App 2 — Reel Builder (`apps/reel_builder`)](#app-2--reel-builder-appsreel_builder)
  - [App 3 — Uploader (`apps/uploader`)](#app-3--uploader-appsuploader)
  - [Labelling Training Data](#labelling-training-data)
  - [Training](#training)
  - [Template Images](#template-images)
  - [Tech Stack](#tech-stack)
  - [Development](#development)
  - [Project Structure](#project-structure-1)
  - [Quick Start](#quick-start)
    - [1. Set Up Environment](#1-set-up-environment)
    - [2. Get Clips Off PS5](#2-get-clips-off-ps5)
    - [3. Run the Full Pipeline](#3-run-the-full-pipeline)
  - [Labelling Training Data](#labelling-training-data-1)


---

## Architecture

The project is split into three independent apps:

```
apps/
├── poller/          # Watches your YouTube channel for new uploads
├── reel_builder/    # Processes a video and produces a highlight reel
└── uploader/        # Uploads a finished reel back to YouTube
```

The full automated flow is: **poller** detects a new upload → invokes **reel_builder** → invokes **uploader**.
All three can also be run independently.

---

## Pipeline Overview

```
PS5 clips (.mp4)
    ↓ Step 1   — FFmpeg normalise (1080p30, libx264)
    ↓ Step 2   — PySceneDetect → scene segments
    ↓ Step 3   — librosa audio spike detection
    ↓ Step 4   — VideoMAE classifier → goal / celebration / replay / scoring_chance / other
    ↓ Step 4b  — GOAL! banner detection (OpenCV template match, raw video scan)
    ↓ Step 4c  — Banner override: banner-detected segments are forced to "goal"; banner is ground truth
    ↓ Step 4d  — Game clock detection (OCR): tag final 8s + 20s post-buzzer as game_end
    ↓ Step 4d.5— VS screen detection: tag pre-game intro segment
    ↓ Step 4d.9— Pause menu detection: demote ESC/menu segments
    ↓ Step 4e  — Score/SOG change detection (OCR)
    ↓ Step 4f  — Chain goal → celebration → replay sequences
    ↓ Step 5   — Segment scoring (confidence + audio boost)
    ↓ Step 6   — FFmpeg concat: intro → goal chains → game_end outro
    ↓ Step 7   — End-of-game stats detection: parse skater/goalie tables, save CSVs, build YouTube description
    → data/exports/<game>_reel.mp4
    → data/exports/<game>_reel/<team>_all_skaters.csv  (stats CSVs)
    → data/exports/<game>_reel/<team>_goalies.csv
    → data/exports/<game>_reel.txt                     (YouTube description with stats)
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
| Stats screen detector | Template match + OCR | Detects end-of-game stats screens; parses skater and goalie tables for both teams; exports CSVs and YouTube description |

---

## Project Structure

```
nhl-highlighter/
├── apps/
│   ├── poller/
│   │   └── youtube_watcher.py       # YouTube polling daemon
│   ├── reel_builder/
│   │   ├── pipeline.py              # End-to-end CLI entry point
│   │   ├── configs/                 # Template images and config.yaml
│   │   │   ├── config.yaml
│   │   │   ├── goal_banner_template*.png
│   │   │   ├── game_end_template.png
│   │   │   ├── vs_screen_template.png
│   │   │   ├── pause_menu_template.png
│   │   │   ├── stats_tab_all_skaters_template.png
│   │   │   ├── stats_tab_goalies_template.png
│   │   │   └── stats_back_sort_template.png
│   │   ├── src/
│   │   │   ├── ingestion/           # FFmpeg normalisation
│   │   │   ├── preprocessing/       # Scene splitting, audio analysis
│   │   │   ├── detection/           # VideoMAE classifier + template detectors
│   │   │   │   └── stats_detector.py  # End-of-game stats screen parser
│   │   │   ├── assembly/            # FFmpeg concat + overlays
│   │   │   └── utils/
│   │   └── tools/                   # Dev helpers (debug_banner, etc.)
│   ├── trainer/
│   │   ├── src/training/            # Dataset + HuggingFace Trainer
│   │   └── tools/                   # label_segments, map_training_data, etc.
│   ├── uploader/
│   │   └── upload_youtube.py        # YouTube upload CLI
│   └── shared/
│       ├── data/
│       │   ├── raw/                 # Downloaded PS5 recordings
│       │   ├── processed/           # Normalised clips + scene segments
│       │   ├── labeled/             # Training data (subfolders = class names)
│       │   └── exports/             # Finished reels + per-game stats
│       │       ├── <game>_reel.mp4
│       │       ├── <game>_reel.txt  # YouTube description (title + stats + disclaimer)
│       │       └── <game>_reel/stats/  # Per-team CSV stats tables
│       ├── models/                  # Model checkpoints and exports
│       └── training-data/           # Raw source clips by category
├── config/
│   ├── client_secrets.json          # OAuth 2.0 credentials (not in git)
│   ├── token.json                   # Cached poller OAuth token (not in git)
│   └── youtube_token.json           # Cached uploader OAuth token (not in git)
├── infra/
│   └── ansible/                     # Server provisioning playbooks
├── tests/
│   ├── fixtures/                    # 1-minute video chunks for detector testing
│   └── test_stats_detector.py
├── requirements.txt
└── requirements-server.txt
```

---

## Setup

```bash
cd nhl-highlighter
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
pip install -r requirements-server.txt   # needed for poller + uploader
```

Verify FFmpeg and Tesseract are installed:
```bash
ffmpeg -version
tesseract --version
```

### OAuth Setup (required for poller + uploader)

Both the poller and uploader authenticate with YouTube via OAuth 2.0:

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → create a project
2. Enable **YouTube Data API v3**
3. Create credentials: **OAuth client ID** → Application type: **Desktop app**
4. Download `client_secrets.json` → place at `config/client_secrets.json`
5. Add your Google account as a **Test user** on the OAuth consent screen
6. Run either app once interactively — a browser URL will be printed, paste it into the browser with the right Google account signed in, complete consent, and `token.json` is saved automatically

---

## App 1 — Poller (`apps/poller`)

Polls your YouTube channel for new uploads, downloads them with yt-dlp, runs the reel builder, and triggers the uploader.

**Required environment variables:**

| Variable | Description |
|---|---|
| `YOUTUBE_CHANNEL_ID` | Your channel ID (from youtube.com/account_advanced) |
| `OAUTH_CLIENT_SECRETS` | Path to `client_secrets.json` (default: `config/client_secrets.json`) |
| `OAUTH_TOKEN_FILE` | Path to cached token (default: `config/token.json`) |
| `APP_DIR` | Root of the project (default: `/opt/nhl-highlighter`) |

**Optional:**

| Variable | Default | Description |
|---|---|---|
| `POLL_INTERVAL_SECS` | `900` | Seconds between polls |
| `MIN_VIDEO_DURATION_SECS` | `1800` | Skip videos shorter than this |
| `UPLOAD_PRIVACY` | `unlisted` | Privacy of uploaded reels |

**Run:**
```powershell
$env:YOUTUBE_CHANNEL_ID = "UCxxxxxxxxxxxxxxxxxxxx"
$env:OAUTH_CLIENT_SECRETS = "k:\nhl-highlighter\config\client_secrets.json"
$env:OAUTH_TOKEN_FILE = "k:\nhl-highlighter\config\token.json"
$env:APP_DIR = "k:\nhl-highlighter"
python apps/poller/youtube_watcher.py
```

---

## App 2 — Reel Builder (`apps/reel_builder`)

Processes a raw game recording and produces a highlight reel.

**Run:**
```bash
python apps/reel_builder/pipeline.py --input apps/shared/data/raw
```

Optional flags:
```bash
python apps/reel_builder/pipeline.py \
    --input apps/shared/data/raw \
    --checkpoint apps/shared/models/checkpoints/videomae_nhl \
    --music path/to/music.mp3 \
    --max_clips 10
```

Output is saved to `data/exports/` automatically.

---

## App 3 — Uploader (`apps/uploader`)

Uploads a finished reel to YouTube.

**Run:**
```bash
python apps/uploader/upload_youtube.py \
    --file data/exports/my_reel.mp4 \
    --title "NHL 25 Highlights" \
    --privacy unlisted
```

Uses the same `client_secrets.json` as the poller but caches its token separately in `config/youtube_token.json` (needs the `youtube.upload` scope). The first run will print a URL — open it in your browser, complete consent, and the token is saved automatically. SCP it to the server for headless use:
```bash
scp config/youtube_token.json root@SERVER:/opt/nhl-highlighter/config/youtube_token.json
```

---

## Labelling Training Data

Sort short scene clips (2–30s) into the matching folder under `data/labeled/`:

```
data/labeled/
    goal/            → scene contains the GOAL! HUD banner
    celebration/     → players celebrating after a goal
    goal_replay/     → slow-motion replay directly following a goal
    other_replay/    → cutscenes, non-goal replays, coach/crowd reactions
    scoring_chance/  → shot on goal that didn't score (no banner)
    other/           → faceoffs, stoppages, normal gameplay
    transition/      → very short black-frame cuts / transition screens between scenes
```

Use the interactive labelling tool:
```bash
python apps/trainer/tools/label_segments.py \
    --segments apps/shared/data/processed/segments/<game_folder> \
    --output apps/shared/data/labeled
```

Keyboard shortcuts: `g` goal · `c` celebration · `r` goal_replay · `n` other_replay · `s` scoring_chance · `t` transition · `o` other · `SPACE` skip · `q` quit

---

## Training

```bash
python -m apps.trainer.src.training.trainer \
    --data_dir apps/shared/data/labeled \
    --output_dir apps/shared/models/checkpoints/videomae_nhl \
    --epochs 10 \
    --batch_size 4
```

---

## Template Images

OpenCV template matching requires cropped reference images in `configs/`. Replace them if your capture setup differs:

| File | What to crop | Used by |
|---|---|---|
| `goal_banner_template*.png` | The "GOAL!" HUD banner | Banner detector |
| `game_end_template.png` | Clock region (top-left HUD) | Game clock detector |
| `vs_screen_template.png` | VS matchup screen | VS screen detector |
| `pause_menu_template.png` | Top-left ~250×80px of pause/ESC menu | Pause menu detector |
| `stats_tab_all_skaters_template.png` | Active "ALL SKATERS" tab label | Stats screen detector |
| `stats_tab_goalies_template.png` | Active "GOALIES" tab label | Stats screen detector |
| `stats_back_sort_template.png` | BACK / SORT button bar (appears when table is fully loaded) | Stats screen detector |

Multiple `goal_banner_template` files (numbered 1–N) handle different team banner colours.

The stats templates are team-independent — they match fixed UI text ("ALL SKATERS", "GOALIES") and a button bar that is identical regardless of which team is playing.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Video processing | FFmpeg, OpenCV, PySceneDetect |
| Audio analysis | librosa |
| OCR | Tesseract (via pytesseract) |
| Model | VideoMAE (HuggingFace Transformers) |
| Training | PyTorch, HuggingFace Trainer |
| YouTube API | google-api-python-client, google-auth-oauthlib |
| Experiment tracking | Weights & Biases |

---

## Development

```bash
pytest tests/
black apps/ && ruff check apps/
```


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

## Labelling Training Data

The classifier is trained on six classes. Sort short scene clips (2–30s) into the matching folder under `data/labeled/`:

```
data/labeled/
    goal/            → scene contains the GOAL! HUD banner
    celebration/     → players celebrating after a goal
    goal_replay/     → slow-motion replay directly following a goal
    other_replay/    → cutscenes, non-goal replays, coach/crowd reactions
    scoring_chance/  → shot on goal that didn't score (puck visible, no banner)
    other/           → faceoffs, stoppages, normal gameplay
    transition/      → very short black-frame cuts between scenes
```

**Tips:**
- `goal` and `celebration`/`goal_replay` frequently appear back-to-back — the pipeline chains them automatically, so label each scene by what it primarily shows
- `goal_replay` = slow-mo shot replay after a goal; `other_replay` = cutscenes, bench/crowd shots, non-goal replays
- `scoring_chance` is the hardest class — include close shots that look like goals but have no banner
- `transition` clips are usually under 1 second and are skipped during reel assembly
- Aim for **50+ clips per class** before training; balance matters more than raw count

Use the interactive labelling script to quickly sort processed scene segments:

```bash
python scripts/label_segments.py \
    --segments data/processed/segments/<game_folder> \
    --output data/labeled
```

Keyboard shortcuts: `g` goal · `c` celebration · `r` goal_replay · `n` other_replay · `s` scoring_chance · `t` transition · `o` other · `SPACE` skip · `q` quit


