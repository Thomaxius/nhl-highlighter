# NHL 25 Highlight Reel Generator

Automatically generate broadcast-style highlight reels from your NHL 25 PS5 clips using a fine-tuned VideoMAE model, audio spike detection, and FFmpeg-based assembly.

## Notes
This project is a quick 100% vibe coded slop for personal use. I didn't even write this disclaimer myself. Please don't let it be a reason for not headhunting, or let it be, but don't let the quality of programming or lack of it be it.

---

## Project Structure

```
nhl-highlighter/
├── configs/
│   └── config.yaml          # All tunable parameters
├── data/
│   ├── raw/                 # Clips copied from PS5 (USB / PS App)
│   ├── processed/           # Normalised clips + scene segments
│   ├── labeled/             # Training data (subfolders = class names)
│   │   ├── goal/
│   │   ├── save/
│   │   ├── hit/
│   │   ├── fight/
│   │   ├── faceoff/
│   │   ├── stoppage/
│   │   └── other/
│   └── exports/             # Final highlight reels
├── models/
│   ├── checkpoints/         # Fine-tuned VideoMAE checkpoint
│   └── exported/            # ONNX / quantised exports
├── notebooks/               # Exploration & labelling helpers
├── scripts/                 # One-off utility scripts
├── src/
│   ├── ingestion/
│   │   └── ingest.py        # PS5 clip import & FFmpeg normalisation
│   ├── preprocessing/
│   │   ├── scene_splitter.py  # PySceneDetect scene cutting
│   │   └── audio_analyzer.py  # librosa energy spike detection
│   ├── detection/
│   │   └── classifier.py    # VideoMAE inference wrapper
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

Verify FFmpeg is installed:
```bash
ffmpeg -version
```
If not: `brew install ffmpeg`

### 2. Get Clips Off PS5

- Insert a FAT32/exFAT USB stick into the PS5
- Go to **Media Gallery → Select clips → Copy to USB**
- Copy the `.mp4` files into `data/raw/`

Or use the PlayStation App to download clips directly to your Mac.

### 3. Run the Full Pipeline

```bash
python pipeline.py --input data/raw --output data/exports/reel.mp4
```

> **Note:** On first run the model will use the base VideoMAE weights (no fine-tuning yet). See the Training section below.

### 4. Label Your Data (For Training)

Sort your scene segments into `data/labeled/<class>/`:

```
data/labeled/
    goal/   → clips where you score
    save/   → big saves
    hit/    → big body checks
    fight/  → fights
    other/  → everything else
```

Aim for **100+ clips per class** before training. Use [Label Studio](https://labelstud.io/) for a GUI labelling workflow.

### 5. Fine-Tune the Model

```bash
python -m src.training.trainer \
    --data_dir data/labeled \
    --output_dir models/checkpoints/videomae_nhl \
    --epochs 10 \
    --batch_size 4
```

Track experiments at [wandb.ai](https://wandb.ai) (set `WANDB_API_KEY` in your environment).

For cloud GPU training (recommended), use [Runpod](https://runpod.io) or [Lambda Labs](https://lambdalabs.com) — an RTX 4090 node runs ~$0.50/hr.

### 6. Run with Your Fine-Tuned Model

```bash
python pipeline.py \
    --input data/raw \
    --output data/exports/reel.mp4 \
    --checkpoint models/checkpoints/videomae_nhl \
    --music path/to/music.mp3 \
    --max_clips 10
```

---

## Pipeline Overview

```
PS5 clips (.mp4)
    ↓ FFmpeg normalise (1080p30, libx264)
    ↓ PySceneDetect → scene segments
    ↓ librosa audio spike detection (crowd roar, goal horn)
    ↓ VideoMAE classifier → goal / save / hit / fight / other
    ↓ Segment scoring (confidence + audio boost)
    ↓ FFmpeg concat + text overlays + music mix
    → highlight_reel.mp4
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Video processing | FFmpeg, OpenCV, PySceneDetect, MoviePy |
| Audio analysis | librosa, pydub |
| Model | VideoMAE (HuggingFace Transformers) |
| Training | PyTorch, HuggingFace Trainer, scikit-learn |
| Experiment tracking | Weights & Biases |
| Config | PyYAML, Click |

---

## Development

```bash
# Run tests
pytest tests/

# Format
black src/ && ruff check src/
```
