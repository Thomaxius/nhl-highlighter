"""Test EOG detection (with OCR verification) across all segments of the cut norm clip."""
import logging, sys, cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
sys.path.insert(0, "/Users/tsantamaa/home-projects/nhl-highlighter")

from src.detection.pause_menu_detector import PauseMenuDetector

SEGS_DIR  = Path("data/processed/segments/NHL 25_20260312181120_cut_norm")
TMPL_EOG  = Path("configs/pause_menu_template_end_of_game.png")

eog_detector = PauseMenuDetector(template_path=TMPL_EOG, threshold=0.60, sample_frames=6)

segments = sorted(SEGS_DIR.glob("*.mp4"))
print(f"Scanning {len(segments)} segments with OCR verification...\n")

# Build cumulative timeline so we can report wall-clock time in the full clip
cum_t = 0.0
for seg in segments:
    cap = cv2.VideoCapture(str(seg))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    dur = n / fps
    start_t = cum_t
    cum_t  += dur

    res = eog_detector.detect(seg, require_ocr_text=["END OF GAME", "END OF", "END"])
    status = "HIT " if res["detected"] else f"    "
    ocr_snippet = res.get("ocr_text", "").replace("\n", " ").strip()[:60]
    print(
        f"{status} {seg.name}  conf={res['max_conf']:.3f}  "
        f"t={start_t:.0f}-{cum_t:.0f}s"
        + (f"  OCR: {ocr_snippet!r}" if res.get("ocr_text") else "")
    )
