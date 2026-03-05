"""Scan a clip for banner confidence, sampling up to 150 frames."""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection.banner_detector import BannerDetector, ROI_X, ROI_Y, ROI_W, ROI_H

clip = sys.argv[1] if len(sys.argv) > 1 else "data/found/goal_clip_24m24s.mp4"
detector = BannerDetector()

cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Clip: {clip}  —  {total} frames @ {fps:.0f}fps")

# Sample up to 150 evenly-spaced frames
indices = np.linspace(0, total - 1, min(total, 150), dtype=int)

peak_conf = 0.0
peak_t = 0.0
peak_frame = None
peak_roi = None

for i in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ret, frame = cap.read()
    if not ret:
        continue
    conf = detector._match_template_in_frame(frame)
    t = i / fps
    if conf > 0.5:
        print(f"  t={t:.2f}s  frame={i}  conf={conf:.3f}")
    if conf > peak_conf:
        peak_conf = conf
        peak_t = t
        if conf > 0.7:
            peak_frame = frame.copy()
            h, w = frame.shape[:2]
            peak_roi = frame[int(ROI_Y*h):int((ROI_Y+ROI_H)*h),
                             int(ROI_X*w):int((ROI_X+ROI_W)*w)].copy()

cap.release()
print(f"\nPeak: t={peak_t:.2f}s  conf={peak_conf:.3f}  threshold={detector.threshold}")
print(f"Would detect: {peak_conf >= detector.threshold}")

if peak_frame is not None:
    cv2.imwrite("data/found/banner_frame_best.png", peak_frame)
    cv2.imwrite("data/found/banner_roi_best.png", peak_roi)
    print("Saved best frame → data/found/banner_frame_best.png")
    print("Saved ROI crop   → data/found/banner_roi_best.png")
