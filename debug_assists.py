"""Debug script: test assists-banner template matching at known timestamps."""
import cv2
import numpy as np

video = "data/processed/NHL 25_20260312194807_from24m_norm.mp4"
tpl = cv2.imread("configs/after_goal_assists_template2.png", cv2.IMREAD_GRAYSCALE)
print(f"Template: {tpl.shape[1]}x{tpl.shape[0]}")

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps} fps, {total} frames, {total/fps:.1f}s")

# ROI settings matching AssistsBannerDetector
ROI_X, ROI_Y, ROI_W, ROI_H = 0.0, 0.0, 0.55, 0.15
scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Test at 1s intervals around 4:37 (277s)
for t in range(270, 290):
    frame_idx = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    h, w = frame.shape[:2]
    x1, y1 = int(ROI_X * w), int(ROI_Y * h)
    x2, y2 = int((ROI_X + ROI_W) * w), int((ROI_Y + ROI_H) * h)
    roi = frame[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    best = 0.0
    best_scale = 0.0
    for scale in scales:
        th, tw = tpl.shape
        nw, nh = int(tw * scale), int(th * scale)
        if nw > roi_gray.shape[1] or nh > roi_gray.shape[0]:
            continue
        scaled = cv2.resize(tpl, (nw, nh))
        result = cv2.matchTemplate(roi_gray, scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best:
            best = max_val
            best_scale = scale

    marker = " <<<" if best >= 0.6 else ""
    print(f"t={t}s: conf={best:.3f} (scale={best_scale}){marker}")

    # Save frame at peak for visual inspection
    if best >= 0.5:
        cv2.imwrite(f"/tmp/assists_debug_t{t}.png", frame)

cap.release()
