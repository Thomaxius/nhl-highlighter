"""Quick test: EOG template confidence at key timestamps in the norm clip."""
import cv2

VIDEO = "data/processed/NHL 25_20260312181120_cut_norm.mp4"
TMPL  = "configs/pause_menu_template_end_of_game.png"

tmpl = cv2.cvtColor(cv2.imread(TMPL), cv2.COLOR_BGR2GRAY)
cap  = cv2.VideoCapture(VIDEO)
fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
roi_y2 = max(int(h * 0.20), tmpl.shape[0] + 1)
roi_x2 = max(int(w * 0.35), tmpl.shape[1] + 1)
total  = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
print(f"Video duration: {total:.1f}s  Template {tmpl.shape[1]}x{tmpl.shape[0]}  ROI {roi_x2}x{roi_y2}\n")

windows = [
    ("After OT goal — searching for EOG screen", range(1060, 1200, 5)),
]

for label, rng in windows:
    print(f"--- {label} ---")
    for t_s in rng:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_s * fps))
        ret, fr = cap.read()
        if not ret:
            print(f"  t={t_s}s  (no frame)")
            continue
        roi = cv2.cvtColor(fr[:roi_y2, :roi_x2], cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        print(f"  t={t_s}s  conf={res.max():.4f}")
    print()

cap.release()
