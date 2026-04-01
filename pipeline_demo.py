"""
Pipeline Demo — lấy 1 frame từ RTSP (hoặc file ảnh),
xử lý qua từng bước và lưu ảnh minh họa từng bước.

Bước 1: Raw frame
Bước 2: YOLO detection (bounding boxes)
Bước 3: ByteTrack tracking (track IDs + trails)
Bước 4: ROI filter (polygon mask)
Bước 5: Line counting IN/OUT
Bước 6: Congestion alert overlay
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# ── Config ────────────────────────────────────────────────────────────────────
RTSP_URL = None   # không dùng RTSP
IMAGE_PATH = "img0518-15911852124592083120330.webp"
MODEL_PATH = "yolo26m_20261303.pt"
OUTPUT_DIR = Path("pipeline_steps")
OUTPUT_DIR.mkdir(exist_ok=True)

# Màu sắc
COLORS = {
    "car":            (59,  130, 246),   # blue
    "motorbike":      (16,  185, 129),   # green
    "truck":          (245, 158,  11),   # amber
    "bus":            (239,  68,  68),   # red
    "pedestrian":     (168,  85, 247),   # purple
    "bicycle":        (236,  72, 153),   # pink
    "container truck":(14,  165, 233),   # sky
}
DEFAULT_COLOR = (156, 163, 175)

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_label(img, text, pos=(20, 50), scale=1.4, color=(255,255,255)):
    """Thêm tiêu đề bước lên góc trên trái."""
    overlay = img.copy()
    # Background pill
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(overlay, (pos[0]-10, pos[1]-th-10), (pos[0]+tw+10, pos[1]+10), (15,23,42), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def draw_box(img, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
    cv2.putText(img, text, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

def save(img, step, name):
    path = OUTPUT_DIR / f"step{step}_{name}.jpg"
    cv2.imwrite(str(path), img)
    print(f"  Saved: {path}")
    return path

# ── Step 0: Load image ────────────────────────────────────────────────────────

print(f"\n[0/6] Loading image: {IMAGE_PATH}")
# dùng PIL để đọc webp, rồi convert sang BGR cho OpenCV
from PIL import Image as PILImage
pil_img = PILImage.open(IMAGE_PATH).convert("RGB")
frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

H, W = frame.shape[:2]
print(f"  Frame: {W}x{H}")

# ── Step 1: Raw frame ─────────────────────────────────────────────────────────

print("\n[1/6] Step 1 — Raw frame")
s1 = frame.copy()
add_label(s1, "BUOC 1: Raw RTSP Frame", color=(100, 220, 100))
cv2.putText(s1, f"{W}x{H}  |  RTSP Stream", (20, H-20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1, cv2.LINE_AA)
save(s1, 1, "raw_frame")

# ── Step 2: YOLO detection ────────────────────────────────────────────────────

print("\n[2/6] Step 2 — YOLO Detection")
try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    results = model(frame, verbose=False, conf=0.25, imgsz=640)[0]
    boxes_data = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names.get(cls_id, f"cls{cls_id}")
        boxes_data.append((x1, y1, x2, y2, label, conf))
    print(f"  Detected {len(boxes_data)} objects")
except Exception as e:
    print(f"  YOLO not available ({e}), using mock detections")
    # Mock detections cho demo
    boxes_data = [
        (200, 300, 380, 430, "car",       0.92),
        (500, 280, 720, 420, "truck",     0.87),
        (800, 350, 950, 460, "motorbike", 0.81),
        (100, 400, 250, 510, "car",       0.78),
        (650, 420, 800, 520, "bus",       0.85),
    ]

s2 = frame.copy()
for (x1, y1, x2, y2, label, conf) in boxes_data:
    color = COLORS.get(label, DEFAULT_COLOR)
    draw_box(s2, x1, y1, x2, y2, label, conf, color)
add_label(s2, f"BUOC 2: YOLOv8 Detection  [{len(boxes_data)} objects]", color=(59, 200, 246))
save(s2, 2, "yolo_detection")

# ── Step 3: ByteTrack tracking ────────────────────────────────────────────────

print("\n[3/6] Step 3 — ByteTrack Tracking")
s3 = frame.copy()

# Giả lập track IDs và trails
track_colors = [(59,130,246),(16,185,129),(245,158,11),(239,68,68),(168,85,247)]
np.random.seed(42)

for i, (x1, y1, x2, y2, label, conf) in enumerate(boxes_data):
    tid = 10 + i * 7  # track IDs
    color = track_colors[i % len(track_colors)]
    cx, cy = (x1+x2)//2, (y1+y2)//2

    # Vẽ trail (giả lập quỹ đạo)
    trail_pts = [(cx + np.random.randint(-30, 10)*j//5,
                  cy + np.random.randint(-5, 5) + j*8)
                 for j in range(5, 0, -1)]
    for j in range(len(trail_pts)-1):
        alpha = (j+1) / len(trail_pts)
        tc = tuple(int(c * alpha) for c in color)
        cv2.line(s3, trail_pts[j], trail_pts[j+1], tc, 2)

    # Bounding box
    cv2.rectangle(s3, (x1, y1), (x2, y2), color, 2)

    # Track ID badge
    tid_text = f"#{tid}"
    (tw, th), _ = cv2.getTextSize(tid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(s3, (cx-tw//2-4, cy-th-8), (cx+tw//2+4, cy+4), color, -1)
    cv2.putText(s3, tid_text, (cx-tw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2, cv2.LINE_AA)

add_label(s3, "BUOC 3: ByteTrack — Track IDs + Trails", color=(16, 220, 129))
save(s3, 3, "bytetrack")

# ── Step 4: ROI filter ────────────────────────────────────────────────────────

print("\n[4/6] Step 4 — ROI Filter")
s4 = frame.copy()

# ROI polygon (giữa frame)
roi_pts = np.array([
    [int(W*0.15), int(H*0.3)],
    [int(W*0.85), int(H*0.3)],
    [int(W*0.9),  int(H*0.85)],
    [int(W*0.1),  int(H*0.85)],
], dtype=np.int32)

# Dim vùng ngoài ROI
mask = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(mask, [roi_pts], 255)
dark = s4.copy()
dark = (dark * 0.35).astype(np.uint8)
roi_mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
s4 = np.where(roi_mask_3ch, s4, dark)

# Vẽ viền ROI
cv2.polylines(s4, [roi_pts], True, (59, 130, 246), 3)

# Vẽ các điểm góc
for pt in roi_pts:
    cv2.circle(s4, tuple(pt), 8, (59, 130, 246), -1)
    cv2.circle(s4, tuple(pt), 8, (255, 255, 255), 2)

# Label ROI
mid = roi_pts.mean(axis=0).astype(int)
cv2.putText(s4, "ROI Zone", (mid[0]-50, mid[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (59, 130, 246), 2, cv2.LINE_AA)

# Xe trong/ngoài ROI
def pt_in_poly(cx, cy, poly):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

for (x1, y1, x2, y2, label, conf) in boxes_data:
    cx, cy = (x1+x2)//2, (y1+y2)//2
    inside = pt_in_poly(cx, cy, roi_pts)
    color = (16, 185, 129) if inside else (100, 100, 100)
    tag = "IN" if inside else "OUT"
    cv2.rectangle(s4, (x1, y1), (x2, y2), color, 2)
    cv2.putText(s4, tag, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

add_label(s4, "BUOC 4: ROI Filter — Chi xet xe trong vung", color=(59, 130, 246))
save(s4, 4, "roi_filter")

# ── Step 5: Line counting ─────────────────────────────────────────────────────

print("\n[5/6] Step 5 — Line Counting")
s5 = frame.copy()

line_y = int(H * 0.55)

# Counting line gradient
for x in range(0, W, 4):
    t = x / W
    c = (int(59 + (239-59)*t), int(130 + (68-130)*t), int(246 + (68-246)*t))
    cv2.line(s5, (x, line_y), (x+4, line_y), c, 3)

# Nhãn LINE
cv2.putText(s5, "COUNTING LINE", (W//2-90, line_y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# Mũi tên IN (đi xuống, xanh lá)
for xi in [W//4, W//2, 3*W//4]:
    cv2.arrowedLine(s5, (xi, line_y-60), (xi, line_y-10),
                    (16, 185, 129), 3, tipLength=0.4)
cv2.putText(s5, "IN  (top->bottom)", (W//4-20, line_y-70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 185, 129), 2, cv2.LINE_AA)

# Mũi tên OUT (đi lên, đỏ)
for xi in [W//3, 2*W//3]:
    cv2.arrowedLine(s5, (xi, line_y+60), (xi, line_y+10),
                    (239, 68, 68), 3, tipLength=0.4)
cv2.putText(s5, "OUT (bottom->top)", (W//3-20, line_y+85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (239, 68, 68), 2, cv2.LINE_AA)

# Counter panel
panel_x, panel_y = W-260, 80
cv2.rectangle(s5, (panel_x, panel_y), (W-20, panel_y+120), (15, 23, 42), -1)
cv2.rectangle(s5, (panel_x, panel_y), (W-20, panel_y+120), (59, 130, 246), 2)
cv2.putText(s5, "IN  : 47", (panel_x+15, panel_y+45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (16, 185, 129), 2, cv2.LINE_AA)
cv2.putText(s5, "OUT : 12", (panel_x+15, panel_y+90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (239, 68, 68), 2, cv2.LINE_AA)

add_label(s5, "BUOC 5: Line Counting IN / OUT", color=(255, 200, 50))
save(s5, 5, "line_counting")

# ── Step 6: Congestion alert ──────────────────────────────────────────────────

print("\n[6/6] Step 6 — Congestion Alert")
s6 = s5.copy()  # kế thừa từ step 5

# Tất cả bounding boxes mờ đỏ
for (x1, y1, x2, y2, label, conf) in boxes_data:
    cv2.rectangle(s6, (x1, y1), (x2, y2), (239, 68, 68), 2)

# Alert banner overlay
banner_h = 80
overlay = s6.copy()
cv2.rectangle(overlay, (0, H//2-banner_h//2), (W, H//2+banner_h//2), (220, 38, 38), -1)
cv2.addWeighted(overlay, 0.82, s6, 0.18, 0, s6)

cv2.putText(s6, "!! KET XE NGHIEM TRONG !!",
            (W//2-330, H//2+12), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(s6, "15 phuong tien | 23s | nguong: 10",
            (W//2-270, H//2+45), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 200, 200), 2, cv2.LINE_AA)

# Traffic light simulation (góc phải trên)
tl_x, tl_y = W-110, 20
cv2.rectangle(s6, (tl_x, tl_y), (tl_x+80, tl_y+200), (30, 30, 30), -1)
cv2.rectangle(s6, (tl_x, tl_y), (tl_x+80, tl_y+200), (80, 80, 80), 2)
# Red ON
cv2.circle(s6, (tl_x+40, tl_y+40),  28, (239, 68, 68), -1)
# Yellow OFF
cv2.circle(s6, (tl_x+40, tl_y+100), 28, (60, 50, 20), -1)
# Green OFF
cv2.circle(s6, (tl_x+40, tl_y+160), 28, (20, 50, 30), -1)
cv2.putText(s6, "RED", (tl_x+18, tl_y+195),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (239, 68, 68), 1)

add_label(s6, "BUOC 6: Congestion Alert + Traffic Light", color=(239, 68, 68))
save(s6, 6, "congestion_alert")

# ── Done ──────────────────────────────────────────────────────────────────────

print(f"\nDone! 6 images saved to: {OUTPUT_DIR.absolute()}")
print("\nFiles:")
for f in sorted(OUTPUT_DIR.glob("step*.jpg")):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name}  ({size_kb} KB)")
