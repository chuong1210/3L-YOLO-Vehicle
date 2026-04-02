"""
Pipeline Demo — lấy 1 frame từ file ảnh,
xử lý qua từng bước và lưu ảnh minh họa từng bước.

Step 1: Raw Frame
Step 2: YOLO Detection
Step 3: ByteTrack Tracking
Step 4: ROI Filter
Step 5: Line Counting IN/OUT
Step 6: Congestion Alert
Step 7: Full Pipeline Strip (poster composite)
"""

import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH  = "img0518-15911852124592083120330.webp"
MODEL_PATH  = "yolo26m_20261303.pt"
OUTPUT_DIR  = Path("pipeline_steps")
OUTPUT_DIR.mkdir(exist_ok=True)

# Màu sắc class
CLASS_COLORS = {
    "car":             (59,  130, 246),
    "motorbike":       (16,  185, 129),
    "truck":           (245, 158,  11),
    "bus":             (239,  68,  68),
    "pedestrian":      (168,  85, 247),
    "bicycle":         (236,  72, 153),
    "container truck": (14,  165, 233),
}
DEFAULT_COLOR = (156, 163, 175)

# Màu & nhãn mỗi step (BGR)
STEP_META = {
    1: {"color": (150, 150, 150), "en": "Raw Frame",        "vi": "Khung hinh goc"},
    2: {"color": (246, 130,  59), "en": "YOLO Detection",   "vi": "Nhan dien doi tuong"},
    3: {"color": (129, 185,  16), "en": "ByteTrack",        "vi": "Theo doi va dinh danh"},
    4: {"color": (246, 130, 246), "en": "ROI Filter",       "vi": "Loc vung quan sat"},
    5: {"color": ( 50, 200, 255), "en": "Line Counting",    "vi": "Dem xe qua duong ket"},
    6: {"color": ( 68,  68, 239), "en": "Congestion Alert", "vi": "Canh bao un tac"},
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_step_label(img: np.ndarray, step: int, detail: str = "") -> np.ndarray:
    """Header bar kiểu poster: badge số bước + tên EN + tên VI + chi tiết."""
    out    = img.copy()
    h, w   = out.shape[:2]
    meta   = STEP_META[step]
    color  = meta["color"]
    bar_h  = 68

    # Semi-transparent dark bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 15, 30), -1)
    cv2.addWeighted(overlay, 0.88, out, 0.12, 0, out)

    # Accent left strip
    cv2.rectangle(out, (0, 0), (5, bar_h), color, -1)

    # Circle badge step number
    cx, cy = 36, bar_h // 2
    cv2.circle(out, (cx, cy), 22, color, -1)
    cv2.circle(out, (cx, cy), 22, (255, 255, 255), 2)
    num_txt = str(step)
    (nw, nh), _ = cv2.getTextSize(num_txt, FONT_BOLD, 0.9, 2)
    cv2.putText(out, num_txt, (cx - nw // 2, cy + nh // 2),
                FONT_BOLD, 0.9, (10, 15, 30), 2, cv2.LINE_AA)

    # Title EN
    cv2.putText(out, meta["en"], (68, 28), FONT_BOLD, 0.85, color, 2, cv2.LINE_AA)

    # Subtitle VI
    sub = meta["vi"] + (f"  |  {detail}" if detail else "")
    cv2.putText(out, sub, (68, 55), FONT, 0.50, (180, 210, 240), 1, cv2.LINE_AA)

    # Bottom separator line
    cv2.line(out, (0, bar_h), (w, bar_h), color, 1)

    return out


def draw_box(img, x1, y1, x2, y2, label, conf, color):
    """Bounding box + label tag."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, y1 - 4), FONT, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


def save(img, step, name):
    path = OUTPUT_DIR / f"step{step}_{name}.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    kb = path.stat().st_size // 1024
    print(f"  Saved: {path}  ({kb} KB)")
    return img


def pt_in_poly(cx, cy, poly):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0


# ── Load image ────────────────────────────────────────────────────────────────
print(f"\n[0] Loading: {IMAGE_PATH}")
from PIL import Image as PILImage
pil_img = PILImage.open(IMAGE_PATH).convert("RGB")
frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
H, W    = frame.shape[:2]
print(f"    {W}x{H}")


# ── Step 1: Raw Frame ─────────────────────────────────────────────────────────
print("\n[1/6] Raw Frame")
s1 = frame.copy()

# Watermark info
cv2.putText(s1, f"{W} x {H} px   |   RTSP / Camera Feed",
            (20, H - 16), FONT, 0.6, (120, 140, 160), 1, cv2.LINE_AA)

s1 = add_step_label(s1, 1, f"{W}x{H}px")
save(s1, 1, "raw_frame")


# ── Step 2: YOLO Detection ────────────────────────────────────────────────────
print("\n[2/6] YOLO Detection")
try:
    from ultralytics import YOLO
    model     = YOLO(MODEL_PATH)
    results   = model(frame, verbose=False, conf=0.25, imgsz=640)[0]
    boxes_data = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id  = int(box.cls[0])
        conf    = float(box.conf[0])
        label   = model.names.get(cls_id, f"cls{cls_id}")
        boxes_data.append((x1, y1, x2, y2, label, conf))
    print(f"    Detected {len(boxes_data)} objects")
except Exception as e:
    print(f"    YOLO fallback ({e})")
    boxes_data = [
        (200, 300, 380, 430, "car",       0.92),
        (500, 280, 720, 420, "truck",     0.87),
        (800, 350, 950, 460, "motorbike", 0.81),
        (100, 400, 250, 510, "car",       0.78),
        (650, 420, 800, 520, "bus",       0.85),
    ]

s2 = frame.copy()
for (x1, y1, x2, y2, label, conf) in boxes_data:
    draw_box(s2, x1, y1, x2, y2, label, conf, CLASS_COLORS.get(label, DEFAULT_COLOR))

# Confidence legend (góc trên phải)
legend_x = W - 220
cv2.rectangle(s2, (legend_x - 8, 80), (W - 8, 80 + len(CLASS_COLORS) * 26 + 14),
              (10, 15, 30), -1)
for i, (cls, col) in enumerate(CLASS_COLORS.items()):
    y = 100 + i * 26
    cv2.rectangle(s2, (legend_x, y - 12), (legend_x + 16, y + 4), col, -1)
    cv2.putText(s2, cls, (legend_x + 22, y), FONT, 0.48, (210, 220, 235), 1, cv2.LINE_AA)

s2 = add_step_label(s2, 2, f"{len(boxes_data)} objects  |  conf > 0.25")
save(s2, 2, "yolo_detection")


# ── Step 3: ByteTrack ─────────────────────────────────────────────────────────
print("\n[3/6] ByteTrack Tracking")
s3 = frame.copy()

track_palette = [
    (59, 130, 246), (16, 185, 129), (245, 158, 11),
    (239, 68, 68),  (168, 85, 247), (236, 72, 153),
]
np.random.seed(42)

for i, (x1, y1, x2, y2, label, conf) in enumerate(boxes_data):
    color = track_palette[i % len(track_palette)]
    tid   = 10 + i * 7
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Vẽ trail giả lập
    trail = [
        (cx + np.random.randint(-20, 8) * j // 4,
         cy + np.random.randint(-4, 4)  + j * 6)
        for j in range(5, 0, -1)
    ]
    for j in range(len(trail) - 1):
        alpha = (j + 2) / (len(trail) + 1)
        tc    = tuple(int(c * alpha) for c in color)
        cv2.line(s3, trail[j], trail[j + 1], tc, 2, cv2.LINE_AA)

    # Box
    cv2.rectangle(s3, (x1, y1), (x2, y2), color, 2)

    # Track ID badge (pill)
    tid_txt = f"# {tid}"
    (tw, th), _ = cv2.getTextSize(tid_txt, FONT_BOLD, 0.65, 2)
    bx1, by1 = cx - tw // 2 - 6, cy - th - 10
    bx2, by2 = cx + tw // 2 + 6, cy + 4
    cv2.rectangle(s3, (bx1, by1), (bx2, by2), color, -1)
    cv2.rectangle(s3, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
    cv2.putText(s3, tid_txt, (cx - tw // 2, cy),
                FONT_BOLD, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

s3 = add_step_label(s3, 3, f"{len(boxes_data)} tracks  |  Trail history")
save(s3, 3, "bytetrack")


# ── Step 4: ROI Filter ────────────────────────────────────────────────────────
print("\n[4/6] ROI Filter")
s4 = frame.copy()

roi_pts = np.array([
    [int(W * 0.12), int(H * 0.28)],
    [int(W * 0.88), int(H * 0.28)],
    [int(W * 0.92), int(H * 0.88)],
    [int(W * 0.08), int(H * 0.88)],
], dtype=np.int32)

# Dim ngoài ROI
mask          = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(mask, [roi_pts], 255)
dark          = (s4 * 0.30).astype(np.uint8)
roi_3ch       = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0
s4            = np.where(roi_3ch, s4, dark)

# ROI border với glow effect (3 lớp viền độ trong khác nhau)
for thickness, alpha_mul in [(8, 0.25), (4, 0.5), (2, 1.0)]:
    overlay2 = s4.copy()
    cv2.polylines(overlay2, [roi_pts], True, (246, 130, 246), thickness)
    cv2.addWeighted(overlay2, alpha_mul, s4, 1 - alpha_mul, 0, s4)

# Góc điểm ROI
for pt in roi_pts:
    cv2.circle(s4, tuple(pt), 9, (246, 130, 246), -1)
    cv2.circle(s4, tuple(pt), 9, (255, 255, 255), 2)

# "ROI Zone" text ở giữa
mid = roi_pts.mean(axis=0).astype(int)
(rw, rh), _ = cv2.getTextSize("ROI Zone", FONT_BOLD, 1.1, 2)
cv2.putText(s4, "ROI Zone",
            (mid[0] - rw // 2, mid[1] + rh // 2),
            FONT_BOLD, 1.1, (246, 130, 246), 2, cv2.LINE_AA)

# Box xe: xanh = trong ROI, xám = ngoài
in_count = 0
for (x1, y1, x2, y2, label, conf) in boxes_data:
    cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
    inside  = pt_in_poly(cx, cy, roi_pts)
    color   = (129, 185, 16) if inside else (80, 80, 80)
    tag     = "IN ROI" if inside else "outside"
    if inside:
        in_count += 1
    cv2.rectangle(s4, (x1, y1), (x2, y2), color, 2)
    cv2.putText(s4, tag, (x1, y1 - 6), FONT, 0.55, color, 1, cv2.LINE_AA)

s4 = add_step_label(s4, 4, f"{in_count}/{len(boxes_data)} inside ROI")
save(s4, 4, "roi_filter")


# ── Step 5: Line Counting ─────────────────────────────────────────────────────
print("\n[5/6] Line Counting")
s5 = frame.copy()

line_y = int(H * 0.55)

# Counting line gradient (blue → red)
for x in range(0, W, 3):
    t = x / W
    c = (int(59 + (239 - 59) * t), int(130 + (68 - 130) * t), int(246 + (68 - 246) * t))
    cv2.line(s5, (x, line_y - 1), (x + 3, line_y - 1), c, 2)
    cv2.line(s5, (x, line_y + 1), (x + 3, line_y + 1), c, 2)

# Nhãn line
(lw, lh), _ = cv2.getTextSize("COUNTING LINE", FONT_BOLD, 0.72, 2)
lx = W // 2 - lw // 2
cv2.rectangle(s5, (lx - 8, line_y - lh - 14), (lx + lw + 8, line_y - 4),
              (10, 15, 30), -1)
cv2.putText(s5, "COUNTING LINE", (lx, line_y - 8),
            FONT_BOLD, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

# Mũi tên IN ↓
for xi in [W // 5, W // 2, 4 * W // 5]:
    cv2.arrowedLine(s5, (xi, line_y - 70), (xi, line_y - 12),
                    (129, 185, 16), 3, cv2.LINE_AA, tipLength=0.35)
cv2.putText(s5, "IN  (top → bottom)", (W // 5 - 10, line_y - 80),
            FONT, 0.65, (129, 185, 16), 2, cv2.LINE_AA)

# Mũi tên OUT ↑
for xi in [W // 3, 2 * W // 3]:
    cv2.arrowedLine(s5, (xi, line_y + 70), (xi, line_y + 12),
                    (68, 68, 239), 3, cv2.LINE_AA, tipLength=0.35)
cv2.putText(s5, "OUT (bottom → top)", (W // 3 - 10, line_y + 95),
            FONT, 0.65, (68, 68, 239), 2, cv2.LINE_AA)

# Counter panel (góc phải dưới)
px, py = W - 240, H - 170
cv2.rectangle(s5, (px - 10, py - 10), (W - 10, H - 10), (10, 15, 30), -1)
cv2.rectangle(s5, (px - 10, py - 10), (W - 10, H - 10), (50, 200, 255), 2)
cv2.putText(s5, "COUNTER", (px, py + 22), FONT_BOLD, 0.72, (200, 220, 255), 2)
cv2.line(s5, (px - 10, py + 32), (W - 10, py + 32), (50, 200, 255), 1)
cv2.putText(s5, "IN  :  47", (px, py + 70), FONT_BOLD, 1.0, (129, 185, 16), 2, cv2.LINE_AA)
cv2.putText(s5, "OUT :  12", (px, py + 115), FONT_BOLD, 1.0, (68, 68, 239), 2, cv2.LINE_AA)
cv2.putText(s5, "TOTAL: 59", (px, py + 148), FONT, 0.65, (200, 220, 255), 1, cv2.LINE_AA)

s5 = add_step_label(s5, 5, "IN=47  OUT=12  TOTAL=59")
save(s5, 5, "line_counting")


# ── Step 6: Congestion Alert ──────────────────────────────────────────────────
print("\n[6/6] Congestion Alert")
s6 = frame.copy()

# Tất cả box màu đỏ mờ
for (x1, y1, x2, y2, label, conf) in boxes_data:
    overlay_b = s6.copy()
    cv2.rectangle(overlay_b, (x1, y1), (x2, y2), (40, 40, 200), -1)
    cv2.addWeighted(overlay_b, 0.25, s6, 0.75, 0, s6)
    cv2.rectangle(s6, (x1, y1), (x2, y2), (68, 68, 239), 2)

# Dim toàn frame nhẹ
dim = s6.copy()
cv2.rectangle(dim, (0, 0), (W, H), (20, 20, 60), -1)
cv2.addWeighted(dim, 0.35, s6, 0.65, 0, s6)

# Alert banner trung tâm
bh      = 90
by      = H // 2 - bh // 2
banner  = s6.copy()
cv2.rectangle(banner, (0, by), (W, by + bh), (30, 20, 180), -1)
cv2.addWeighted(banner, 0.90, s6, 0.10, 0, s6)
# Border viền alert
cv2.rectangle(s6, (4, by + 2), (W - 4, by + bh - 2), (100, 80, 255), 2)

(aw, ah), _ = cv2.getTextSize("!! CONGESTION DETECTED !!", FONT_BOLD, 1.1, 3)
cv2.putText(s6, "!! CONGESTION DETECTED !!",
            (W // 2 - aw // 2, by + 44),
            FONT_BOLD, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
cv2.putText(s6, "15 vehicles  |  wait 23s  |  threshold: 10",
            (W // 2 - 250, by + 76),
            FONT, 0.68, (200, 180, 255), 2, cv2.LINE_AA)

# Traffic light widget (góc trên phải)
tl_x, tl_y, tl_r = W - 90, 80, 24
box_pad = 14
cv2.rectangle(s6, (tl_x - box_pad, tl_y - box_pad),
              (tl_x + tl_r * 2 + box_pad, tl_y + tl_r * 6 + box_pad + 30),
              (20, 20, 20), -1)
cv2.rectangle(s6, (tl_x - box_pad, tl_y - box_pad),
              (tl_x + tl_r * 2 + box_pad, tl_y + tl_r * 6 + box_pad + 30),
              (80, 80, 80), 2)
# Red ON (glow)
for offset, alpha in [(tl_r + 6, 0.2), (tl_r + 2, 0.4)]:
    glow = s6.copy()
    cv2.circle(glow, (tl_x + tl_r, tl_y + tl_r), offset, (50, 50, 200), -1)
    cv2.addWeighted(glow, alpha, s6, 1 - alpha, 0, s6)
cv2.circle(s6, (tl_x + tl_r, tl_y + tl_r), tl_r, (68, 68, 239), -1)
# Yellow OFF
cv2.circle(s6, (tl_x + tl_r, tl_y + tl_r * 3), tl_r, (30, 30, 30), -1)
cv2.circle(s6, (tl_x + tl_r, tl_y + tl_r * 3), tl_r, (60, 60, 60), 1)
# Green OFF
cv2.circle(s6, (tl_x + tl_r, tl_y + tl_r * 5), tl_r, (20, 40, 20), -1)
cv2.circle(s6, (tl_x + tl_r, tl_y + tl_r * 5), tl_r, (40, 60, 40), 1)
cv2.putText(s6, "RED",
            (tl_x + 2, tl_y + tl_r * 6 + 20),
            FONT, 0.55, (68, 68, 239), 1, cv2.LINE_AA)

s6 = add_step_label(s6, 6, "15 vehicles  |  level: CRITICAL")
save(s6, 6, "congestion_alert")


# ── Step 7: Full Pipeline Strip ───────────────────────────────────────────────
print("\n[7] Pipeline Strip — All steps")

all_steps = [s1, s2, s3, s4, s5, s6]

TW  = 380                              # width mỗi thumbnail
TH  = int(TW * H / W)                  # giữ tỉ lệ
GAP = 6                                # khoảng cách giữa các thumbnail
AW  = 42                               # arrow width
HDR = 72                               # header height

# Resize từng step
thumbs = [cv2.resize(s, (TW, TH)) for s in all_steps]

# Tổng chiều rộng
total_w = len(thumbs) * TW + (len(thumbs) - 1) * (GAP + AW)
total_h = HDR + TH

canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
canvas[:] = (12, 18, 36)  # dark navy background

# Header
cv2.rectangle(canvas, (0, 0), (total_w, HDR), (18, 26, 50), -1)
title_txt = "Vehicle Detection Pipeline  —  YOLOv8 + ByteTrack + ROI + Line Counting + Congestion"
(tw, th), _ = cv2.getTextSize(title_txt, FONT_BOLD, 0.72, 2)
cv2.putText(canvas, title_txt, ((total_w - tw) // 2, 30),
            FONT_BOLD, 0.72, (180, 210, 255), 2, cv2.LINE_AA)

# Step labels bên dưới title
for i, meta in enumerate(STEP_META.values()):
    lbl_x = i * (TW + GAP + AW) + TW // 2
    color  = meta["color"]
    txt    = f"Step {i+1}"
    (lw, lh), _ = cv2.getTextSize(txt, FONT, 0.55, 1)
    cv2.putText(canvas, txt, (lbl_x - lw // 2, 58),
                FONT, 0.55, color, 1, cv2.LINE_AA)
cv2.line(canvas, (0, HDR - 2), (total_w, HDR - 2), (40, 60, 100), 1)

# Thumbnails + arrows
for i, thumb in enumerate(thumbs):
    x_off = i * (TW + GAP + AW)
    canvas[HDR: HDR + TH, x_off: x_off + TW] = thumb

    # Arrow → giữa các thumbnail
    if i < len(thumbs) - 1:
        ax     = x_off + TW + GAP
        ay_mid = HDR + TH // 2
        color  = STEP_META[i + 1]["color"]

        # Arrow shaft
        cv2.line(canvas, (ax, ay_mid), (ax + AW - 8, ay_mid), color, 3, cv2.LINE_AA)
        # Arrowhead
        pts = np.array([
            [ax + AW - 8, ay_mid - 8],
            [ax + AW,     ay_mid],
            [ax + AW - 8, ay_mid + 8],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], color)

path = OUTPUT_DIR / "step7_pipeline_strip.jpg"
cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 97])
kb = path.stat().st_size // 1024
print(f"  Saved: {path}  ({kb} KB)")

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\nDone! 7 images saved to: {OUTPUT_DIR.absolute()}")
print()
for f in sorted(OUTPUT_DIR.glob("step*.jpg")):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
