"""
Pipeline tạo dữ liệu thiếu sáng — từng bước có ảnh minh họa.
"""
import cv2
import numpy as np
from pathlib import Path

SRC = r"c:\Users\chuon\Downloads\Optimize\vid6_mp4-92_jpg.rf.964c3ff9555626d1e10031e549c75f36.jpg"
OUT = Path(r"c:\Users\chuon\Downloads\Optimize\lowlight_steps")
OUT.mkdir(exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX

def add_label(img, step_num, title, subtitle=""):
    out = img.copy()
    h, w = out.shape[:2]
    # Dark bar top
    cv2.rectangle(out, (0,0), (w, 52), (0,0,0), -1)
    cv2.putText(out, f"Step {step_num}: {title}", (10,22), FONT, 0.65, (255,255,255), 2)
    if subtitle:
        cv2.putText(out, subtitle, (10,44), FONT, 0.45, (180,220,255), 1)
    return out

img = cv2.imread(SRC)
assert img is not None, f"Không tìm thấy ảnh: {SRC}"
h, w = img.shape[:2]
# Resize để đồng nhất
img = cv2.resize(img, (960, int(960*h/w)))
h, w = img.shape[:2]

rng = np.random.default_rng(42)

# ── Step 0: Ảnh gốc ──────────────────────────────────────────────────────────
s0 = add_label(img, 0, "Anh goc (ban ngay)", "Input: bien kich thuoc 960px")
cv2.imwrite(str(OUT/"step0_original.jpg"), s0, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("Step 0 saved")

# ── Step 1: Giảm độ sáng ─────────────────────────────────────────────────────
brightness = 0.70  # random in [0.6, 0.8] — dùng 0.70 cho minh họa
f = img.astype(np.float32) / 255.0
f_dark = np.clip(f * brightness, 0, 1)
img_dark = (f_dark * 255).astype(np.uint8)

s1 = add_label(img_dark, 1, "Giam do sang", f"factor = {brightness} (random 0.6~0.8)")
cv2.imwrite(str(OUT/"step1_brightness.jpg"), s1, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("Step 1 saved")

# ── Step 2: Gamma Correction ──────────────────────────────────────────────────
gamma = 3.0  # random in [2.0, 5.0]
f_gamma = np.power(np.clip(f_dark, 0, 1), gamma)
img_gamma = (f_gamma * 255).astype(np.uint8)

s2 = add_label(img_gamma, 2, "Gamma Correction", f"gamma = {gamma} (random 2.0~5.0)  |  I_out = I_in ^ gamma")
cv2.imwrite(str(OUT/"step2_gamma.jpg"), s2, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("Step 2 saved")

# ── Step 3: Nhiễu Gaussian ────────────────────────────────────────────────────
noise_std = 0.18
gauss = rng.normal(0, noise_std, f_gamma.shape).astype(np.float32)
f_gauss = np.clip(f_gamma + gauss, 0, 1)
img_gauss = (f_gauss * 255).astype(np.uint8)

s3 = add_label(img_gauss, 3, "Nhieu Gaussian", f"std = {noise_std}  |  N ~ N(0, {noise_std}^2)")
cv2.imwrite(str(OUT/"step3_gaussian_noise.jpg"), s3, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("Step 3 saved")

# ── Step 4: Nhiễu Poisson ─────────────────────────────────────────────────────
img_p = np.clip(f_gauss * 255, 0, 255).astype(np.uint8)
img_poisson = (rng.poisson(img_p.astype(np.float32) / 10.0) * 10.0)
img_final = np.clip(img_poisson, 0, 255).astype(np.uint8)

s4 = add_label(img_final, 4, "Nhieu Poisson", "X_poi ~ Poisson(pixel/10) x 10  |  mo phong cam bien camera")
cv2.imwrite(str(OUT/"step4_poisson_noise.jpg"), s4, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("Step 4 saved")

# ── Step 5: So sánh Before / After ──────────────────────────────────────────
# Ghép 2 ảnh trái/phải
left = img.copy()
right = img_final.copy()

# Vẽ nhãn lớn
def big_label(im, text, color):
    out = im.copy()
    cv2.rectangle(out, (0, out.shape[0]-48), (out.shape[1], out.shape[0]), (0,0,0), -1)
    cv2.putText(out, text, (12, out.shape[0]-14), FONT, 0.9, color, 2)
    return out

left  = big_label(left,  "TRUOC (ban ngay)", (100, 220, 100))
right = big_label(right, "SAU (thieu sang)", (80, 120, 255))

# Thêm đường phân cách
divider = np.zeros((h, 6, 3), dtype=np.uint8)
divider[:] = (255, 255, 255)
compare = np.hstack([left, divider, right])

# Header
header = np.zeros((56, compare.shape[1], 3), dtype=np.uint8)
cv2.putText(header, "So Sanh: Anh Goc vs Anh Thieu Sang Tong Hop",
            (10, 36), FONT, 0.75, (255, 255, 255), 2)
compare = np.vstack([header, compare])

cv2.imwrite(str(OUT/"step5_before_after.jpg"), compare, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("Step 5 saved")

# ── Step 6: Strip tất cả steps ───────────────────────────────────────────────
steps = [
    (img,       "Step 0\nOriginal"),
    (img_dark,  "Step 1\nBrightness"),
    (img_gamma, "Step 2\nGamma"),
    (img_gauss, "Step 3\nGaussian"),
    (img_final, "Step 4\nPoisson"),
]

TW = 220  # thumbnail width
TH = int(TW * h / w)
thumbs = []
for im, label in steps:
    th = cv2.resize(im, (TW, TH))
    # Label bar
    bar = np.zeros((38, TW, 3), dtype=np.uint8)
    for i, line in enumerate(label.split("\n")):
        cv2.putText(bar, line, (4, 16+i*18), FONT, 0.42, (200,230,255), 1)
    thumbs.append(np.vstack([th, bar]))

# Arrow between thumbnails
arrow = np.zeros((TH+38, 28, 3), dtype=np.uint8)
for y in range(TH//2 - 8, TH//2 + 8):
    cv2.arrowedLine(arrow, (2, y+8), (22, y+8), (100,200,100), 1, tipLength=0.5)

strip_parts = []
for i, t in enumerate(thumbs):
    strip_parts.append(t)
    if i < len(thumbs)-1:
        strip_parts.append(arrow)

strip = np.hstack(strip_parts)

# Header
header2 = np.zeros((46, strip.shape[1], 3), dtype=np.uint8)
cv2.putText(header2, "Pipeline Tao Du Lieu Thieu Sang (Low-Light Synthesis)",
            (8, 30), FONT, 0.62, (255,255,200), 2)
strip_final = np.vstack([header2, strip])

cv2.imwrite(str(OUT/"step6_pipeline_strip.jpg"), strip_final, [cv2.IMWRITE_JPEG_QUALITY, 95])
print("Step 6 (strip) saved")

print(f"\nDone! 7 anh luu tai: {OUT}")
files = sorted(OUT.glob("*.jpg"))
for f in files:
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name}  ({size_kb} KB)")
