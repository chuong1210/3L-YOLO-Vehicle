# Hướng Dẫn Tối Ưu Nhận Diện Phương Tiện Giao Thông Trong Điều Kiện Thiếu Sáng Với 3L-YOLO

> **Tài liệu tham khảo chính:** Han, Z.; Yue, Z.; Liu, L. *3L-YOLO: A Lightweight Low-Light Object Detection Algorithm.* Appl. Sci. 2025, 15, 90.
>
> **Mục tiêu:** Áp dụng các kỹ thuật từ 3L-YOLO vào pipeline YOLO hiện có để cải thiện mAP khi nhận diện phương tiện giao thông (pedestrian, bicycle, motorbike, bus, truck, container truck, car) trong môi trường ánh sáng yếu.

---

## 1. Tổng Quan Vấn Đề

Nhận diện đối tượng trong điều kiện thiếu sáng gặp 3 thách thức lớn:

- **Độ tương phản yếu (low contrast):** Đối tượng gần như hòa lẫn vào nền, đặc biệt ở ban đêm hoặc đường hầm.
- **Nhiễu cao (high noise):** Cảm biến camera tạo nhiễu Gaussian và Poisson khi ánh sáng yếu.
- **Biên mờ (blurred boundaries):** Đường viền phương tiện không rõ ràng, dẫn đến bounding box sai lệch.

Hầu hết các phương pháp truyền thống xử lý bằng 2 bước: **tăng cường ảnh → phát hiện đối tượng**. Nhược điểm là tốn tài nguyên tính toán lớn (thêm 40–60 GFLOPs cho module tăng cường) và không phải lúc nào tăng cường ảnh cũng cải thiện kết quả — đôi khi còn gây thêm nhiễu và biến dạng.

**3L-YOLO giải quyết vấn đề này bằng cách loại bỏ hoàn toàn module tăng cường ảnh**, thay vào đó cải tiến trực tiếp kiến trúc mạng để trích xuất đặc trưng tốt hơn trong điều kiện ánh sáng yếu.

---

## 2. Kiến Trúc 3L-YOLO — 3 Cải Tiến Cốt Lõi

3L-YOLO dựa trên YOLOv8n và đưa ra 3 cải tiến chính:

### 2.1. C2f Module Cải Tiến Với Switchable Atrous Convolution (SAConv)

**Vấn đề:** Trong ảnh thiếu sáng, tín hiệu đối tượng yếu và bị chìm trong nền. Các convolution thông thường có receptive field nhỏ, không nắm bắt đủ thông tin ngữ cảnh toàn cục.

**Giải pháp:** Tích hợp SAConv vào module C2f của YOLOv8n.

**Cách hoạt động:**

- SAConv gồm 3 thành phần: Pre-Global Context → Switchable Atrous Convolution → Post-Global Context.
- Sử dụng 2 atrous convolution với dilation rate khác nhau (rate=1 và rate=3).
- Một hàm switch `S(x)` học cách phân bổ trọng số giữa 2 nhánh, cho phép trích xuất đặc trưng đa tỷ lệ linh hoạt.
- Công thức: `y = S(x)·Conv(x, w, 1) + (1 − S(x))·Conv(x, w + Δw, r)` với `r = 3`.

**Quy tắc thay thế:** Trong C2f module, tất cả convolution **trừ convolution đầu tiên và cuối cùng** được thay bằng SAConv. Cấu trúc còn lại của C2f giữ nguyên.

**Kết quả:** mAP@0.5 tăng 0.5% trên ExDark dataset, đồng thời giảm GFLOPs từ 8.1 xuống 7.4.

**Triển khai Python (ý tưởng):**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchableAtrousConv(nn.Module):
    """
    Switchable Atrous Convolution (SAConv).
    Kết hợp 2 atrous conv với dilation rate khác nhau
    thông qua một hàm switch học được.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=3):
        super().__init__()
        padding_1 = kernel_size // 2  # dilation=1
        padding_r = kernel_size // 2 * dilation  # dilation=r

        # Pre-Global Context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gc_conv = nn.Conv2d(in_channels, in_channels, 1)

        # Hai nhánh atrous convolution
        self.conv_d1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding_1, dilation=1)
        self.conv_dr = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride=stride, padding=padding_r, dilation=dilation)

        # Switch function
        self.switch_pool = nn.AvgPool2d(5, stride=1, padding=2)
        self.switch_conv = nn.Conv2d(in_channels, 1, 1)

        # Post-Global Context
        self.post_gc_pool = nn.AdaptiveAvgPool2d(1)
        self.post_gc_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        # Pre-Global Context: thêm thông tin toàn cục vào input
        gc = self.gc_conv(self.global_pool(x))
        x = x + gc

        # Switch function: quyết định trọng số cho mỗi nhánh
        switch = torch.sigmoid(self.switch_conv(self.switch_pool(x)))

        # Hai nhánh atrous conv
        out_d1 = self.conv_d1(x)
        out_dr = self.conv_dr(x)

        # Kết hợp theo trọng số
        out = switch * out_d1 + (1 - switch) * out_dr

        # Post-Global Context
        post_gc = self.post_gc_conv(self.post_gc_pool(out))
        out = out + post_gc

        return out
```

### 2.2. Neck Module Đa Tỷ Lệ Với Channel Attention (ECA)

**Vấn đề:** Neck truyền thống của YOLOv8 (FPN + PAN) không tận dụng đủ các đặc trưng nông (shallow features), đặc biệt thiếu thông tin vị trí quan trọng cho các đối tượng nhỏ trong điều kiện thiếu sáng.

**Giải pháp:** Mở rộng FPN+PAN theo hướng BiFPN, kết hợp ECA attention.

**Chi tiết thiết kế:**

1. **Thêm kết nối ngang (horizontal connections):**
   - Fuse đặc trưng P2 vào P3 trong FPN → bảo toàn thông tin vị trí phong phú.
   - Merge đặc trưng P3, P4 vào P4, P5 trong PANet.
   - Sử dụng **feature concatenation** (không dùng feature weighting như BiFPN gốc).

2. **ECA Module sau mỗi bước concatenation:**
   - Global Average Pooling → 1D Convolution (kernel size tự thích ứng) → Sigmoid → Channel-wise weighting.
   - Kernel size `k` được tính tự động: `k = |log₂(C)/r + b/r|_odd` với `r=2, b=1`.
   - ECA cho phép tương tác cross-channel cục bộ mà không giảm chiều, rất nhẹ.

**Kết quả:** mAP@0.5 tăng 0.7% so với baseline.

**Triển khai ECA Module:**

```python
import math

class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA).
    Tự động tính kernel size dựa trên số kênh.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Tính kernel size tự thích ứng
        k_size = int(abs(math.log2(channels) / gamma + b / gamma))
        k_size = k_size if k_size % 2 else k_size + 1  # đảm bảo là số lẻ

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling: (B, C, H, W) → (B, C, 1, 1)
        y = self.avg_pool(x)

        # Reshape cho 1D conv: (B, C, 1, 1) → (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)

        # 1D Conv: local cross-channel interaction
        y = self.conv(y)

        # Reshape lại: (B, 1, C) → (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Sigmoid → channel weights
        y = self.sigmoid(y)

        # Channel-wise weighting
        return x * y.expand_as(x)
```

### 2.3. Dynamic Detection Head Với Deformable Convolution (DCNv3)

**Vấn đề:** Trong điều kiện thiếu sáng, đối tượng có sự biến đổi lớn về lớp, vị trí và tỷ lệ. Cần phân biệt đối tượng khỏi nền có độ tương phản thấp.

**Giải pháp:** Detection head cascade 3 loại attention: Spatial → Scale → Channel.

**Chi tiết:**

1. **Spatial Attention (DCNv3):** Sử dụng deformable convolution v3 để tự học offset cho các sampling point, giúp tập trung vào vùng đối tượng thay vì nền. DCNv3 cải tiến hơn DCNv2 nhờ separable convolution, cơ chế multi-group, và chuẩn hóa modulation scalar.

2. **Scale Attention:** Global pooling + 1×1 Conv + Sigmoid → tính trọng số cho từng mức tỷ lệ, gộp đặc trưng đa tỷ lệ theo trọng số.

3. **Channel Attention:** Global pooling → 2 fully connected layers → Normalization → điều khiển ngưỡng chuyển đổi kênh đặc trưng.

**Công thức tổng hợp:**
```
W(F) = πC(πL(πS(F)·F)·F)·F
```
trong đó πS = Spatial, πL = Scale, πC = Channel attention.

**Kết quả:** Đây là cải tiến mang lại hiệu quả lớn nhất — mAP@0.5 tăng 3.2% khi sử dụng riêng lẻ.

### 2.4. MPDIoU Loss — Thay Thế CIoU

**Vấn đề:** CIoU loss khó tối ưu khi predicted box và ground truth có cùng tỷ lệ nhưng khác kích thước.

**Giải pháp:** MPDIoU không chỉ đo overlap mà còn tính khoảng cách giữa 2 góc (trên-trái và dưới-phải) của predicted box và ground truth.

```python
def mpdiou_loss(pred_boxes, target_boxes, img_w, img_h):
    """
    MPDIoU Loss.
    pred_boxes, target_boxes: (x1, y1, x2, y2) format.
    """
    # Tính IoU thông thường
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-7)

    # Khoảng cách góc trên-trái
    d1_sq = (target_boxes[:, 0] - pred_boxes[:, 0]) ** 2 + (target_boxes[:, 1] - pred_boxes[:, 1]) ** 2
    # Khoảng cách góc dưới-phải
    d2_sq = (target_boxes[:, 2] - pred_boxes[:, 2]) ** 2 + (target_boxes[:, 3] - pred_boxes[:, 3]) ** 2

    diagonal_sq = img_w ** 2 + img_h ** 2

    mpdiou = iou - d1_sq / diagonal_sq - d2_sq / diagonal_sq
    loss = 1 - mpdiou

    return loss.mean()
```

---

## 3. Kết Quả Thực Nghiệm Của 3L-YOLO

### 3.1. Ablation Study (ExDark Dataset)

| C2f_SAConv | MSFCA_Neck | DCNv3_Dyhead | mAP@0.5 (%) | Params (M) | GFLOPs |
|:---:|:---:|:---:|:---:|:---:|:---:|
| — | — | — | 66.1 | 3.01 | 8.1 |
| ✓ | — | — | 66.6 (+0.5) | 3.31 | 7.4 |
| — | ✓ | — | 66.6 (+0.5) | 3.24 | 8.5 |
| — | — | ✓ | 68.2 (+2.1) | 5.19 | 17.6 |
| ✓ | ✓ | — | 68.1 (+2.0) | 3.97 | 12.0 |
| ✓ | ✓ | ✓ | **68.8 (+2.7)** | 5.89 | 16.6 |

### 3.2. So Sánh Với Các Phương Pháp Khác (ExDark)

| Phương pháp | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs |
|---|:---:|:---:|:---:|:---:|
| YOLOv5n | 65.1% | 38.2% | 2.5 | 7.1 |
| YOLOv7-tiny | 63.5% | 35.5% | 6.04 | 13.3 |
| YOLOv8n | 66.1% | 39.6% | 3.01 | 8.1 |
| Zero_DCE + YOLOv8n | 63.9% | 38.5% | 3.08 | **73.7** |
| LOL-YOLO | 68.1% | 42.3% | 5.66 | 20.6 |
| **3L-YOLO** | **68.8%** | 42.0% | 5.89 | 16.6 |

**Điểm nổi bật:** 3L-YOLO đạt mAP cao nhất trong khi tiết kiệm 57 GFLOPs so với phương pháp dùng Zero_DCE image enhancement, và 4 GFLOPs so với LOL-YOLO.

---

## 4. Áp Dụng Vào Pipeline Nhận Diện Phương Tiện Giao Thông Của Bạn

Dựa trên notebook hiện tại của bạn (train YOLOv3 với 7 classes: pedestrian, bicycle, motorbike, bus, truck, container truck, car), dưới đây là các bước cụ thể để tích hợp kỹ thuật từ 3L-YOLO.

### 4.1. Tạo Dataset Low-Light Tổng Hợp (Tương Tự ExDark+)

Nếu dataset hiện tại chủ yếu là ảnh ban ngày, bạn cần tạo thêm ảnh thiếu sáng tổng hợp:

```python
import cv2
import numpy as np
from pathlib import Path

def synthesize_low_light(image, brightness_range=(0.6, 0.8),
                         gamma_range=(2.0, 5.0),
                         noise_std_range=(0.1, 0.3)):
    """
    Tổng hợp ảnh thiếu sáng từ ảnh gốc theo phương pháp 3L-YOLO.
    4 bước: Đánh giá → Giảm sáng → Gamma → Thêm nhiễu.
    """
    img = image.astype(np.float32) / 255.0

    # Bước 1: Kiểm tra — chỉ xử lý ảnh đủ sáng
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 80:  # ảnh đã tối, bỏ qua
        return image

    # Bước 2: Giảm sáng ngẫu nhiên xuống 60–80%
    brightness_factor = np.random.uniform(*brightness_range)
    img = img * brightness_factor

    # Bước 3: Gamma correction mô phỏng thiếu sáng
    gamma = np.random.uniform(*gamma_range)
    img = np.power(np.clip(img, 0, 1), gamma)

    # Bước 4: Thêm nhiễu Gaussian + Poisson
    noise_std = np.random.uniform(*noise_std_range)
    gaussian_noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    img = img + gaussian_noise

    # Nhiễu Poisson
    img_poisson = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_poisson = np.random.poisson(img_poisson.astype(np.float32) / 10.0) * 10.0
    img = np.clip(img_poisson, 0, 255).astype(np.uint8)

    return img


def augment_dataset_low_light(src_dir, dst_dir, ratio=0.5):
    """
    Tạo phiên bản low-light cho một phần dataset.
    ratio: tỷ lệ ảnh được tạo bản low-light (0.5 = 50%).
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    images = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png"))
    selected = np.random.choice(images, size=int(len(images) * ratio), replace=False)

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dark_img = synthesize_low_light(img)
        save_name = f"lowlight_{img_path.name}"
        cv2.imwrite(str(dst_path / save_name), dark_img)
        # Copy label file tương ứng (giữ nguyên annotation)
        label_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
        if label_path.exists():
            import shutil
            dst_label_dir = dst_path.parent / "labels"
            dst_label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(label_path, dst_label_dir / f"lowlight_{label_path.name}")

    print(f"Đã tạo {len(selected)} ảnh low-light tổng hợp")
```

### 4.2. Tối Ưu Cấu Hình Training Cho Điều Kiện Thiếu Sáng

Dựa trên cài đặt thực nghiệm của 3L-YOLO và notebook hiện tại của bạn, đây là các điều chỉnh được khuyến nghị:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # hoặc model pretrained của bạn

model.train(
    data="dataset.yaml",
    device=0,

    # === EPOCHS & PATIENCE ===
    epochs=200,              # 3L-YOLO train 200 epochs (bạn đang dùng 150)
    patience=50,             # giữ nguyên — cho model thêm thời gian

    # === IMAGE SIZE ===
    imgsz=640,               # giữ nguyên — 3L-YOLO cũng dùng 640

    # === BATCH & OPTIMIZER ===
    batch=24,                # 3L-YOLO dùng 24 (bạn đang dùng 8 — tăng nếu đủ VRAM)
    optimizer="SGD",         # 3L-YOLO dùng SGD (bạn đang dùng AdamW)
    lr0=0.01,                # 3L-YOLO dùng 0.01 (bạn đang dùng 0.001)
    momentum=0.937,          # 3L-YOLO dùng 0.937
    weight_decay=0.0005,     # 3L-YOLO dùng 5e-4

    # === AUGMENTATION CHO LOW-LIGHT ===
    # Tăng cường đa dạng màu sắc mô phỏng điều kiện ánh sáng khác nhau
    hsv_h=0.015,             # giữ nguyên — biến đổi hue nhẹ
    hsv_s=0.7,               # giữ nguyên — biến đổi saturation mạnh
    hsv_v=0.4,               # giữ nguyên — biến đổi value/brightness mạnh
    mosaic=1.0,
    mixup=0.2,               # giữ nguyên — tốt cho anti-overfitting
    copy_paste=0.1,
    scale=0.6,
    translate=0.1,
    degrees=10.0,
    fliplr=0.5,

    # === PERFORMANCE ===
    cache=True,
    workers=8,

    name="traffic_lowlight_optimized"
)
```

### 4.3. Tích Hợp Các Module 3L-YOLO Vào YOLOv8 (Custom Model)

Để tích hợp đầy đủ 3 cải tiến, bạn cần tạo custom YAML cho YOLOv8:

**Bước 1: Tạo file `custom_modules.py`**

```python
# custom_modules.py
import torch
import torch.nn as nn
import math

class ECA(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class SAConv2d(nn.Module):
    """Simplified Switchable Atrous Convolution"""
    def __init__(self, in_ch, out_ch, k=3, s=1, dilation=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, s, padding=k//2, dilation=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, k, s, padding=k//2*dilation, dilation=dilation)
        self.switch = nn.Sequential(
            nn.AvgPool2d(5, stride=1, padding=2),
            nn.Conv2d(in_ch, 1, 1),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        s = self.switch(x)
        out = s * self.conv1(x) + (1 - s) * self.conv2(x)
        return self.act(self.bn(out))
```

**Bước 2: Đăng ký modules với Ultralytics**

```python
# Thêm vào đầu script training
from ultralytics.nn.modules import conv, block
from custom_modules import ECA, SAConv2d

# Đăng ký custom modules
# (Cách chính xác phụ thuộc vào phiên bản ultralytics,
#  tham khảo docs: https://docs.ultralytics.com)
```

**Bước 3: Tạo custom model YAML**

```yaml
# 3l-yolo-traffic.yaml
nc: 7
names: ['pedestrian', 'bicycle', 'motorbike', 'bus', 'truck', 'container truck', 'car']

# Backbone giữ tương tự YOLOv8n, thay C2f bằng C2f_SAConv
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]  # TODO: thay bằng C2f_SAConv
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

# Neck: thêm ECA sau mỗi Concat, thêm kết nối P2→P3
neck:
  # ... (cấu trúc tùy chỉnh theo kiến trúc 3L-YOLO)
```

> **Lưu ý:** Tích hợp đầy đủ custom model YAML đòi hỏi kiến thức sâu về codebase Ultralytics. Nếu bạn mới bắt đầu, hãy ưu tiên áp dụng các kỹ thuật ở mục 4.2 (cấu hình training) và mục 4.1 (data augmentation) trước — hai phần này đã mang lại cải thiện đáng kể mà không cần chỉnh sửa kiến trúc mạng.

### 4.4. Post-Processing Cho Điều Kiện Thiếu Sáng

```python
def predict_low_light(model, image_path, conf_threshold=0.20):
    """
    Predict với cấu hình tối ưu cho ảnh thiếu sáng.
    Giảm conf threshold vì đối tượng trong điều kiện thiếu sáng
    thường có confidence thấp hơn bình thường.
    """
    results = model.predict(
        source=image_path,
        conf=conf_threshold,    # hạ xuống 0.20 thay vì 0.25 mặc định
        iou=0.5,                # NMS IoU threshold
        augment=True,           # Test-Time Augmentation — cải thiện recall
        agnostic_nms=False,
        max_det=100,
        device=0
    )
    return results
```

---

## 5. Quy Trình Tối Ưu Hoàn Chỉnh (Step-by-Step)

```
┌─────────────────────────────────────────────────────────┐
│                   BƯỚC 1: CHUẨN BỊ DỮ LIỆU              │
│                                                          │
│  1a. Thu thập thêm ảnh ban đêm / thiếu sáng thực tế     │
│  1b. Tổng hợp ảnh low-light từ ảnh gốc (mục 4.1)       │
│  1c. Trộn ảnh gốc + ảnh low-light (tỷ lệ 60:40)        │
│  1d. Đảm bảo cân bằng classes (đặc biệt bicycle, bus)   │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│               BƯỚC 2: CHỌN MODEL CƠ SỞ                  │
│                                                          │
│  • YOLOv8n: nhẹ nhất, phù hợp edge device               │
│  • YOLOv8s: cân bằng tốt giữa accuracy và speed         │
│  • Transfer learning từ COCO pretrained weights          │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│             BƯỚC 3: TỐI ƯU CẤU HÌNH TRAINING            │
│                                                          │
│  • SGD optimizer, lr=0.01, momentum=0.937                │
│  • 200 epochs, patience=50                               │
│  • batch=24, imgsz=640                                   │
│  • HSV augmentation mạnh (mô phỏng ánh sáng biến đổi)  │
│  • Mosaic + Mixup + Copy-paste                           │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│          BƯỚC 4: (NÂN CAO) TÍCH HỢP 3L-YOLO            │
│                                                          │
│  4a. Thay C2f → C2f_SAConv trong backbone                │
│  4b. Thêm ECA attention + shallow feature fusion ở neck  │
│  4c. Dynamic Head với DCNv3 attention cascade             │
│  4d. Thay CIoU loss → MPDIoU loss                        │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│              BƯỚC 5: ĐÁNH GIÁ & TRIỂN KHAI               │
│                                                          │
│  • Đánh giá riêng trên tập ảnh thiếu sáng               │
│  • Giảm conf threshold xuống 0.20 cho inference          │
│  • Bật Test-Time Augmentation (augment=True)             │
│  • Export ONNX/TensorRT cho triển khai thực tế           │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Các Mẹo Bổ Sung Cho Nhận Diện Giao Thông Ban Đêm

1. **Tăng cường HSV mạnh:** `hsv_v=0.4` là mức tốt — mô phỏng biến đổi độ sáng trên đường (đèn pha, đèn đường, bóng tối).

2. **Multi-scale training:** Bật `multi_scale=True` giúp model khái quát hóa tốt hơn cho các phương tiện ở khoảng cách khác nhau.

3. **Close mosaic cuối training:** Thêm `close_mosaic=15` — tắt mosaic 15 epoch cuối giúp fine-tune chính xác hơn.

4. **Ưu tiên Recall cho an toàn giao thông:** Trong ứng dụng giao thông, bỏ sót phương tiện (false negative) nguy hiểm hơn báo nhầm (false positive). Hạ `conf` threshold khi inference.

5. **Xử lý class imbalance:** Nếu class "pedestrian" hoặc "bicycle" ít mẫu hơn, cân nhắc dùng `copy_paste=0.2` hoặc oversampling.

6. **Loss function:** Nếu không thể tích hợp MPDIoU, YOLOv8 mặc định dùng CIoU vẫn là lựa chọn tốt. Ưu tiên tích hợp ECA và augmentation trước.

---

## 7. Tham Khảo

- Han, Z.; Yue, Z.; Liu, L. **3L-YOLO: A Lightweight Low-Light Object Detection Algorithm.** *Appl. Sci.* 2025, 15, 90. [DOI: 10.3390/app15010090](https://doi.org/10.3390/app15010090)
- Ultralytics YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- ECA-Net: Wang, Q. et al. CVPR 2020.
- DetectoRS (SAConv): Qiao, S. et al. CVPR 2021.
- Dynamic Head: Dai, X. et al. CVPR 2021.
- MPDIoU: Ma, S.; Xu, Y. arXiv 2023.
