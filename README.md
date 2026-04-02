# Deep Learning — Midterm & Final Project

WideResNet + SE Attention for image classification.

# Author: Van Cong Toan, Phan Dinh Trung

## Overview

|              | Midterm                       | Final                               |
| ------------ | ----------------------------- | ----------------------------------- |
| Dataset      | Animals-5 (custom, 2,750 ảnh) | CIFAR-10 (60K) & CIFAR-100 (60K)    |
| Model        | WRN-16-2 + SE (M1)            | WRN-28-4 + SE (M2)                  |
| Parameters   | ~0.7M                         | ~5.9M                               |
| Image size   | 224×224 và 32×32              | 32×32                               |
| Epochs       | 100                           | 200                                 |
| Best results | 32: 79.60%, 224: TBD          | CIFAR-10: 96.36%, CIFAR-100: 79.67% |

---

## 1. Kiến trúc chung — WideResNet + SE Attention

Cả midterm và final dùng cùng kiến trúc, chỉ khác depth và widen factor.

```
Input (IMG_SIZE × IMG_SIZE × 3)
  → Stem Conv 3×3 (S=1, P=1) → 16 channels
  → Group 1: WideResBlock × N (S=1)          → 16 × widen channels
  → Group 2: WideResBlock × N (first S=2)    → 32 × widen channels
  → Group 3: WideResBlock × N (first S=2)    → 64 × widen channels
  → BN → ReLU → AdaptiveAvgPool(1×1) → Flatten → FC → num_classes
```

**Mỗi WideResBlock:**

```
Input → BN → ReLU → Conv 3×3 → Dropout(0.3) → BN → ReLU → Conv 3×3 → SE Block → (+) → Output
  ↑                                                                                  ↑
  └──────────────────────── Skip connection (shortcut) ──────────────────────────────┘
```

**SE Block (Squeeze-and-Excitation):**

```
Input → AvgPool(1×1) → FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid → × Input
```

### So sánh M1 vs M2

| Thông số             | M1 (Midterm)                                        | M2 (Final)                                         |
| -------------------- | --------------------------------------------------- | -------------------------------------------------- |
| Depth                | 16                                                  | 28                                                 |
| Widen factor         | 2                                                   | 4                                                  |
| Blocks per group (N) | 2                                                   | 4                                                  |
| Channels             | 16 → 32 → 64 → 128                                  | 16 → 64 → 128 → 256                                |
| Parameters           | ~0.7M                                               | ~5.9M                                              |
| Lý do chọn           | Dataset nhỏ (2500 ảnh) → model nhỏ để tránh overfit | Dataset lớn (50K ảnh) → model lớn để tận dụng data |

### Tại sao model này mạnh?

1. **Skip connection** — gradient không bị mất khi train model sâu
2. **Pre-activation (BN → ReLU → Conv)** — chuẩn hoá trước convolution, train ổn định hơn
3. **SE Attention** — tự học channel nào quan trọng cho ảnh hiện tại
4. **Widen thay vì deeper** — tăng chiều rộng hiệu quả hơn tăng chiều sâu (theo paper WRN)

---

## 2. Midterm — Animals-5

### Dataset

| Thông số | Giá trị                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------ |
| Source   | [Animals-10 (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10) — chọn 5 class |
| Classes  | butterfly, cat, chicken, dog, horse                                                              |
| Train    | 500 ảnh/class × 5 = 2,500 ảnh                                                                    |
| Test     | 50 ảnh/class × 5 = 250 ảnh                                                                       |
| Dedup    | MD5 hash để loại ảnh trùng                                                                       |

### Training recipe

| Thành phần    | Cấu hình                                                     |
| ------------- | ------------------------------------------------------------ |
| Optimizer     | SGD (lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True) |
| Scheduler     | CosineAnnealingLR (100 epochs)                               |
| Loss          | CrossEntropyLoss (label_smoothing=0.1)                       |
| Augmentation  | Resize + RandomCrop + HFlip + ColorJitter + Cutout           |
| Cutout size   | IMG_SIZE // 4 (224→56, 32→8)                                 |
| Dropout       | 0.3                                                          |
| Batch size    | 8 (224×224), 128 (32×32)                                     |
| Normalization | ImageNet mean/std                                            |

### Kết quả

| Image Size | Best Test Acc | Train Acc (cuối) | Gap  | Ghi chú                |
| ---------- | ------------- | ---------------- | ---- | ---------------------- |
| 32×32      | 79.60%        | ~95%             | ~17% | Overfit do dataset nhỏ |
| 224×224    | TBD           | TBD              | TBD  | Chờ kết quả            |

### Phân tích overfit

WRN-28-4 (5.9M params) ban đầu cho gap 30% (train 98% vs test 73%). Giảm sang WRN-16-2 (0.7M params) cải thiện rõ rệt: gap giảm còn 17%, best test tăng từ 73% → 79%. Dataset 2,500 ảnh vẫn nhỏ so với 0.7M params nên overfit là không tránh khỏi.

### Cách chạy

1. Upload notebook `WRN_Animals_Kaggle.ipynb` lên Kaggle
2. **+ Add Input** → `animals-5-custom-train` và `animals-5-custom-test`
3. **Settings** → GPU T4
4. Lần 1: `IMG_SIZE = 224` → Run All
5. Lần 2: đổi `IMG_SIZE = 32` → Restart & Run All

Thời gian: ~30-60 phút (224), ~3-5 phút (32) trên T4.

---

## 3. Final — CIFAR-10 & CIFAR-100

### Dataset

|            | CIFAR-10 | CIFAR-100 |
| ---------- | -------- | --------- |
| Classes    | 10       | 100       |
| Train      | 50,000   | 50,000    |
| Test       | 10,000   | 10,000    |
| Image size | 32×32    | 32×32     |

### Training recipe

| Thành phần    | CIFAR-10                                                | CIFAR-100            |
| ------------- | ------------------------------------------------------- | -------------------- |
| Optimizer     | SGD (lr=0.1, momentum=0.9, wd=5e-4, nesterov)           | Giống                |
| Scheduler     | CosineAnnealingLR (200 epochs)                          | Giống                |
| Loss          | CrossEntropyLoss (label_smoothing=0.1)                  | Giống                |
| Augmentation  | RandomCrop(32,pad=4) + HFlip + AutoAugment + Cutout(16) | Cutout(8) thay vì 16 |
| Batch size    | 128                                                     | 128                  |
| Normalization | CIFAR-10 mean/std                                       | CIFAR-100 mean/std   |

**Cutout 16 (CIFAR-10) vs 8 (CIFAR-100):** CIFAR-100 có 100 class → cần giữ nhiều thông tin hơn nên cutout nhỏ hơn.

### Kết quả

| Dataset   | Best Test Acc | Train Acc (cuối) | Gap         | Overfit?                           |
| --------- | ------------- | ---------------- | ----------- | ---------------------------------- |
| CIFAR-10  | **96.36%**    | ~93%             | Val > Train | Không — do augmentation mạnh       |
| CIFAR-100 | **79.67%**    | ~89%             | ~10%        | Nhẹ — chấp nhận được cho 100 class |

**CIFAR-10 val > train:** Hiện tượng bình thường khi dùng AutoAugment + Cutout. Augmentation chỉ áp dụng lúc train → train khó hơn test → val accuracy cao hơn. Đây là dấu hiệu regularization tốt.

### Cách chạy

1. Upload `WRN_CIFAR10.ipynb` hoặc `WRN_CIFAR100.ipynb` lên Kaggle
2. **Settings** → GPU T4, Internet ON
3. Bấm **Run All** → chờ ~2-3 tiếng

---

## 4. Cấu trúc thư mục

```
├── midterm/
│   ├── WRN_Animals_Kaggle.ipynb       # Notebook train Animals-5 (224 & 32)
│   └── README.md
├── final/
│   ├── WRN_CIFAR10.ipynb              # Notebook train CIFAR-10
│   ├── WRN_CIFAR100.ipynb             # Notebook train CIFAR-100
│   └── README.md
├── reports/
│   ├── DL_Midterm_Report.docx         # Báo cáo giữa kỳ
│   ├── DL_Final_Report.docx           # Báo cáo cuối kỳ
│   └── DL_Final_Report.pdf
└── README.md                          # File này
```

## 5. Output files (mỗi lần train)

| File                                   | Mô tả                                                |
| -------------------------------------- | ---------------------------------------------------- |
| `best_model.pth`                       | Weights accuracy cao nhất — giám khảo load để verify |
| `last_checkpoint.pth`                  | Checkpoint để resume nếu bị disconnect               |
| `charts_*.png` / `training_curves.png` | Biểu đồ Loss và Accuracy                             |
| `model_summary.txt`                    | Thông số model + log từng epoch                      |
| `training_log.json`                    | Dữ liệu JSON (vẽ lại biểu đồ nếu cần)                |

## 6. Giải thích các thông số chính

| Thông số        | Ý nghĩa                                                     | Midterm            | Final              |
| --------------- | ----------------------------------------------------------- | ------------------ | ------------------ |
| Depth           | Tổng số layer → (depth-4)/6 block mỗi group                 | 16 (2 block/group) | 28 (4 block/group) |
| Widen factor    | Hệ số nhân channel                                          | 2 (channels ×2)    | 4 (channels ×4)    |
| Batch size      | Số ảnh xử lý mỗi lần cập nhật weights                       | 8 hoặc 128         | 128                |
| Epochs          | Số lần duyệt toàn bộ dataset                                | 100                | 200                |
| Learning rate   | Tốc độ cập nhật weights, giảm dần theo cosine               | 0.1 → 0            | 0.1 → 0            |
| Label smoothing | Làm mềm label (0→0.05, 1→0.95) → giảm overconfidence        | 0.1                | 0.1                |
| Cutout          | Che random 1 vùng vuông → buộc model không phụ thuộc 1 vùng | IMG/4              | 16 hoặc 8          |
| Dropout         | Tắt random 30% neuron khi train → giảm overfit              | 0.3                | 0.3                |

## 7. Yêu cầu hệ thống

- Python 3.8+
- PyTorch 1.13+
- torchvision 0.14+
- GPU: NVIDIA T4 (Kaggle/Colab free tier)

## 8. Tham khảo

- [Wide Residual Networks (Zagoruyko & Komodakis, 2016)](https://arxiv.org/abs/1605.07146)
- [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)
- [AutoAugment (Cubuk et al., 2019)](https://arxiv.org/abs/1805.09501)
- [Cutout (DeVries & Taylor, 2017)](https://arxiv.org/abs/1708.04552)
- [Animals-10 Dataset (Corrado, Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
