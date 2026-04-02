# WideResNet-16-2 + SE Attention — Animals-5 (Midterm)

Custom CNN model trained on a custom 5-class animal dataset at **224×224** and **32×32** resolutions.

# Author: Van Cong Toan, Phan Dinh Trung

## Dataset

| Thông số | Giá trị |
| -------- | ------- |
| Source | [Animals-10 (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10) — chọn 5 class |
| Classes | butterfly, cat, chicken, dog, horse |
| Train | 500 ảnh/class × 5 = 2,500 ảnh |
| Test | 50 ảnh/class × 5 = 250 ảnh |
| Dedup | MD5 hash để loại ảnh trùng |

## Kiến trúc model

```
Input (IMG_SIZE × IMG_SIZE × 3)
  → Stem Conv 3×3 (S=1, P=1) → IMG_SIZE × IMG_SIZE × 16
  → Group 1: WideResBlock ×2 (S=1)          → IMG_SIZE × IMG_SIZE × 32
  → Group 2: WideResBlock ×2 (first S=2)    → IMG_SIZE/2 × IMG_SIZE/2 × 64
  → Group 3: WideResBlock ×2 (first S=2)    → IMG_SIZE/4 × IMG_SIZE/4 × 128
  → BN → ReLU → AdaptiveAvgPool(1×1) → Flatten → FC → 5 classes
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

| Thông số | Giá trị |
| -------- | ------- |
| Depth | 16 layers |
| Widen factor | 2 |
| Parameters | ~0.7M |
| Blocks | 3 groups × 2 blocks = 6 blocks |
| Channels | 16 → 32 → 64 → 128 |

### Tại sao chọn WRN-16-2 thay vì WRN-28-4?

Dataset chỉ có 2,500 ảnh train — rất nhỏ so với CIFAR (50,000 ảnh). WRN-28-4 (5.9M params) overfit nặng trên dataset này (train 98% vs test 73%, gap 30%). Giảm depth 28→16 và widen 4→2 giúp giảm params từ 5.9M → 0.7M, giảm gap xuống ~17% và tăng best test accuracy từ 73% → 79%.

Model vẫn giữ nguyên kiến trúc WRN + SE Attention như bài cuối kỳ (WRN-28-4 trên CIFAR-10/100), chỉ thay đổi depth và widen factor.

## Cấu trúc thư mục

```
├── WRN_Animals_Kaggle.ipynb   # Notebook Kaggle — train cả 224×224 và 32×32
├── README.md
```

## Cách chạy

### Trên Kaggle

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **File → Upload Notebook**
2. Chọn file `WRN_Animals_Kaggle.ipynb`
3. **+ Add Input** → search `animals-5-custom-train` → Add → tương tự `animals-5-custom-test`
4. **Settings** → **Accelerator** → chọn **GPU T4 ×2**
5. Bấm **Run All**
6. Chờ ~30-60 phút (224×224) hoặc ~3-5 phút (32×32)

### Chạy 2 lần

Đề yêu cầu train cùng model trên 2 kích thước ảnh:

1. **Lần 1:** `IMG_SIZE = 224` → Restart & Run All → lấy kết quả
2. **Lần 2:** đổi `IMG_SIZE = 32` → Restart & Run All → lấy kết quả

Output lưu ở 2 thư mục riêng, không đè nhau.

## Training recipe

| Thành phần | Cấu hình |
| ---------- | -------- |
| Optimizer | SGD (lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True) |
| Scheduler | CosineAnnealingLR (100 epochs) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Augmentation | Resize + RandomCrop + HorizontalFlip + ColorJitter + Cutout |
| Cutout size | IMG_SIZE // 4 (224→56, 32→8) |
| Dropout | 0.3 |
| Batch size | 8 (224×224), 128 (32×32) |
| Epochs | 100 |
| Normalization | ImageNet mean/std |

### Tại sao chọn các giá trị này?

- **ImageNet mean/std** thay vì CIFAR mean/std: ảnh Animals là ảnh thực tế (giống ImageNet), không phải ảnh 32×32 như CIFAR
- **CosineAnnealingLR**: learning rate giảm mượt, accuracy cuối cao hơn MultiStepLR
- **ColorJitter**: thêm augmentation màu sắc vì ảnh động vật có biến thể màu nhiều
- **Cutout tỉ lệ theo IMG_SIZE**: 224→56px, 32→8px — giữ tỉ lệ che ảnh nhất quán
- **Batch 8 cho 224×224**: WRN-16-2 + ảnh 224×224 vừa đủ RAM trên T4 (16GB)
- **Label smoothing 0.1**: ngăn model quá tự tin → giảm overfit trên dataset nhỏ

## Output files

| File | Mô tả |
| ---- | ----- |
| `best_model.pth` | Weights của model có accuracy cao nhất — dùng để verify |
| `last_checkpoint.pth` | Checkpoint để resume nếu bị disconnect |
| `charts_{IMG_SIZE}.png` | Biểu đồ Loss và Accuracy theo epoch |
| `model_summary.txt` | Thông số model + log accuracy mỗi epoch |
| `training_log.json` | Dữ liệu training dạng JSON |

## Kết quả

| Image Size | Best Test Accuracy | Train Acc (cuối) | Gap | Ghi chú |
| ---------- | ------------------ | ---------------- | --- | ------- |
| 32×32 | 79.60% | ~95% | ~17% | Overfit do dataset nhỏ (2500 ảnh) |
| 224×224 | TBD | TBD | TBD | Chờ kết quả |

### Phân tích overfit (32×32)

- Dataset chỉ 2,500 ảnh cho model 0.7M params → vẫn thừa capacity
- Ảnh 32×32 mất nhiều spatial detail → khó phân biệt các class tương tự
- Đã cải thiện từ WRN-28-4 (gap 30%, test 73%) → WRN-16-2 (gap 17%, test 79%)
- Gap ~17% là chấp nhận được cho dataset kích thước này

## Predict ảnh tự chọn

```python
import torch
import torchvision.transforms as T
from PIL import Image

CLASSES = ['butterfly', 'cat', 'chicken', 'dog', 'horse']

model = WideResNet(depth=16, widen_factor=2, num_classes=5)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

transform = T.Compose([
    T.Resize((32, 32)),  # hoặc (224, 224)
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

img = Image.open('your_image.jpg')
x = transform(img).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
    top5 = torch.topk(probs, 5)
    for i in range(5):
        print(f'{CLASSES[top5.indices[i]]}: {top5.values[i]*100:.1f}%')
```

## Yêu cầu

- Python 3.8+
- PyTorch 1.13+
- torchvision 0.14+
- GPU: NVIDIA T4 (Kaggle free tier)
- Thời gian train: ~30-60 phút (224×224), ~3-5 phút (32×32) trên T4

## So sánh với Final (CIFAR-10/100)

| | Midterm (Animals-5) | Final (CIFAR-10/100) |
| --- | --- | --- |
| Model | WRN-16-2 + SE | WRN-28-4 + SE |
| Parameters | ~0.7M | ~5.9M |
| Dataset size | 2,500 train | 50,000 train |
| Image size | 224×224 và 32×32 | 32×32 |
| Epochs | 100 | 200 |

## Tham khảo

- [Wide Residual Networks (Zagoruyko & Komodakis, 2016)](https://arxiv.org/abs/1605.07146)
- [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)
- [Animals-10 Dataset (Corrado, Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10)