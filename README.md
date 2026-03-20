# WideResNet-28-4 + SE Attention — CIFAR-10 / CIFAR-100

Custom CNN model đạt **~95-96% trên CIFAR-10** và **~79-81% trên CIFAR-100**.

# Author: Van Cong Toan

## Kiến trúc model

```
Input (32×32×3)
  → Stem Conv 3×3 (S=1, P=1) → 32×32×16
  → Group 1: WideResBlock ×4 (S=1)          → 32×32×64
  → Group 2: WideResBlock ×4 (first S=2)    → 16×16×128
  → Group 3: WideResBlock ×4 (first S=2)    → 8×8×256
  → BN → ReLU → AvgPool → Flatten → FC → 10 classes (hoặc 100)
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

| Thông số     | Giá trị                         |
| ------------ | ------------------------------- |
| Depth        | 28 layers                       |
| Widen factor | 4                               |
| Parameters   | ~5.9M                           |
| Blocks       | 3 groups × 4 blocks = 12 blocks |
| Channels     | 16 → 64 → 128 → 256             |

## Cấu trúc thư mục

```
├── WRN_CIFAR10.ipynb          # Notebook Kaggle — CIFAR-10
├── WRN_CIFAR100.ipynb         # Notebook Kaggle — CIFAR-100
├── Colab_CIFAR10.ipynb        # Notebook Colab — CIFAR-10 (lưu checkpoint lên Google Drive)
├── Colab_CIFAR100.ipynb       # Notebook Colab — CIFAR-100 (lưu checkpoint lên Google Drive)
├── WRN_Architecture_v3.html   # Sơ đồ kiến trúc model (mở bằng trình duyệt)
└── README.md
```

## Cách chạy

### Trên Kaggle

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **File → Upload Notebook**
2. Chọn file `WRN_CIFAR10.ipynb` hoặc `WRN_CIFAR100.ipynb`
3. **Settings** → **Accelerator** → chọn **GPU T4 ×2** (cần verify phone trước)
4. **Settings** → **Internet** → bật **ON**
5. Bấm **Run All**
6. Chờ ~2-3 tiếng → kết quả ở tab **Output**

> ⚠️ Kaggle xoá file khi session kết thúc. Đừng bấm Stop Session. Đóng tab thì session vẫn chạy ngầm (tối đa 9h).

### Trên Google Colab

1. Vào [colab.research.google.com](https://colab.research.google.com) → **File → Upload Notebook**
2. Chọn file `Colab_CIFAR10.ipynb` hoặc `Colab_CIFAR100.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**
4. Bấm **Run All**
5. Lần đầu sẽ yêu cầu **cho phép truy cập Google Drive** → bấm Allow
6. Chờ ~2-3 tiếng → kết quả lưu trong Google Drive

> ✅ Bản Colab lưu checkpoint lên Google Drive. Nếu bị disconnect → Run All lại → tự resume từ epoch cuối.

## Training recipe

| Thành phần   | Cấu hình                                                               |
| ------------ | ---------------------------------------------------------------------- |
| Optimizer    | SGD (lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)           |
| Scheduler    | CosineAnnealingLR (200 epochs)                                         |
| Loss         | CrossEntropyLoss (label_smoothing=0.1)                                 |
| Augmentation | RandomCrop(32, pad=4) + HorizontalFlip + AutoAugment(CIFAR10) + Cutout |
| Cutout size  | 16 (CIFAR-10), 8 (CIFAR-100)                                           |
| Dropout      | 0.3                                                                    |
| Batch size   | 128                                                                    |
| Epochs       | 200                                                                    |

### Tại sao chọn các giá trị này?

- **CosineAnnealingLR** thay vì MultiStepLR: learning rate giảm mượt hơn, accuracy cuối cao hơn ~0.5%
- **AutoAugment + Cutout**: augmentation mạnh nhất cho CIFAR, tăng ~2-3% accuracy
- **Label smoothing 0.1**: ngăn model quá tự tin → generalize tốt hơn
- **Nesterov momentum**: hội tụ nhanh hơn SGD thường
- **Cutout 16 (CIFAR-10) vs 8 (CIFAR-100)**: CIFAR-100 có 100 class, cần giữ nhiều thông tin hơn nên cutout nhỏ hơn

## Output files

Sau khi train xong, các file được tạo ra:

| File                  | Mô tả                                                                |
| --------------------- | -------------------------------------------------------------------- |
| `best_model.pth`      | Weights của model có accuracy cao nhất — dùng để predict ảnh mới     |
| `last_checkpoint.pth` | Checkpoint để resume nếu bị disconnect (chỉ bản Colab lưu lên Drive) |
| `training_curves.png` | Biểu đồ Loss và Accuracy theo epoch                                  |
| `model_summary.txt`   | Thông số model + log accuracy mỗi epoch                              |
| `training_log.json`   | Dữ liệu training dạng JSON (để vẽ lại biểu đồ nếu cần)               |

## Kết quả kỳ vọng

| Dataset   | Accuracy |
| --------- | -------- |
| CIFAR-10  | 95-96%   |
| CIFAR-100 | 79-81%   |

> Accuracy tăng mạnh nhất ở epoch 150-200 khi learning rate giảm rất thấp. Đừng lo nếu epoch 50-100 còn thấp.

### Tại sao model này mạnh?

1. **Skip connection** — gradient không bị mất khi train model sâu
2. **BatchNorm** — chuẩn hoá dữ liệu giữa các layer, train ổn định hơn
3. **SE Attention** — tự học channel nào quan trọng cho ảnh hiện tại
4. **Augmentation mạnh** — AutoAugment + Cutout buộc model học feature tổng quát

## Predict ảnh tự chọn

Sau khi train xong, load model và predict:

```python
import torch
import torchvision.transforms as T
from PIL import Image

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model = WideResNet(depth=28, widen_factor=4, num_classes=10)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# Preprocess
transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
])

# Predict
img = Image.open('your_image.jpg')
x = transform(img).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
    top5 = torch.topk(probs, 5)
    for i in range(5):
        print(f'{CLASSES[top5.indices[i]]}: {top5.values[i]*100:.1f}%')
```

> ⚠️ CIFAR-10 chỉ nhận diện 10 loại: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Đưa ảnh ngoài 10 loại này sẽ cho kết quả sai.

## Yêu cầu

- Python 3.8+
- PyTorch 1.13+
- torchvision 0.14+
- GPU: NVIDIA T4 (Kaggle/Colab free tier) hoặc tương đương
- Thời gian train: ~2-3 giờ / dataset trên T4

## Tham khảo

- [Wide Residual Networks (Zagoruyko & Komodakis, 2016)](https://arxiv.org/abs/1605.07146)
- [Squeeze-and-Excitation Networks (Hu et al., 2018)](https://arxiv.org/abs/1709.01507)
- [AutoAugment (Cubuk et al., 2019)](https://arxiv.org/abs/1805.09501)
- [Cutout (DeVries & Taylor, 2017)](https://arxiv.org/abs/1708.04552)
