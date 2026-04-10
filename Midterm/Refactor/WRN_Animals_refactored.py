"""
WideResNet-16-2 + SE Attention — Animals-5 (Custom Dataset)
Refactored: mỗi block liệt kê tường minh, dễ sửa S/P/kernel/channels

SỬA IMG_SIZE rồi chạy lại toàn bộ:
  - Lần 1: IMG_SIZE = 224
  - Lần 2: IMG_SIZE = 32
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ============================================================
# CONFIG — SỬA Ở ĐÂY
# ============================================================
IMG_SIZE      = 224       # ← ĐỔI: 224 hoặc 32
NUM_CLASSES   = 5
BATCH_SIZE    = 32 if IMG_SIZE == 224 else 128
EPOCHS        = 100
LR            = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
DROPOUT       = 0.3
LABEL_SMOOTH  = 0.1

TRAIN_DIR = '/kaggle/input/datasets/nyantony/animals-5-custom-train/train'
TEST_DIR  = '/kaggle/input/datasets/nyantony/animals-5-custom-test/test'
OUT_DIR   = f'/kaggle/working/animals_wrn_{IMG_SIZE}'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device} | Image size: {IMG_SIZE}×{IMG_SIZE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

for label, path in [('Train', TRAIN_DIR), ('Test', TEST_DIR)]:
    classes = sorted(os.listdir(path))
    counts = {c: len(os.listdir(os.path.join(path, c))) for c in classes}
    print(f'{label}: {counts}')


# ============================================================
# SE BLOCK — giống file CIFAR-10, copy y nguyên
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


# ============================================================
# WIDE RESIDUAL BLOCK — giống file CIFAR-10
# ============================================================
class WideResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, reduction=16)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                      kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = self.drop(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = self.se(out)
        return out + self.shortcut(x)


# ============================================================
# WIDRESNET-16-2 — LIỆT KÊ TƯỜNG MINH
# ============================================================
# depth=16, widen_factor=2 → n = (16-4)/6 = 2 blocks/group
# Channels: [16, 32, 64, 128] = [16, 16*2, 32*2, 64*2]
#
# VỚI IMG_SIZE=32:
#   Input 32×32×3
#   → conv1(3→16)                        → 32×32×16
#   → Group1: 2 blocks (16→32, 32→32)    → 32×32×32
#   → Group2: 2 blocks (32→64 s=2, 64)   → 16×16×64
#   → Group3: 2 blocks (64→128 s=2, 128) → 8×8×128
#   → BN→ReLU→AvgPool→FC(128→5)
#
# VỚI IMG_SIZE=224:
#   Input 224×224×3
#   → conv1                              → 224×224×16
#   → Group1                             → 224×224×32
#   → Group2 (s=2)                       → 112×112×64
#   → Group3 (s=2)                       → 56×56×128
#   → AvgPool adaptive (1×1)             → 1×1×128
#   → FC(128→5)
class WideResNet(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()

        # === Lớp đầu vào ===
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1, bias=False)

        # === GROUP 1: giữ nguyên kích thước, 16→32 kênh ===
        self.block1_1 = WideResBlock(in_ch=16, out_ch=32, stride=1, dropout=dropout)  # có shortcut conv1x1 (16≠32)
        self.block1_2 = WideResBlock(in_ch=32, out_ch=32, stride=1, dropout=dropout)  # identity shortcut

        # === GROUP 2: giảm ÷2, 32→64 kênh ===
        self.block2_1 = WideResBlock(in_ch=32, out_ch=64, stride=2, dropout=dropout)  # shortcut conv1x1 s=2
        self.block2_2 = WideResBlock(in_ch=64, out_ch=64, stride=1, dropout=dropout)  # identity

        # === GROUP 3: giảm ÷2, 64→128 kênh ===
        self.block3_1 = WideResBlock(in_ch=64,  out_ch=128, stride=2, dropout=dropout) # shortcut conv1x1 s=2
        self.block3_2 = WideResBlock(in_ch=128, out_ch=128, stride=1, dropout=dropout) # identity

        # === Đầu ra ===
        self.bn_final = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128, num_classes)   # 128 → 5

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 3, IMG_SIZE, IMG_SIZE)
        x = self.conv1(x)          # → (B, 16, H, W)

        # Group 1 — stride=1, kích thước giữ nguyên
        x = self.block1_1(x)       # → (B, 32, H, W)
        x = self.block1_2(x)       # → (B, 32, H, W)

        # Group 2 — stride=2 ở block đầu
        x = self.block2_1(x)       # → (B, 64, H/2, W/2)
        x = self.block2_2(x)       # → (B, 64, H/2, W/2)

        # Group 3 — stride=2 ở block đầu
        x = self.block3_1(x)       # → (B, 128, H/4, W/4)
        x = self.block3_2(x)       # → (B, 128, H/4, W/4)

        # Head
        x = F.relu(self.bn_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)   # → (B, 128, 1, 1)  bất kể H/W
        x = torch.flatten(x, 1)           # → (B, 128)
        x = self.fc(x)                     # → (B, 5)
        return x


# ============================================================
# DATA — Animals-5
# ============================================================
class Cutout:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, y - self.size//2), min(h, y + self.size//2)
        x1, x2 = max(0, x - self.size//2), min(w, x + self.size//2)
        img[:, y1:y2, x1:x2] = 0.0
        return img

MEAN = (0.485, 0.456, 0.406)    # ImageNet mean (ảnh thực tế)
STD  = (0.229, 0.224, 0.225)

cutout_size  = IMG_SIZE // 4     # 224→56, 32→8
crop_padding = IMG_SIZE // 8     # 224→28, 32→4

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomCrop(IMG_SIZE, padding=crop_padding),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    Cutout(cutout_size),
])
test_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

trainset = ImageFolder(TRAIN_DIR, transform=train_transform)
testset  = ImageFolder(TEST_DIR,  transform=test_transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f'Classes: {trainset.classes}')
print(f'Train: {len(trainset)} | Test: {len(testset)}')


# ============================================================
# MODEL
# ============================================================
model = WideResNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {num_params:,}')


# ============================================================
# TRAINING
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

CKPT_PATH = os.path.join(OUT_DIR, 'last_checkpoint.pth')
start_epoch = 0
best_acc = 0.0
history = {'epoch':[], 'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'lr':[]}

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_acc']
    history = ckpt['history']
    print(f'Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%')
else:
    print('Training from scratch.')


def train_one_epoch(loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, 100.0 * correct / total


print(f'Starting training — epochs {start_epoch+1} to {EPOCHS}')
t0 = time.time()

for epoch in range(start_epoch, EPOCHS):
    lr_now = optimizer.param_groups[0]['lr']
    train_loss, train_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(test_loader)
    scheduler.step()

    history['epoch'].append(epoch + 1)
    history['train_loss'].append(round(train_loss, 5))
    history['train_acc'].append(round(train_acc, 2))
    history['val_loss'].append(round(val_loss, 5))
    history['val_acc'].append(round(val_acc, 2))
    history['lr'].append(round(lr_now, 6))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'best_model.pth'))

    torch.save({
        'epoch': epoch, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'best_acc': best_acc, 'history': history,
    }, CKPT_PATH)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        elapsed = time.time() - t0
        print(f'Epoch {epoch+1:>3d}/{EPOCHS} | '
              f'train_loss={train_loss:.4f} acc={train_acc:.1f}% | '
              f'val_loss={val_loss:.4f} acc={val_acc:.1f}% | '
              f'best={best_acc:.2f}% | lr={lr_now:.6f} | {elapsed/60:.1f}min')

with open(os.path.join(OUT_DIR, 'training_log.json'), 'w') as f:
    json.dump(history, f, indent=2)
print(f'\nDone! Best test accuracy: {best_acc:.2f}%')
print(f'Total time: {(time.time()-t0)/60:.1f} min')


# ============================================================
# PLOT
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['epoch'], history['train_acc'], label='Train Acc')
ax1.plot(history['epoch'], history['val_acc'],   label='Test Acc')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Acc (%)'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_title(f'Accuracy — WRN-16-2 on Animals ({IMG_SIZE}×{IMG_SIZE})')
ax2.plot(history['epoch'], history['train_loss'], label='Train Loss')
ax2.plot(history['epoch'], history['val_loss'],   label='Test Loss')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_title(f'Loss — WRN-16-2 on Animals ({IMG_SIZE}×{IMG_SIZE})')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f'charts_{IMG_SIZE}.png'), dpi=150)
plt.show()
print(f'Best test accuracy ({IMG_SIZE}×{IMG_SIZE}): {best_acc:.2f}%')
