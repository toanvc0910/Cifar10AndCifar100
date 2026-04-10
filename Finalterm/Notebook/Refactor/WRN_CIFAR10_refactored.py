"""
WideResNet-28-4 + SE Attention — CIFAR-10
Refactored: mỗi block liệt kê tường minh, dễ sửa S/P/kernel/channels
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# CONFIG — SỬA Ở ĐÂY
# ============================================================
DATASET       = 'cifar10'
NUM_CLASSES   = 10
BATCH_SIZE    = 128
EPOCHS        = 200
LR            = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
DROPOUT       = 0.3
CUTOUT_SIZE   = 16
LABEL_SMOOTH  = 0.1

OUT_DIR = '/kaggle/working/cifar10_wrn'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# ============================================================
# SE BLOCK — Squeeze-and-Excitation
# ============================================================
# Input:  (B, C, H, W)
# Output: (B, C, H, W) — cùng shape, nhưng mỗi kênh được nhân trọng số [0,1]
#
# Luồng: AvgPool → Flatten → Linear(C → C//r) → ReLU → Linear(C//r → C) → Sigmoid → Reshape → Nhân
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # channels=128, reduction=16 → fc1: 128→8, fc2: 8→128
        self.fc1 = nn.Linear(channels, channels // reduction)   # Nén
        self.fc2 = nn.Linear(channels // reduction, channels)    # Mở rộng

    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze: (B,C,H,W) → (B,C,1,1) → (B,C)
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation: (B,C) → (B,C//r) → (B,C)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        # Scale: (B,C) → (B,C,1,1) rồi nhân element-wise
        w = w.view(b, c, 1, 1)
        return x * w


# ============================================================
# WIDE RESIDUAL BLOCK — Pre-activation
# ============================================================
# Luồng chính:  BN→ReLU→Conv3x3(stride)→Dropout→BN→ReLU→Conv3x3(s=1)→SE
# Shortcut:     Conv1x1(stride) nếu cần khớp shape, hoặc identity
# Output:       nhánh_chính + shortcut
class WideResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.3):
        super().__init__()
        # --- Nhánh chính ---
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop  = nn.Dropout(dropout)
        self.se    = SEBlock(out_ch, reduction=16)

        # --- Shortcut ---
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                      kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()  # Không làm gì

    def forward(self, x):
        # Nhánh chính (pre-activation)
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)           # Conv 3x3, có thể stride=2
        out = self.drop(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)           # Conv 3x3, stride=1
        out = self.se(out)              # SE attention
        # Cộng shortcut
        return out + self.shortcut(x)


# ============================================================
# WIDRESNET-28-4 — LIỆT KÊ TƯỜNG MINH TỪNG BLOCK
# ============================================================
# depth=28, widen_factor=4 → n = (28-4)/6 = 4 blocks/group
# Channels: [16, 64, 128, 256] = [16, 16*4, 32*4, 64*4]
#
# TỔNG QUAN:
#   Input 32×32×3
#   → conv1(3→16, k=3, s=1, p=1)                    → 32×32×16
#   → Group1: 4 blocks (16→64, rồi 64→64 ×3, s=1)   → 32×32×64
#   → Group2: 4 blocks (64→128 s=2, rồi 128→128 ×3) → 16×16×128
#   → Group3: 4 blocks (128→256 s=2, rồi 256→256 ×3)→ 8×8×256
#   → BN→ReLU→AvgPool→Flatten→Linear(256→10)
class WideResNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        # === Lớp đầu vào ===
        # 32×32×3 → 32×32×16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1, bias=False)

        # === GROUP 1: giữ nguyên 32×32, mở rộng 16→64 kênh ===
        self.block1_1 = WideResBlock(in_ch=16,  out_ch=64, stride=1, dropout=dropout)  # 32×32×16  → 32×32×64  (có shortcut conv1x1)
        self.block1_2 = WideResBlock(in_ch=64,  out_ch=64, stride=1, dropout=dropout)  # 32×32×64  → 32×32×64  (identity shortcut)
        self.block1_3 = WideResBlock(in_ch=64,  out_ch=64, stride=1, dropout=dropout)  # 32×32×64  → 32×32×64
        self.block1_4 = WideResBlock(in_ch=64,  out_ch=64, stride=1, dropout=dropout)  # 32×32×64  → 32×32×64

        # === GROUP 2: giảm 32→16, mở rộng 64→128 kênh ===
        self.block2_1 = WideResBlock(in_ch=64,  out_ch=128, stride=2, dropout=dropout) # 32×32×64  → 16×16×128 (shortcut conv1x1 s=2)
        self.block2_2 = WideResBlock(in_ch=128, out_ch=128, stride=1, dropout=dropout) # 16×16×128 → 16×16×128 (identity)
        self.block2_3 = WideResBlock(in_ch=128, out_ch=128, stride=1, dropout=dropout) # 16×16×128 → 16×16×128
        self.block2_4 = WideResBlock(in_ch=128, out_ch=128, stride=1, dropout=dropout) # 16×16×128 → 16×16×128

        # === GROUP 3: giảm 16→8, mở rộng 128→256 kênh ===
        self.block3_1 = WideResBlock(in_ch=128, out_ch=256, stride=2, dropout=dropout) # 16×16×128 → 8×8×256   (shortcut conv1x1 s=2)
        self.block3_2 = WideResBlock(in_ch=256, out_ch=256, stride=1, dropout=dropout) # 8×8×256   → 8×8×256   (identity)
        self.block3_3 = WideResBlock(in_ch=256, out_ch=256, stride=1, dropout=dropout) # 8×8×256   → 8×8×256
        self.block3_4 = WideResBlock(in_ch=256, out_ch=256, stride=1, dropout=dropout) # 8×8×256   → 8×8×256

        # === Đầu ra ===
        self.bn_final = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, num_classes)   # 256 → 10

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
        # x: (B, 3, 32, 32)
        x = self.conv1(x)          # → (B, 16, 32, 32)

        # Group 1
        x = self.block1_1(x)       # → (B, 64, 32, 32)
        x = self.block1_2(x)       # → (B, 64, 32, 32)
        x = self.block1_3(x)       # → (B, 64, 32, 32)
        x = self.block1_4(x)       # → (B, 64, 32, 32)

        # Group 2
        x = self.block2_1(x)       # → (B, 128, 16, 16)  ← stride=2 giảm nửa
        x = self.block2_2(x)       # → (B, 128, 16, 16)
        x = self.block2_3(x)       # → (B, 128, 16, 16)
        x = self.block2_4(x)       # → (B, 128, 16, 16)

        # Group 3
        x = self.block3_1(x)       # → (B, 256, 8, 8)    ← stride=2 giảm nửa
        x = self.block3_2(x)       # → (B, 256, 8, 8)
        x = self.block3_3(x)       # → (B, 256, 8, 8)
        x = self.block3_4(x)       # → (B, 256, 8, 8)

        # Head
        x = F.relu(self.bn_final(x), inplace=True)  # → (B, 256, 8, 8)
        x = F.adaptive_avg_pool2d(x, 1)             # → (B, 256, 1, 1)
        x = torch.flatten(x, 1)                     # → (B, 256)
        x = self.fc(x)                               # → (B, 10)
        return x


# ============================================================
# DATA — CIFAR-10
# ============================================================
class Cutout:
    """Che ngẫu nhiên 1 vùng vuông trên ảnh → buộc model không dựa 1 chỗ."""
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        y, x = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, y - self.size//2), min(h, y + self.size//2)
        x1, x2 = max(0, x - self.size//2), min(w, x + self.size//2)
        img[:, y1:y2, x1:x2] = 0.0
        return img

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2471, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    Cutout(CUTOUT_SIZE),
])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

trainset = torchvision.datasets.CIFAR10(root='/kaggle/working/data', train=True,  download=True, transform=train_transform)
testset  = torchvision.datasets.CIFAR10(root='/kaggle/working/data', train=False, download=True, transform=test_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f'Train: {len(trainset)} | Test: {len(testset)}')


# ============================================================
# MODEL
# ============================================================
model = WideResNet(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {num_params:,}')
print(model)


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
    ckpt = torch.load(CKPT_PATH, map_location=device)
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

    elapsed = time.time() - t0
    eta = elapsed / (epoch - start_epoch + 1) * (EPOCHS - epoch - 1) if epoch > start_epoch else 0
    print(f'Epoch {epoch+1:>3d}/{EPOCHS} | lr={lr_now:.5f} | '
          f'train_loss={train_loss:.4f} acc={train_acc:.2f}% | '
          f'val_loss={val_loss:.4f} acc={val_acc:.2f}% | '
          f'best={best_acc:.2f}% | ETA={eta/60:.0f}min')

total_time = time.time() - t0
print(f'\nDone in {total_time/3600:.1f}h. Best val accuracy: {best_acc:.2f}%')

with open(os.path.join(OUT_DIR, 'training_log.json'), 'w') as f:
    json.dump(history, f, indent=2)


# ============================================================
# PLOT
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['epoch'], history['train_loss'], label='Train Loss')
ax1.plot(history['epoch'], history['val_loss'],   label='Val Loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(history['epoch'], history['train_acc'], label='Train Acc')
ax2.plot(history['epoch'], history['val_acc'],   label='Val Acc')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Acc (%)'); ax2.set_title(f'Accuracy — Best: {best_acc:.2f}%'); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=150)
plt.show()
